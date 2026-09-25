"""Small Transformers video adapter shared by smoke, teachers and student."""
import inspect
import json
import re
import time
import functools
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


@functools.lru_cache(maxsize=64)
def decode(path, frames=8, start=0., duration=None, pixels=224, crop_right=False):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        meta = next(reader)
        fps, length = float(meta['fps']), float(meta['duration'])
        start = min(start, max(0., length - 1 / fps))
        end = min(length, start + duration) if duration else length
        indices = np.linspace(int(start * fps), max(int(start * fps), int(end * fps) - 1), frames).astype(int)
        wanted = set(indices.tolist())
        found = {}
        for index, frame in enumerate(reader):
            if index in wanted:
                im = Image.frombytes('RGB', meta['size'], frame)
                if crop_right:
                    if im.width != 2 * im.height:
                        raise ValueError('Unexpected source PVS layout; refuse unverified crop')
                    im = im.crop((im.width // 2, 0, im.width, im.height))
                im.thumbnail((pixels, pixels), Image.Resampling.LANCZOS)
                found[index] = np.asarray(im)
            if index >= indices[-1]:
                break
        if not found:
            raise RuntimeError('No decodable frames')
        array = np.stack([found.get(int(i), found[max(found)]) for i in indices])
        effective_fps = (frames - 1) / max((indices[-1] - indices[0]) / fps, 1 / fps)
        return array, effective_fps, {'source_fps': fps, 'duration': length,
                 'original_size':meta['size'],
                 'crop_xyxy':([meta['size'][0]//2,0,*meta['size']] if crop_right else [0,0,*meta['size']]),
                 'timestamps': (indices / fps).tolist(), 'shape': list(array.shape)}
    finally:
        reader.close()


class VLM:
    def __init__(self, model_path, lora=False, max_memory_gib=2.9, source_crop_right=False):
        self.source_crop_right = source_crop_right
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=56*56, max_pixels=224*224)
        n = torch.cuda.device_count()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation='eager',
            device_map='auto', max_memory={**{i: f'{max_memory_gib}GiB' for i in range(n)}, 'cpu': '24GiB'})
        if any(str(d) in ('cpu', 'disk') for d in self.model.hf_device_map.values()):
            raise RuntimeError(f'Model offload required: {self.model.hf_device_map}')
        if lora:
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.,
                target_modules=r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)',
                task_type='CAUSAL_LM'))
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.eval()
        self.timings = []

    def inputs(self, paths, prompt, local=False):
        content, videos, fps, traces = [], [], [], []
        for index, path in enumerate(paths):
            content.append({'type': 'text', 'text': f'Video {chr(65+index)}: global overview.'})
            views = [(0., None)] + ([(2., 1.)] if local else [])
            for start, duration in views:
                array, frequency, trace = decode(path, start=start, duration=duration, crop_right=self.source_crop_right)
                videos.append(array); fps.append(frequency); traces.append(trace)
                content.append({'type':'video'})
        content.append({'type': 'text', 'text': prompt})
        text = self.processor.apply_chat_template([{'role':'user','content':content}],
                         tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=videos, fps=fps, padding=True, return_tensors='pt')
        device = self.model.get_input_embeddings().weight.device
        return inputs.to(device), traces

    def generate(self, inputs, max_tokens=48, temperature=0.8):
        self.model.eval()
        start = time.monotonic()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature>0,
                   temperature=temperature if temperature>0 else None, top_p=0.95 if temperature>0 else None,
                   use_cache=True, pad_token_id=self.processor.tokenizer.eos_token_id)
        tail = outputs[:, inputs['input_ids'].shape[1]:]
        text = self.processor.batch_decode(tail, skip_special_tokens=True)[0]
        self.timings.append({'seconds':time.monotonic()-start,'input_tokens':inputs['input_ids'].shape[1],
                             'output_tokens':tail.shape[1]})
        return text, tail

    def logprobs(self, inputs, completion):
        self.model.train(torch.is_grad_enabled())
        joined = dict(inputs)
        joined['input_ids'] = torch.cat([inputs['input_ids'], completion], dim=1)
        joined['attention_mask'] = torch.ones_like(joined['input_ids'])
        base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        if 'logits_to_keep' in inspect.signature(base.forward).parameters:
            joined['logits_to_keep'] = completion.shape[1] + 1
        logits = self.model(**joined, use_cache=False).logits
        logits = logits[:, -(completion.shape[1]+1):-1].float()
        return logits.log_softmax(-1).gather(-1, completion.unsqueeze(-1)).squeeze(-1)


SCORE_PROMPT = ('Rate overall human-perceived visual quality of this rendered video, including spatial artifacts '
                'and temporal stability. Ignore artistic appeal and object identity. '
                'Return only JSON {"score": number}, using 1.00 (very poor) to 5.00 (excellent).')


def parse_score(text):
    match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not match or not 1 <= float(match[1]) <= 5:
        raise ValueError(f'Invalid score output: {text!r}')
    return float(match[1])
