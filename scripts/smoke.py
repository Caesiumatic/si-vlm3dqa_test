"""One real-video pair: teacher, rollouts, policy update and adapter reload."""
import argparse
import gc
import json
import pathlib
import time
import numpy as np
import torch
from reward import fidelity, advantages, check
from vlm import VLM, SCORE_PROMPT, parse_score

p = argparse.ArgumentParser()
p.add_argument('--model', default='models/Qwen2.5-VL-3B-Instruct')
p.add_argument('--videos', nargs=2, required=True)
p.add_argument('--output', default='runs/smoke')
args = p.parse_args()
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(17)
np.random.seed(17)
result = {'status': 'running', 'started': time.time(), 'formula_checks': check(),
          'gpus': [{'name': torch.cuda.get_device_name(i), 'capability': torch.cuda.get_device_capability(i)}
                   for i in range(torch.cuda.device_count())]}


def record():
    (out / 'result.json').write_text(json.dumps(result, indent=2))


record()
try:
    t = time.monotonic()
    runner = VLM(args.model, lora=True, source_crop_right=True)
    result['load_seconds'] = time.monotonic() - t
    result['device_map'] = {k: str(v) for k, v in runner.model.hf_device_map.items()}
    print('loaded', result['load_seconds'], result['device_map'], flush=True)
    teacher_inputs, traces = runner.inputs(args.videos,
        'Compare the overall perceptual quality of videos A and B. Assess visible spatial defects '
        'and temporal stability. Briefly identify visible evidence and finish with JSON '
        '{"preference":"A"} or {"preference":"B"} or {"preference":"near_equal"}.', local=True)
    teacher, _ = runner.generate(teacher_inputs, max_tokens=128, temperature=0.6)
    result['teacher'] = {'text': teacher, 'observations': traces}
    print('teacher', teacher, flush=True)
    record()
    del teacher_inputs
    import re
    label = re.search(r'"preference"\s*:\s*"(A|B|near_equal)"', teacher)
    if label is None:
        raise RuntimeError('Teacher output unparseable; no invented supervision')
    target = {'A': 1., 'B': 0., 'near_equal': .5}[label[1]]
    groups = []
    for path in args.videos:
        inputs, trace = runner.inputs([path], SCORE_PROMPT, local=True)
        group = {'inputs': inputs, 'scores': [], 'completions': [], 'texts': [], 'observations': trace}
        for _ in range(4):
            text, completion = runner.generate(inputs, max_tokens=40, temperature=1.)
            print('rollout', text, flush=True)
            group['scores'].append(parse_score(text))
            group['texts'].append(text)
            group['completions'].append(completion)
        groups.append(group)
    rewards = [fidelity(groups[0]['scores'], groups[1]['scores'], target),
               fidelity(groups[1]['scores'], groups[0]['scores'], 1-target)]
    advs = [advantages(r) for r in rewards]
    result['groups'] = [{'scores':g['scores'], 'texts':g['texts'], 'rewards':r.tolist(),
                        'advantages':a.tolist(), 'observations':g['observations']}
                       for g, r, a in zip(groups, rewards, advs)]
    record()
    if not any(np.any(a != 0) for a in advs):
        raise RuntimeError('Both real rollout groups have zero reward variance')
    trainable = [(name, param) for name, param in runner.model.named_parameters() if param.requires_grad]
    before = {name: param.detach().cpu().clone() for name, param in trainable}
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=1e-5)
    t = time.monotonic()
    optimizer.zero_grad()
    losses = []
    for group, adv in zip(groups, advs):
        for completion, a in zip(group['completions'], adv):
            if a == 0:
                continue
            with torch.no_grad():
                old = runner.logprobs(group['inputs'], completion).detach()
            logp = runner.logprobs(group['inputs'], completion)
            ratio = (logp - old).exp()
            # First on-policy GRPO step; base reference equals old policy and KL gradient is zero here.
            policy = torch.minimum(ratio * float(a), ratio.clamp(.8, 1.2) * float(a))
            loss = -policy.mean() / 8
            loss.backward()
            losses.append(float(loss.detach()))
            del logp, ratio, policy, loss, old
    grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.))
    if not np.isfinite(grad_norm) or grad_norm == 0:
        raise RuntimeError(f'Invalid gradient norm: {grad_norm}')
    optimizer.step()
    delta = sum(float((param.detach().cpu() - before[name]).square().sum()) for name, param in trainable)
    if not np.isfinite(delta) or delta <= 0:
        raise RuntimeError(f'No finite parameter update: {delta}')
    result['update'] = {'seconds':time.monotonic()-t, 'grad_norm':grad_norm,
                        'parameter_delta_squared':delta,'loss_terms':losses}
    checkpoint = out / 'adapter'
    runner.model.save_pretrained(checkpoint)
    runner.processor.save_pretrained(checkpoint)
    with torch.no_grad():
        expected = runner.logprobs(groups[0]['inputs'], groups[0]['completions'][0]).cpu()
    del optimizer, trainable, before
    base = runner.model.unload()
    from peft import PeftModel
    runner.model = PeftModel.from_pretrained(base, checkpoint).eval()
    gc.collect()
    with torch.no_grad():
        observed = runner.logprobs(groups[0]['inputs'], groups[0]['completions'][0]).cpu()
    reload_error = float((expected-observed).abs().max())
    if reload_error > 0.001:
        raise RuntimeError(f'Checkpoint reload mismatch: {reload_error}')
    result['reload_max_abs_error'] = reload_error
    result['checkpoint_bytes'] = sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file())
    result['status'] = 'passed'
except Exception as error:
    result['status'] = 'failed'
    result['error'] = repr(error)
    raise
finally:
    result['elapsed_seconds'] = time.time()-result['started']
    result['max_memory_allocated'] = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    result['max_memory_reserved'] = [torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())]
    if 'runner' in globals():
        result['generation_timings'] = runner.timings
    record()
    print(json.dumps(result, indent=2), flush=True)
