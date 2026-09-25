"""Exact block-diagonal eager attention without allocating masked cross-block entries."""
import math
import torch
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen

ORIGINAL = qwen.Qwen2_5_VLVisionAttention.forward


def segmented(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
    length = hidden_states.shape[0]
    q,k,v = self.qkv(hidden_states).reshape(length,3,self.num_heads,-1).permute(1,0,2,3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb,rotary_pos_emb),dim=-1)
        cos,sin = emb.cos(),emb.sin()
    else:
        cos,sin = position_embeddings
    q,k = qwen.apply_rotary_pos_emb_vision(q,k,cos,sin)
    bounds = cu_seqlens.tolist()
    sizes = [b-a for a,b in zip(bounds,bounds[1:])]
    def attend(q,k,v):
        weights = torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.head_dim)
        weights = weights.softmax(-1,dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights,v)
    if len(set(sizes)) == 1:
        size = sizes[0]
        q,k,v = [x.reshape(-1,size,self.num_heads,self.head_dim).transpose(1,2) for x in (q,k,v)]
        output = attend(q,k,v).transpose(1,2).reshape(length,-1)
    else:
        pieces = [attend(q[a:b].transpose(0,1),k[a:b].transpose(0,1),v[a:b].transpose(0,1))
                  .transpose(0,1).reshape(b-a,-1) for a,b in zip(bounds,bounds[1:])]
        output = torch.cat(pieces,dim=0)
    return self.proj(output)


def install_and_check():
    rng_state = torch.get_rng_state()
    torch.random.default_generator.manual_seed(917)
    checks=[]
    for bounds in ([0,16,32,48],[0,17,48,64]):
        module=qwen.Qwen2_5_VLVisionAttention(32,4).double()
        x=torch.randn(bounds[-1],32,dtype=torch.float64,requires_grad=True)
        angles=torch.randn(bounds[-1],8,dtype=torch.float64)
        kwargs={'cu_seqlens':torch.tensor(bounds,dtype=torch.int32),
                'position_embeddings':(angles.cos(),angles.sin())}
        first=ORIGINAL(module,x,**kwargs)
        first_grad=torch.autograd.grad(first.square().sum(),x)[0]
        second=segmented(module,x,**kwargs)
        second_grad=torch.autograd.grad(second.square().sum(),x)[0]
        forward_error=float((first-second).abs().max())
        gradient_error=float((first_grad-second_grad).abs().max())
        if forward_error>1e-6 or gradient_error>1e-6:
            raise AssertionError((forward_error,gradient_error))
        checks.append({'bounds':bounds,'forward_max_error':forward_error,'gradient_max_error':gradient_error})
    qwen.Qwen2_5_VLVisionAttention.forward=segmented
    torch.set_rng_state(rng_state)
    return checks
