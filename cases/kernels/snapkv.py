import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SnapKV(nn.Module):
  def __init__(self, kernel_size=5, kv_head_num=32, head_num=32, head_dim=128):
    super().__init__()
    self.kernel_size = kernel_size
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.hidden_size = head_num * head_dim
    self.kv_hidden_size = kv_head_num * head_dim
    self.group_size = self.head_num // self.kv_head_num
    assert self.group_size == 1

  def forward(self, q, k, v):
    q_len = q.shape[1]
    kv_len = k.shape[1]
    batch_size = q.shape[0]
    mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    probs = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()

    # snapkv
    snapkv_score = probs 
    snapkv_score = snapkv_score.sum(dim=2)
    snapkv_score = F.avg_pool1d(snapkv_score, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

    return out.view(batch_size, q_len, self.head_num, self.head_dim), snapkv_score.view(batch_size, self.kv_head_num, kv_len)

  def prepare(self, batch_size=1, q_len=4096, kv_len=4096, dtype=torch.float16, device=torch.cuda.current_device()):
    q = torch.randn(batch_size, q_len, self.head_num, self.head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, kv_len, self.head_num, self.head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, kv_len, self.head_num, self.head_dim, dtype=dtype, device=device)

    ret = {
      'input': {
        'q': q,
        'k': k,
        'v': v,
      },
      'output': ['out', 'snapkv_score']
    }
    return ret
  

def get_model():
  return SnapKV()