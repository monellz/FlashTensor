import io
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import click
import onnx
import onnxsim


class AttnH2O(nn.Module):
  def __init__(self, kv_head_num, head_num, head_dim, masked):
    super().__init__()
    self.kv_head_num = kv_head_num
    self.head_num = head_num
    self.head_dim = head_dim
    self.masked = masked
    self.group_size = self.head_num // self.kv_head_num

  def forward_masked(self, q, k, v):
    q_len = q.shape[0]
    kv_len = k.shape[0]
    mask = torch.full((1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

    q = q.view(q_len, self.head_num, self.head_dim)
    k = k.view(kv_len, self.kv_head_num, self.head_dim)
    v = v.view(kv_len, self.kv_head_num, self.head_dim)
    if self.group_size > 1:
      k = k[:, :, None, :].expand(kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(kv_len, self.head_num, self.head_dim)
      v = v[:, :, None, :].expand(kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(kv_len, self.head_num, self.head_dim)
  
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores += mask
    scores = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(scores.to(q.dtype), v).transpose(0, 1).contiguous()

    if self.group_size > 1:
      h2o_score = scores.reshape(self.kv_head_num, self.group_size, q_len, kv_len).mean(dim=1)
    else:
      h2o_score = scores
    h2o_score = h2o_score.sum(dim=1)
    return out.view(q_len, self.head_num * self.head_dim), h2o_score.view(self.kv_head_num, kv_len)
  
  def forward_wo_mask(self, q, k, v):
    q_len = q.shape[0]
    kv_len = k.shape[0]

    q = q.view(q_len, self.head_num, self.head_dim)
    k = k.view(kv_len, self.kv_head_num, self.head_dim)
    v = v.view(kv_len, self.kv_head_num, self.head_dim)
    if self.group_size > 1:
      k = k[:, :, None, :].expand(kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(kv_len, self.head_num, self.head_dim)
      v = v[:, :, None, :].expand(kv_len, self.kv_head_num, self.group_size, self.head_dim).reshape(kv_len, self.head_num, self.head_dim)
  
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    scores = F.softmax(scores.float(), dim=-1)
    out = torch.matmul(scores.to(q.dtype), v).transpose(0, 1).contiguous()

    if self.group_size > 1:
      h2o_score = scores.reshape(self.kv_head_num, self.group_size, q_len, kv_len).mean(dim=1)
    else:
      h2o_score = scores
    h2o_score = h2o_score.sum(dim=1)
    return out.view(q_len, self.head_num * self.head_dim), h2o_score.view(self.kv_head_num, kv_len)
  
  def forward(self, q, k, v):
    if self.masked:
      return self.forward_masked(q, k, v)
    else:
      return self.forward_wo_mask(q, k, v)
  

@click.command()
@click.option('--q_len', type=int, default=4096)
@click.option('--kv_len', type=int, default=4096)
@click.option('--head_num', type=int, default=32)
@click.option('--kv_head_num', type=int, default=32)
@click.option('--head_dim', type=int, default=128)
@click.option('--masked', type=bool, default=True)
@click.option('--type', type=str, default='f16')
@click.option('--show_result', is_flag=True, default=False)
def main(q_len, kv_len, head_num, kv_head_num, head_dim, masked, type, show_result):
  from asuka.utils import loss, perf

  from asuka.translate import asuka_from_onnx
  from asuka.transform import fission
  from asuka.partition.kernel import Kernel
  from asuka.backend.tensorrt import TensorRTBuilder

  torch.manual_seed(0)
  assert type in ['f16', 'bf16']
  print(f"{q_len=} {kv_len=} {head_num=} {kv_head_num=} {head_dim=} {masked=} {type=} {show_result=}")

  hidden_size = head_num * head_dim
  kv_hidden_size = kv_head_num * head_dim

  device = torch.cuda.current_device()
  dtype = torch.float16 if type == 'f16' else torch.bfloat16

  q = torch.randn(q_len, hidden_size, dtype=dtype, device=device)
  k = torch.randn(kv_len, kv_hidden_size, dtype=dtype, device=device)
  v = torch.randn(kv_len, kv_hidden_size, dtype=dtype, device=device)

  model = AttnH2O(kv_head_num, head_num, head_dim, masked).eval().to(device)
  onnx_bytes = io.BytesIO()
  torch.onnx.export(
    model,
    args=(q, k, v),
    f=onnx_bytes,
    input_names=["q", "k", "v"],
    output_names=["out", "h2o_score"],
    verbose=False,
  )
  onnx_model = onnx.load_model_from_string(onnx_bytes.getvalue())
  print(onnx.helper.printable_graph(onnx_model.graph))
  func_name = model.__class__.__name__
  module = asuka_from_onnx(onnx_model, func_name)
  # module = fission(module)
  module.dump()

  kernel = Kernel.from_all_ops(module, func_name, f"{func_name}_full")
  builder = TensorRTBuilder(kernel, check_before_run=False, engine_dir=None, cache_dir=None, verbose=False)
  print(onnx.helper.printable_graph(builder.onnx_model.graph))
  trt_f = builder.build_callable()
  torch_f = model.forward

  out_trt, h2o_score_trt = trt_f(q, k, v)
  out_torch, h2o_score_torch = torch_f(q, k, v)
  torch.cuda.synchronize()
  if show_result:
    print(f"{out_trt=}")
    print(f"{out_torch=}")
    print(f"{h2o_score_trt=}")
    print(f"{h2o_score_torch=}")
  print(f"{loss(out_trt, out_torch, remove_zero=True)=}")
  print(f"{loss(h2o_score_trt, h2o_score_torch, remove_zero=True)=}")

  perf(
    label="torch",
    f=torch_f,
    args=[q, k, v],
    warmup=10,
    run=10,
    profile=False,
  )

  perf(
    label="trt",
    f=trt_f,
    args=[q, k, v],
    warmup=10,
    run=10,
    profile=True,
  )
  
if __name__ == "__main__":
  main()
