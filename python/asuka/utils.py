import time
import ctypes
import torch
import numpy as np

_cudart = ctypes.CDLL('libcudart.so')


def get_sm_num(dev_id=torch.cuda.current_device()):
  assert torch.cuda.is_available()
  dev_prop = torch.cuda.get_device_properties(dev_id)
  return dev_prop.multi_processor_count


def profile_start():
  ret = _cudart.cudaProfilerStart()
  if ret != 0:
    raise Exception("cudaProfilerStart() returned %d" % ret)


# FIXME: not stop profiling
def profile_stop():
  ret = _cudart.cudaProfilerStop()
  if ret != 0:
    raise Exception("cudaProfilerStop() returned %d" % ret)


def perf(label, f, args, kwargs={}, gflops=None, mem_gb=None, run=10, warmup=4, profile=False):
  torch.cuda.synchronize()
  for _ in range(warmup):
    torch.cuda.synchronize()
    o = f(*args, **kwargs)
    torch.cuda.synchronize()

  if profile:
    profile_start()
  ms = []
  for _ in range(run):
    torch.cuda.synchronize()
    tik = time.time()
    o = f(*args, **kwargs)
    torch.cuda.synchronize()
    tok = time.time()
    ms.append((tok - tik) * 1000.0)
  if profile:
    profile_stop()

  min_ms = np.min(ms)
  max_ms = np.max(ms)
  avg_ms = np.mean(ms)
  msg = f'[{label}] avg {avg_ms:.4f} ms, min {min_ms:.4f} ms, max {max_ms:.4f} ms'
  if gflops is not None:
    msg += f', {gflops / (avg_ms / 1000.0)} gflops/s'
  if mem_gb is not None:
    msg += f', {mem_gb / (avg_ms / 1000.0)} gb/s'

  msg += f' ({run} runs, {warmup} warmups)' if not profile else f' ({run} runs, {warmup} warmups, profiled)'
  print(msg)


def loss(out, ref, remove_zero=False):
  assert out.shape == ref.shape, f"{out.shape=} {ref.shape=}"
  if out.dtype in {torch.int32, torch.int64}:
    err_num = torch.sum(out != ref).item()
    total_num = out.numel()
    return {"err_num": err_num, "err": err_num / total_num}
  else:
    abs_max_loss = (out.float() - ref.float()).abs().max().item()
    abs_mean_loss = (out.float() - ref.float()).abs().mean().item()
    deno = ref.float().abs()
    if remove_zero:
      zero_mask = deno == 0
      deno[zero_mask] = 1.0
    rel_max_loss = ((out.float() - ref.float()) / deno).abs().max().item()
    rel_mean_loss = ((out.float() - ref.float()) / deno).abs().mean().item()
    return {'abs_max': abs_max_loss, 'abs_mean': abs_mean_loss, 'rel_max': rel_max_loss, 'rel_mean': rel_mean_loss, 'remove_zero': remove_zero}
