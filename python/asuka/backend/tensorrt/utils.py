import tensorrt as trt
import os
import torch

_trt_to_torch_dtype_dict = {
  trt.float16: torch.float16,
  trt.float32: torch.float32,
  trt.int64: torch.int64,
  trt.int32: torch.int32,
  trt.int8: torch.int8,
  trt.bool: torch.bool,
  trt.bfloat16: torch.bfloat16,
  trt.fp8: torch.float8_e4m3fn,
}


def trt_dtype_to_torch(dtype):
  ret = _trt_to_torch_dtype_dict.get(dtype)
  assert ret is not None, f'Unsupported dtype: {dtype}'
  return ret


# Sets up the builder to use the timing cache file, and creates it if it does not already exist
def setup_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
  buffer = b""
  if os.path.exists(timing_cache_path):
    with open(timing_cache_path, mode="rb") as timing_cache_file:
      buffer = timing_cache_file.read()
  timing_cache: trt.ITimingCache = config.create_timing_cache(buffer)
  config.set_timing_cache(timing_cache, True)


# Saves the config's timing cache to file
def save_timing_cache(config: trt.IBuilderConfig, timing_cache_path: os.PathLike):
  timing_cache: trt.ITimingCache = config.get_timing_cache()
  with open(timing_cache_path, "wb") as timing_cache_file:
    timing_cache_file.write(memoryview(timing_cache.serialize()))
