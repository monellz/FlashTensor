from pathlib import Path

import torch
import tensorrt as trt

from .utils import save_timing_cache, setup_timing_cache, trt_dtype_to_torch
from asuka.backend import Builder
from asuka.translate import onnx_from_asuka_kernel

import onnx


class TensorRTBuilder(Builder):

  def __init__(self, kernel, engine_dir=None, cache_dir=None, check_before_run=False, verbose=False):
    super().__init__(kernel)

    self.onnx_model = onnx_from_asuka_kernel(self.kernel.module, self.kernel.kernel_name)

    self.engine_dir = engine_dir
    self.engine_fn = f"{engine_dir}/{kernel.kernel_name}.engine" if engine_dir is not None else None
    self.cache_dir = cache_dir
    self.cache_fn = f"{cache_dir}/timing.cache" if cache_dir is not None else None

    self.verbose = verbose
    self.check_before_run = check_before_run

    self.trt_logger = trt.Logger(trt.Logger.INFO)
    if self.verbose:
      self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

  def build_engine(self):
    onnx_model = self.onnx_model
    onnx_model_buf = onnx_model.SerializeToString()

    trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

    builder = trt.Builder(self.trt_logger)
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(
    #   trt.MemoryPoolType.WORKSPACE, 8 * (2**30)
    # ) # 8 GB
    config.set_preview_feature(trt.PreviewFeature.PROFILE_SHARING_0806, True)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, self.trt_logger)

    if not parser.parse(onnx_model_buf):
      print(f"failed to parse onnx")
      print(onnx.helper.printable_graph(model.graph))
      exit(-1)

    if self.cache_fn is not None:
      print(f"reading timing cache from {self.cache_fn}")
      setup_timing_cache(config, self.cache_fn)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
      print("failed to create engine")
      exit(-1)

    if self.cache_fn is not None:
      print(f"serializing timing cache to file: {self.cache_fn}")
      save_timing_cache(config, self.cache_fn)

    if self.engine_fn is not None:
      with open(self.engine_fn, "wb") as f:
        print(f"serializing engine to file: {self.engine_fn}")
        f.write(engine_bytes)

    return engine_bytes

  def build_callable(self):
    engine_bytes = self.build_engine()

    runtime = trt.Runtime(self.trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    input_specs = []
    output_specs = []
    for i in range(engine.num_io_tensors):
      name = engine.get_tensor_name(i)
      is_input = False
      if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        is_input = True
      dtype = engine.get_tensor_dtype(name)
      shape = engine.get_tensor_shape(name)
      binding = {
        "index": i,
        "name": name,
        "dtype": trt_dtype_to_torch(dtype),
        "shape": tuple(shape),
      }
      if is_input:
        input_specs.append(binding)
      else:
        output_specs.append(binding)

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    # stream = torch.cuda.current_stream()
    if self.check_before_run:

      def f(*args):
        device = args[0].device
        for i, arg in enumerate(args):
          spec = input_specs[i]
          if tuple(arg.shape) != spec["shape"]:
            raise RuntimeError(f"arg {i} need shape {spec['shape']} but got {tuple(arg.shape)}")
          if arg.dtype != spec["dtype"]:
            raise RuntimeError(f"arg {i} need dtype {spec['dtype']} but got {arg.dtype}")
          ptr = arg.data_ptr()
          context.set_tensor_address(spec["name"], ptr)

        outputs = []
        for i, spec in enumerate(output_specs):
          res = torch.zeros(spec["shape"], dtype=spec["dtype"], device=device)
          ptr = res.data_ptr()
          context.set_tensor_address(spec["name"], ptr)
          outputs.append(res)
        ok = context.execute_async_v3(stream.cuda_stream)
        assert ok
        if len(outputs) == 1:
          return outputs[0]
        else:
          return tuple(outputs)

      return f
    else:

      def f(*args):
        device = args[0].device
        for i, arg in enumerate(args):
          spec = input_specs[i]
          ptr = arg.data_ptr()
          context.set_tensor_address(spec["name"], ptr)

        outputs = []
        for i, spec in enumerate(output_specs):
          res = torch.empty(spec["shape"], dtype=spec["dtype"], device=device)
          ptr = res.data_ptr()
          context.set_tensor_address(spec["name"], ptr)
          outputs.append(res)
        ok = context.execute_async_v3(stream.cuda_stream)
        assert ok
        if len(outputs) == 1:
          return outputs[0]
        else:
          return tuple(outputs)

      return f
