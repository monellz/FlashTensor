import io
from functools import wraps

import torch
import onnx
import onnxsim
from pathlib import Path

from .translate import asuka_from_onnx
from .transform import fission


def erase_doc_string(onnx_model):
  for node in onnx_model.graph.node:
    assert hasattr(node, 'doc_string')
    node.doc_string = ''


def subgraph(onnx_dir):

  def decorator(model_cls):

    @wraps(model_cls, updated=())
    class SubgraphWrapper:

      def __init__(self, *args, **kwargs):
        self.model = model_cls(*args, **kwargs)
        self.func_name = self.model.__class__.__name__
        self.onnx_dir = onnx_dir
        self.module = None

      def __call__(self, *inputs):
        outputs = self.model(*inputs)
        fn = Path(self.onnx_dir) / f"{self.func_name}.onnx"
        torch.onnx.export(
          self.model,
          args=inputs,
          f=fn,
          input_names=[f"in_{i}" for i in range(len(inputs))],
          output_names=[f"out_{i}" for i in range(len(outputs))],
          verbose=True,
        )
        onnx_model = onnx.load_model(fn)
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check
        erase_doc_string(onnx_model)
        onnx.save(onnx_model, fn)
        self.module = asuka_from_onnx(onnx_model, self.func_name)

        # fission
        self.module = fission(self.module)
        self.module.dump()

        return outputs

    return SubgraphWrapper

  return decorator
