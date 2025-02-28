import io
import importlib
import click
import onnx
import torch

from asuka.translate import asuka_from_onnx
from asuka.transform import fission
from asuka.transform.common import simplify

from asuka.partition.kernel import Kernel
from asuka.partition.connected import Connected

def build_module(model_name):
  py_module = importlib.import_module("." + model_name, package="kernels")
  model = py_module.get_model()
  inout = model.prepare()

  onnx_bytes = io.BytesIO()
  torch.onnx.export(
    model,
    args=tuple(inout['input'].values()),
    f=onnx_bytes,
    input_names=list(inout['input'].keys()),
    output_names=inout['output'],
    verbose=False,
  )
  onnx_model = onnx.load_model_from_string(onnx_bytes.getvalue())
  print(onnx.helper.printable_graph(onnx_model.graph))
  func_name = model.__class__.__name__
  module = asuka_from_onnx(onnx_model, func_name)
  fission(module)
  simplify(module)
  # module.dump()
  return module, func_name


@click.command()
@click.option('--model_name', '-m', type=str, required=True, default="h2o")
def main(model_name):
  print(f"{model_name=}")
  module, func_name = build_module(model_name)
  module.dump()

  connected = Connected(module, func_name)

if __name__ == "__main__":
  main()