import numpy as np
import torch

import onnx
from onnx import checker
import onnxsim

from asuka.asuka_ffi import ir


def asuka_from_onnx(onnx_model, func_name):
  context = ir.context()
  ir.asuka.load_dialects(context)
  builder = ir.builder(context)
  module = builder.create_module()
  module.context = context

  onnx_type_to_mlir = {
    onnx.TensorProto.DataType.Value('FLOAT16'): builder.get_f16_ty(),
    onnx.TensorProto.DataType.Value('FLOAT'): builder.get_f32_ty(),
    onnx.TensorProto.DataType.Value('INT64'): builder.get_int64_ty(),
    onnx.TensorProto.DataType.Value('BOOL'): builder.get_bool_ty(),
  }
  np_type_to_mlir = {
    np.float16: builder.get_f16_ty(),
    np.dtypes.Float16DType(): builder.get_f16_ty(),
    np.float32: builder.get_f32_ty(),
  }

  graph = onnx_model.graph

  def gather_tensor_types(vals):
    types = []
    for x in vals:
      shape = [dim.dim_value for dim in x.type.tensor_type.shape.dim]
      dtype = x.type.tensor_type.elem_type
      types.append(builder.get_ranked_tensor_ty(onnx_type_to_mlir[dtype], shape))
    return types

  param_types = gather_tensor_types(graph.input)
  ret_types = gather_tensor_types(graph.output)
  func_type = builder.get_function_ty(param_types, ret_types)

  fn = builder.get_or_insert_function(module, func_name, func_type)
  module.push_back(fn)
  entry = fn.add_entry_block()
  builder.set_insertion_point_to_start(entry)

  val_map = {graph.input[i].name: entry.get_argument(i) for i in range(len(graph.input))}
  const_map = {init.name: onnx.numpy_helper.to_array(init) for init in graph.initializer}
  onnx_val_info = {val.name: val.type for val in graph.value_info}

  def find_val_or_create_tensor_const(name):
    if name in val_map:
      return val_map[name]
    else:
      assert name in const_map, f"{name=}"
      val = const_map[name]
      if val.size == 1:
        val = val.reshape(1)
      if val.dtype == np.float32:
        return builder.get_f32_tensor(val.flatten().tolist(), val.shape)
      elif val.dtype == np.float16:
        return builder.get_f16_tensor(val.astype(np.float32).flatten().tolist(), val.shape)
      elif val.dtype == np.int64:
        return builder.get_int64_tensor(val.astype(np.int64).flatten().tolist(), val.shape)
      else:
        raise NotImplementedError(f"{val.dtype=}")

  for node in graph.node:
    if node.op_type == "Reshape":
      assert len(node.input) == 2
      operand = node.input[0]
      ret_shape_val = node.input[1]
      assert ret_shape_val in const_map
      ret_shape = const_map[ret_shape_val]
      allowzero = onnx.helper.get_node_attr_value(node, "allowzero")
      assert allowzero == 0

      out = builder.create_reshape(val_map[operand], ret_shape)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Transpose":
      assert len(node.input) == 1
      operand = node.input[0]
      perm = onnx.helper.get_node_attr_value(node, "perm")

      out = builder.create_permute(val_map[operand], perm)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "MatMul":
      assert len(node.input) == 2
      lhs = node.input[0]
      rhs = node.input[1]

      out = builder.create_dot(val_map[lhs], val_map[rhs])
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "AveragePool":
      assert len(node.input) == 1
      operand = node.input[0]
      kernel_size = onnx.helper.get_node_attr_value(node, "kernel_shape")
      padding = onnx.helper.get_node_attr_value(node, "pads")
      stride = onnx.helper.get_node_attr_value(node, "strides")
      ceil_mode = onnx.helper.get_node_attr_value(node, "ceil_mode") == 1
      count_include_pad = onnx.helper.get_node_attr_value(node, "count_include_pad") == 1

      assert len(kernel_size) == len(stride)
      assert len(padding) == len(kernel_size) * 2
      # alwayse padding both sides
      padding = padding[:len(padding) // 2]

      out = builder.create_avgpool(val_map[operand], kernel_size, stride, padding, ceil_mode, count_include_pad)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Div":
      assert len(node.input) == 2
      lhs = node.input[0]
      rhs = node.input[1]

      lhs_val = find_val_or_create_tensor_const(lhs)
      rhs_val = find_val_or_create_tensor_const(rhs)
      out = builder.create_div(lhs_val, rhs_val)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Mul":
      assert len(node.input) == 2
      lhs = node.input[0]
      rhs = node.input[1]

      lhs_val = find_val_or_create_tensor_const(lhs)
      rhs_val = find_val_or_create_tensor_const(rhs)
      out = builder.create_mul(lhs_val, rhs_val)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Add":
      assert len(node.input) == 2
      lhs = node.input[0]
      rhs = node.input[1]

      lhs_val = find_val_or_create_tensor_const(lhs)
      rhs_val = find_val_or_create_tensor_const(rhs)
      out = builder.create_add(lhs_val, rhs_val)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Tanh":
      assert len(node.input[0])
      operand = node.input[0]

      out = builder.create_tanh(val_map[operand])
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Log":
      assert len(node.input) == 1
      operand = node.input[0]

      operand_val = find_val_or_create_tensor_const(operand)
      out = builder.create_log(operand_val)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Neg":
      assert len(node.input) == 1
      operand = node.input[0]

      operand_val = find_val_or_create_tensor_const(operand)
      out = builder.create_neg(operand_val)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Cast":
      assert len(node.input) == 1
      operand = node.input[0]
      to = onnx.helper.get_node_attr_value(node, "to")
      mlir_type = onnx_type_to_mlir[to]

      out = builder.create_convert(val_map[operand], mlir_type)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Softmax":
      assert len(node.input) == 1
      operand = node.input[0]
      axis = onnx.helper.get_node_attr_value(node, "axis")
      out = builder.create_softmax(val_map[operand], axis)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "ReduceSum":
      assert len(node.input) == 2
      operand = node.input[0]
      axis = node.input[1]
      assert axis in const_map
      axis = const_map[axis].item()
      keep_dim = onnx.helper.get_node_attr_value(node, "keepdims")

      out = builder.create_reduce(val_map[operand], axis, ir.asuka.REDUCE_ADD, keep_dim == 1)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Constant":
      # do not create arith::constant here because many ops support constant attributes
      value = onnx.helper.get_node_attr_value(node, "value")
      value = onnx.numpy_helper.to_array(value)
      assert len(node.output) == 1
      const_map[node.output[0]] = value
    elif node.op_type == "Trilu":
      assert len(node.input) == 1 or len(node.input) == 2
      diagonal = 0
      if len(node.input) == 2:
        diagonal_val = node.input[1]
        assert diagonal_val in const_map
        diagonal = const_map[diagonal_val].item()
      operand_name = node.input[0]
      assert operand_name in const_map
      operand = const_map[operand_name]
      shape = operand.shape
      mlir_type = np_type_to_mlir[operand.dtype]
      is_upper = onnx.helper.get_node_attr_value(node, "upper") == 1
      # faster than np
      # FIXME: user warning for non-writable numpy arr
      operand_torch = torch.from_numpy(operand)
      # simplify shape to matrix shape
      assert operand_torch.numel() == operand_torch.shape[-2] * operand_torch.shape[-1]
      operand_torch = operand_torch.reshape(operand_torch.shape[-2], operand_torch.shape[-1])
      shape = list(operand_torch.shape)
      if torch.all(operand_torch == torch.inf):
        out = builder.create_inf_trilu(diagonal, is_upper, shape, mlir_type, False)
      elif torch.all(operand_torch == -torch.inf):
        out = builder.create_inf_trilu(diagonal, is_upper, shape, mlir_type, True)
      else:
        raise NotImplementedError()
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Pow":
      assert len(node.input) == 2
      base_name = node.input[0]
      power_name = node.input[1]
      assert power_name in const_map
      base_val = find_val_or_create_tensor_const(base_name)
      power_val = find_val_or_create_tensor_const(power_name)

      out = builder.create_pow(base_val, power_val)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "GreaterOrEqual":
      assert len(node.input) == 2
      lhs = node.input[0]
      rhs = node.input[1]

      lhs_val = find_val_or_create_tensor_const(lhs)
      rhs_val = find_val_or_create_tensor_const(rhs)
      out = builder.create_cmp(lhs_val, rhs_val, ir.asuka.CMP_GE)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    elif node.op_type == "Greater":
      assert len(node.input) == 2
      lhs = node.input[0]
      rhs = node.input[1]

      lhs_val = find_val_or_create_tensor_const(lhs)
      rhs_val = find_val_or_create_tensor_const(rhs)
      out = builder.create_cmp(lhs_val, rhs_val, ir.asuka.CMP_GT)
      assert len(node.output) == 1
      val_map[node.output[0]] = out
    else:
      raise NotImplementedError(f"{node.op_type=}")

  vals = [val_map[out.name] for out in graph.output]
  builder.ret(vals)
  return module


class Counter:

  def __init__(self, start=0):
    self.start = start
    self.cur = start

  def inc(self):
    ret = self.cur
    self.cur += 1
    return ret

  def clear(self):
    self.cur = start


def onnx_from_asuka_kernel(module, kernel_name, simplify=True):
  context = module.context
  kernel = module.get_kernel(kernel_name)
  kernel.dump()

  inputs = []
  input_names = [f"input_{i}" for i in range(kernel.get_num_arguments())]
  outputs = []

  val_map = {}
  for i in range(kernel.get_num_arguments()):
    arg = kernel.arg(i)
    arg_type = arg.get_type().to_ranked_tensor_ty()
    elem_ty = arg_type.get_element_ty()
    shape = arg_type.get_shape()
    onnx_type = None
    if elem_ty.is_f16():
      onnx_type = onnx.TensorProto.FLOAT16
    elif elem_ty.is_f32():
      onnx_type = onnx.TensorProto.FLOAT
    else:
      raise NotImplementedError()

    onnx_arg = onnx.helper.make_tensor_value_info(input_names[i], onnx_type, shape)
    inputs.append(onnx_arg)

    val_map[arg] = input_names[i]

  nodes = []
  inits = []
  counter = Counter()

  def walk_fn(op):
    name = op.get_name()

    def get_np_ty(ty):
      if ty.is_f16():
        return np.float16
      elif ty.is_f32():
        return np.float32
      else:
        raise NotImplementedError(f"{ty}")

    def get_torch_ty(ty):
      if ty.is_f16():
        return torch.float16
      elif ty.is_f32():
        return torch.float32
      else:
        raise NotImplementedError(f"{ty}")

    def get_onnx_ty(ty):
      if ty.is_f16():
        return onnx.TensorProto.DataType.Value('FLOAT16')
      elif ty.is_f32():
        return onnx.TensorProto.DataType.Value('FLOAT')
      else:
        raise NotImplementedError(f"{ty}")

    if name == "arith.constant":
      res_type = op.get_result(0).get_type().to_ranked_tensor_ty()
      shape = res_type.get_shape()
      elem_ty = res_type.get_element_ty()
      np_ty = get_np_ty(elem_ty)
      constant_op = op.to_constant_op()
      val = constant_op.get_splat_float_value()
      assert res_type.get_rank() == 1 and shape[0] == 1
      res_name = f"const_{counter.inc()}"

      init = onnx.numpy_helper.from_array(np.array(val, dtype=np_ty), name=res_name)
      inits.append(init)

      val_map[op.get_result(0)] = res_name
    elif name == "asuka.reshape":
      input_type = op.get_operand(0).get_type().to_ranked_tensor_ty()
      res_type = op.get_result(0).get_type().to_ranked_tensor_ty()
      elem_ty = res_type.get_element_ty()

      shape_name = f"shape_{counter.inc()}"
      shape_init = onnx.numpy_helper.from_array(np.array(res_type.get_shape(), dtype=np.int64), name=shape_name)
      inits.append(shape_init)

      res_name = f"reshape_{counter.inc()}"
      node = onnx.helper.make_node(
        "Reshape",
        inputs=[val_map[op.get_operand(0)], shape_name],
        outputs=[res_name],
        allowzero=0,
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.permute":
      dims = ir.asuka.permute.from_operation(op).get_dims()

      res_name = f"permute_{counter.inc()}"
      node = onnx.helper.make_node(
        "Transpose",
        inputs=[val_map[op.get_operand(0)]],
        outputs=[res_name],
        perm=dims,
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.div":
      lhs = op.get_operand(0)
      rhs = op.get_operand(1)
      res_name = f"div_{counter.inc()}"
      node = onnx.helper.make_node(
        "Div",
        inputs=[val_map[lhs], val_map[rhs]],
        outputs=[res_name],
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.add":
      lhs = op.get_operand(0)
      rhs = op.get_operand(1)
      res_name = f"add_{counter.inc()}"
      node = onnx.helper.make_node(
        "Add",
        inputs=[val_map[lhs], val_map[rhs]],
        outputs=[res_name],
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.log":
      operand = op.get_operand(0)
      res_name = f"log_{counter.inc()}"
      node = onnx.helper.make_node(
        "Log",
        inputs=[val_map[operand]],
        outputs=[res_name],
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.neg":
      operand = op.get_operand(0)
      res_name = f"neg_{counter.inc()}"
      node = onnx.helper.make_node(
        "Neg",
        inputs=[val_map[operand]],
        outputs=[res_name],
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.dot":
      lhs = op.get_operand(0)
      rhs = op.get_operand(1)
      res_name = f"dot_{counter.inc()}"
      node = onnx.helper.make_node(
        "MatMul",
        inputs=[val_map[lhs], val_map[rhs]],
        outputs=[res_name],
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.pow":
      lhs = op.get_operand(0)
      rhs = op.get_operand(1)
      res_name = f"pow_{counter.inc()}"
      node = onnx.helper.make_node(
        "Pow",
        inputs=[val_map[lhs], val_map[rhs]],
        outputs=[res_name],
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.convert":
      arg = op.get_operand(0)
      dst_type = ir.asuka.convert.from_operation(op).get_dst_type()
      res_name = f"convert_{counter.inc()}"
      node = onnx.helper.make_node(
        "Cast",
        inputs=[val_map[arg]],
        outputs=[res_name],
        to=get_onnx_ty(dst_type),
      )
      nodes.append(node)
      val_map[op.get_result(0)] = res_name
    elif name == "asuka.reduce":
      arg = op.get_operand(0)
      res = op.get_result(0)
      res_name = f"reduce_{counter.inc()}"
      op = ir.asuka.reduce.from_operation(op)
      dim = op.get_reduce_dim()
      keep_dim = op.get_keep_dim()

      dim_name = f"reduce_dim_{counter.inc()}"
      dim_init = onnx.numpy_helper.from_array(np.array([dim], dtype=np.int64), name=dim_name)
      inits.append(dim_init)

      if op.is_reduce_add():
        node = onnx.helper.make_node(
          "ReduceSum",
          inputs=[val_map[arg], dim_name],
          outputs=[res_name],
          keepdims=keep_dim,
        )
        nodes.append(node)
        val_map[res] = res_name
      else:
        raise NotImplementedError()
    elif name == "asuka.exp":
      arg = op.get_operand(0)
      res = op.get_result(0)
      res_name = f"exp_{counter.inc()}"

      node = onnx.helper.make_node(
        "Exp",
        inputs=[val_map[arg]],
        outputs=[res_name],
      )
      nodes.append(node)
      val_map[res] = res_name
    elif name == "asuka.return":
      for i in range(op.get_num_operands()):
        arg = op.get_operand(i)
        elem_ty = arg.get_type().to_ranked_tensor_ty().get_element_ty()
        shape = arg.get_type().to_ranked_tensor_ty().get_shape()
        onnx_type = None

        if elem_ty.is_f16():
          onnx_type = onnx.TensorProto.FLOAT16
        elif elem_ty.is_f32():
          onnx_type = onnx.TensorProto.FLOAT
        else:
          raise NotImplementedError()

        tensor = onnx.helper.make_tensor_value_info(val_map[arg], onnx_type, shape)
        outputs.append(tensor)
    elif name == "asuka.softmax":
      arg = op.get_operand(0)
      res = op.get_result(0)
      res_name = f"softmax_{counter.inc()}"
      op = ir.asuka.softmax.from_operation(op)
      dim = op.get_reduce_dim()

      node = onnx.helper.make_node(
        "Softmax",
        inputs=[val_map[arg]],
        outputs=[res_name],
        axis=dim,
      )
      nodes.append(node)
      val_map[res] = res_name
    elif name == "asuka.trilu":
      res = op.get_result(0)
      res_name = f"trilu_{counter.inc()}"
      res_elem_type = res.get_type().to_ranked_tensor_ty().get_element_ty()
      res_elem_torch_type = get_torch_ty(res_elem_type)

      op = ir.asuka.trilu.from_operation(op)
      diagonal = op.get_diagonal()
      is_upper = op.get_is_upper()
      shape = op.get_shape()

      if op.is_inf_val():
        # faster than np.full
        trilu_res = torch.full(shape, torch.inf, dtype=res_elem_torch_type)
      elif op.is_neg_inf_val():
        trilu_res = torch.full(shape, -torch.inf, dtype=res_elem_torch_type)
      else:
        raise NotImplementedError()
      trilu_res = torch.triu(trilu_res, diagonal=diagonal)

      trilu_res_init = onnx.numpy_helper.from_array(trilu_res.numpy(), name=res_name)
      inits.append(trilu_res_init)
      val_map[res] = res_name
    else:
      raise NotImplementedError(f"{op.get_name()}")

  kernel.get_callable_region().front().walk(walk_fn)
  graph = onnx.helper.make_graph(
    nodes=nodes,
    name=kernel_name,
    inputs=inputs,
    outputs=outputs,
    initializer=inits,
  )
  model = onnx.helper.make_model(
    graph,
    producer_name=kernel_name,
  )

  # model = onnx.shape_inference.infer_shapes(model)
  checker.check_model(model, full_check=True)

  if simplify:
    model, check = onnxsim.simplify(model)
    assert check

  return model


if __name__ == "__main__":
  import onnx
  fn = "/home/zhongrx/llm/tune/asuka/cases/AttnH2O.onnx"
  onnx_model = onnx.load_model(fn)
  # print(f"{onnx_model}")
  module = asuka_from_onnx(onnx_model, "attn_h2o")
  print(f"{module=}")
