from dataclasses import dataclass, field

import torch

from asuka.asuka_ffi import ir


@dataclass
class Kernel:
  module: ir.module
  kernel: ir.asuka.kernel = field(init=False)
  func_name: str
  kernel_name: str
  ops: list[ir.operation]
  inputs: list[ir.value] = field(init=False)
  outputs: list[ir.value]

  def __post_init__(self):
    assert len(self.ops) >= 1
    self.inputs = Kernel.get_all_inputs(self.module, self.func_name, self.ops)
    assert Kernel.validate(self.module, self.func_name, self.kernel_name, self.ops, self.inputs, self.outputs)
    self._build_kernel_in_module()
    self.kernel = self.module.get_kernel(self.kernel_name)

  def _erase(self):
    kernel = self.module.get_kernel(self.kernel_name)
    kernel.erase()

  def dump(self):
    kernel = self.module.get_kernel(self.kernel_name)
    kernel.dump()

  @classmethod
  def from_all_ops(cls, module, func_name, kernel_name):
    fn = module.get_function(func_name)
    ops = []
    outputs = []

    def walk_fn(op):
      if "return" in op.get_name():
        for i in range(op.get_num_operands()):
          v = op.get_operand(i)
          outputs.append(v)
      else:
        ops.append(op)

    fn.get_callable_region().front().walk(walk_fn)
    return cls(module, func_name, kernel_name, ops, outputs)

  def _build_kernel_in_module(self):
    assert hasattr(self.module, "context")
    context = self.module.context
    builder = ir.builder(context)
    param_types = [v.get_type() for v in self.inputs]
    ret_types = [v.get_type() for v in self.outputs]
    func_type = builder.get_function_ty(param_types, ret_types)

    kernel = builder.get_or_insert_kernel(self.module, self.kernel_name, func_type)
    self.module.push_back(kernel)
    entry = kernel.add_entry_block()
    builder.set_insertion_point_to_start(entry)

    val_map = {arg: kernel.arg(i) for i, arg in enumerate(self.inputs)}

    # ops is topologically sorted
    for op in self.ops:
      new_op = builder.clone(op)
      for i in range(new_op.get_num_operands()):
        original_arg = op.get_operand(i)
        new_op.set_operand(i, val_map[original_arg])
      for i in range(new_op.get_num_results()):
        original_res = op.get_result(i)
        new_res = new_op.get_result(i)
        val_map[original_res] = new_res

    builder.create_return([val_map[original_res] for original_res in self.outputs])

  @classmethod
  def get_all_inputs(cls, module, func_name, ops):
    fn = module.get_function(func_name)
    # assume ops are topologically sorted
    inputs = [ops[0].get_operand(i) for i in range(ops[0].get_num_operands())]
    inter_res = set(ops[0].get_result(i) for i in range(ops[0].get_num_results()))
    for op in ops[1:]:
      for i in range(op.get_num_operands()):
        arg = op.get_operand(i)
        if arg not in inter_res:
          inputs.append(arg)
      for i in range(op.get_num_results()):
        res = op.get_result(i)
        inter_res.add(res)

    return inputs

  @classmethod
  def get_emptyuse_ops(cls, module, func_name, ops):
    fn = module.get_function(func_name)
    ret_ops = []
    # output of these ops not used by other ops
    for op in ops:
      assert op.get_num_results() == 1
      res = op.get_result(0)
      used = False
      for user in res.get_users():
        if user in ops:
          used = True
          break
      # print(f"{op.get_name()=} {used=}")
      if not used:
        ret_ops.append(op)
    return ret_ops

  @classmethod
  def validate(cls, module, func_name, kernel_name, ops, inputs, outputs):
    # check all op in func
    fn = module.get_function(func_name)
    all_ops = []
    fn.get_callable_region().front().walk(lambda op: all_ops.append(op))
    for op in ops:
      if op not in all_ops:
        raise Exception(f"kernel op {op} not in {func_name}")

    # check input dependency
    # all operands of op must be result of other op in ops, or in inputs
    prev_ops = []
    for op in ops:
      for i in range(op.get_num_operands()):
        operand = op.get_operand(i)
        if operand in inputs:
          continue
        def_op = operand.get_defining_op()
        if def_op is None:
          raise Exception(f"op {op} operand {i} {operand} not in inputs and has None def_op")

        if def_op not in ops:
          raise Exception(f"op {op} operand {i} def_op {def_op} not in ops")
        else:
          if def_op not in prev_ops:
            raise Exception(f"def_op={def_op} ops order error, need topologically sorted")
      prev_ops.append(op)

    # check input dependency
    # all outputs must be result of some op in ops
    all_results = set()
    for op in ops:
      for i in range(op.get_num_results()):
        result = op.get_result(i)
        all_results.add(result)
    for v in outputs:
      if v not in all_results:
        raise Exception(f"result not in ops all results")

    return True

  def get_flops(self):
    flops = 0
    for op in self.ops:
      flops += op.get_flops()
    return flops

  def get_mem_size(self):

    def get_tensor_size(v):
      ty = v.get_type().to_ranked_tensor_ty()
      if ty is not None:
        shape = ty.get_shape()
        size = 1
        for dim in shape:
          size *= dim
        return size
      else:
        return 0

    mem_size = sum(get_tensor_size(v) for v in self.inputs) + sum(get_tensor_size(v) for v in self.outputs)
    return mem_size

  def get_parallelism(self):
    kernel = self.module.get_kernel(self.kernel_name)
    return kernel.get_parallelism()

  def prepare_dummy_inputs(self, device=torch.cuda.current_device()):
    args = []
    for v in self.inputs:
      ty = v.get_type().to_ranked_tensor_ty()
      elem_ty = ty.get_element_ty()
      shape = ty.get_shape()

      if elem_ty.is_f16():
        arg = torch.randn(shape, dtype=torch.float16, device=device)
      elif elem_ty.is_f32():
        arg = torch.randn(shape, dtype=torch.float32, device=device)
      else:
        raise NotImplementedError()
      args.append(arg)
    return args
