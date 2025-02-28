import tempfile
from dataclasses import dataclass, field
from itertools import combinations
from typing import Union

from asuka.asuka_ffi import ir
from asuka.transform.common import module_to_py, kernel_to_py

from .kernel import Kernel


@dataclass
class PartitionConfig:
  module: ir.module
  func_name: str

  partitions: list[list[int]]
  output_ops: list[list[int]]    # all op has only one output
  backends: Union[str, list[str]]
  kernels: list[Kernel] = field(init=False)

  def __post_init__(self):
    self.check_at_most_one_output()
    self.kernels = self._build_kernels()

  def dump_func(self):
    func_op = self.module.get_function(self.func_name)
    func_op.dump()

  def check_at_most_one_output(self):
    fn = self.module.get_function(self.func_name)

    def walk_fn(op):
      assert op.get_num_results() <= 1, f"{op} has {op.get_num_results()} results"

    fn.get_callable_region().front().walk(walk_fn)

  def _build_kernels(self):
    all_ops = []
    func_op = self.module.get_function(self.func_name)
    func_op.get_callable_region().front().walk(lambda op: all_ops.append(op) if "return" not in op.get_name() else None)

    kernels = []
    for i, (ops, out_ops) in enumerate(zip(self.partitions, self.output_ops)):
      kernel_name = f"{self.func_name}_p{i}"
      mlir_ops = [all_ops[i] for i in ops]
      mlir_outputs = [all_ops[i].get_result(0) for i in out_ops]
      kernel = Kernel(
        module=self.module,
        func_name=self.func_name,
        kernel_name=kernel_name,
        ops=mlir_ops,
        outputs=mlir_outputs,
      )
      kernels.append(kernel)

    return kernels

  def check_and_remove_redundancy(self):
    pass

  def reorder_kernels(self):
    # TODO: better sorting
    kernels = []
    old_kernels = self.kernels

    fn = self.module.get_function(self.func_name)
    inter_res = set(fn.arg(i) for i in range(fn.get_num_arguments()))
    while len(old_kernels) > 0:
      # find one we can compute
      found_kernel = None
      for kernel in enumerate(old_kernels):
        can_compute = True
        for arg in kernel.inputs:
          if arg not in inter_res:
            can_compute = False
            break
        if can_compute:
          found_kernel = kernel
          break
      assert found_kernel is not None
      old_kernels.remove(found_kernel)
      kernels.append(found_kernel)

      for res in found_kernel.outputs:
        inter_res.add(res)

    self.kernels = kernels

  def build_callable(self):
    self.reorder_kernels()
    raise NotImplementedError()

  def optimize(self):
    from asuka.transform.common import optimize
    # optimize(self.module)
    # optimize kernel by kernel
    failed_kernels = set()
    for i, kernel in enumerate(self.kernels):
      print(f"optimize {kernel.kernel_name}", flush=True)
      try:
        optimize(kernel.kernel, context=self.module.context)
      except Exception as e:
        print(f"optimize {kernel.kernel_name} failed", flush=True)
        # kernel.kernel.dump()
        # print(e)
        failed_kernels.add(i)
    
    if len(failed_kernels) > 0:
      # remove kernel
      partitions = []
      output_ops = []
      backends = self.backends
      assert isinstance(backends, str), f"{backends=}"
      kernels = []

      for i, kernel in enumerate(self.kernels):
        if i not in failed_kernels:
          partitions.append(self.partitions[i])
          output_ops.append(self.output_ops[i])
          kernels.append(kernel)
        else:
          kernel_name = kernel.kernel_name
          self.module.erase_kernel(kernel_name)
      
      self.partitions = partitions
      self.output_ops = output_ops
      self.backends = backends
      self.kernels = kernels

  def _add_uncovered_ops(self):
    id2op = {}
    fn = self.module.get_function(self.func_name)

    def id_update(op):
      if op.get_name() != "func.return":
        i = len(id2op)
        id2op[i] = op

    fn.get_callable_region().front().walk(id_update)

    all_op_ids = set(id2op.keys())
    for partition in self.partitions:
      for op_id in partition:
        all_op_ids.discard(op_id)

    if len(all_op_ids) > 0:
      for i, op_id in enumerate(all_op_ids):
        op = id2op[op_id]
        print(f"uncovered ops: {op.get_name()}")

        # FIXME: now it should be only 'asuka.avg_pool'
        assert op.get_name() == "asuka.avg_pool", f"{op.get_name()=}"

        new_partition = [op_id]
        new_output_ops = [op_id]

        mlir_ops = [op]
        mlir_outputs = [op.get_result(0)]
        kernel = Kernel(
          module=self.module,
          func_name=self.func_name,
          kernel_name=f"{self.func_name}_uncovered_{i}",
          ops=mlir_ops,
          outputs=mlir_outputs,
        )

        self.partitions.append(new_partition)
        self.output_ops.append(new_output_ops)
        assert isinstance(self.backends, str), f"{self.backends=}"
        self.kernels.append(kernel)

  def profile(self):
    self._add_uncovered_ops()
    py_str = module_to_py(self.module)

    import subprocess
    import re
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
      path = f.name
      f.write(py_str)
    print(f"path: {path}", flush=True)
    out = subprocess.run(["python3", path], capture_output=True)
    print(f"profiling...", flush=True)
    if out.returncode != 0:
      print(out.stderr)
      raise Exception("profile failed")

    out_str = out.stdout.decode("utf-8")
    print(f"{out_str=}")
    perf = {}
    for kernel in self.kernels:
      pattern = f"\[{kernel.kernel_name}\] avg_ms: (\d+\.\d+)"
      avg_ms = re.search(pattern, out_str).group(1)
      print(f"[{kernel.kernel_name}] avg_ms: {avg_ms}")
      perf[kernel.kernel_name] = float(avg_ms)

    return perf

  def codegen(self, perf):
    # TODO: naive impl
    from copy import deepcopy
    kernel_ids = list(range(len(self.kernels)))
    ready = {}
    fn = self.module.get_function(self.func_name)
    func_res = []
    func_arg = set()

    for i in range(fn.get_callable_region().front().get_num_arguments()):
      arg = fn.get_callable_region().front().get_argument(i)
      func_arg.add(arg)
      ready[arg] = True

    def update_ready(op):
      for i in range(op.get_num_results()):
        res = op.get_result(i)
        ready[res] = False
      if op.get_name() == "func.return":
        for i in range(op.get_num_operands()):
          arg = op.get_operand(i)
          func_res.append(arg)

    fn.get_callable_region().front().walk(update_ready)

    def clear_ready(ready):
      for k in ready.keys():
        if k in func_arg:
          ready[k] = True
        else:
          ready[k] = False

    def find_execuable_kernel(kernel_ids, ready):
      for i in kernel_ids:
        kernel = self.kernels[i]
        can_exec = True
        for arg in kernel.inputs:
          if not ready[arg]:
            can_exec = False
            break
        if can_exec:
          return i
      return None

    best_kernels = []
    best_time = float("inf")
    for r in range(1, len(self.kernels) + 1):
      for cur_ids in combinations(kernel_ids, r):
        total_time = 0
        for i in cur_ids:
          kernel = self.kernels[i]
          total_time += perf[kernel.kernel_name]
        if total_time >= best_time:
          continue

        clear_ready(ready)
        kernel_order = []
        cur_ids = list(cur_ids)
        while True:
          kid = find_execuable_kernel(cur_ids, ready)
          if kid is None:
            break
          kernel = self.kernels[kid]
          for arg in kernel.outputs:
            ready[arg] = True
          cur_ids.remove(kid)
          kernel_order.append(kid)

        for arg in func_res:
          if not ready[arg]:
            total_time = float("inf")
            break
        if total_time < best_time:
          best_kernels = kernel_order
          best_time = total_time

    best_kernels = [self.kernels[i] for i in best_kernels]
    print(f"best_kernel:\n")
    for kernel in best_kernels:
      print(f"{kernel.kernel_name}")
    print(f"{best_time=}")

    name_map = {}
    caller = f"\ndef {self.func_name}("
    for i in range(fn.get_callable_region().front().get_num_arguments()):
      arg = fn.get_callable_region().front().get_argument(i)
      name_map[arg] = f"arg_{i}"
      caller += f"arg_{i}, "
    caller = caller[:-2] + "):\n"

    final_py_str = ""

    for i, kernel in enumerate(best_kernels):
      if i == 0:
        final_py_str += kernel_to_py(kernel.kernel, add_import=True, add_benchmark=True)
      else:
        final_py_str += kernel_to_py(kernel.kernel, add_import=False, add_benchmark=True)

      caller += "  "
      for out_i, out in enumerate(kernel.outputs):
        name = f"k{i}_out_{out_i}"
        caller += f"{name}, "
        name_map[out] = name
      caller = caller[:-2] + " = "
      caller += f"{kernel.kernel_name}("
      for arg_i, arg in enumerate(kernel.inputs):
        caller += f"{name_map[arg]}, "
      caller = caller[:-2] + ")\n"

    final_py_str += caller
    ret_str = "  return "
    for res in func_res:
      ret_str += f"{name_map[res]}, "
    ret_str = ret_str[:-2] + "\n"

    final_py_str += ret_str

    print(f"{final_py_str}", flush=True)
    return final_py_str
