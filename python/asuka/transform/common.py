from asuka.asuka_ffi import passes, ir
from .utils import get_pass_manager


def simplify(op, context=None):
  top_pm, pm = get_pass_manager(op, context)
  passes.add_asuka_simplify(pm)
  top_pm.run(op)


def annotate_parallelism(op, context=None):
  top_pm, pm = get_pass_manager(op, context)
  passes.add_asuka_annotate_parallelism(pm)
  top_pm.run(op)


def optimize(op, context=None):
  top_pm, pm = get_pass_manager(op, context)

  def simplify(_pm):
    passes.add_asuka_simplify(pm)
    passes.add_cse(pm)

  simplify(pm)
  passes.add_asuka_erase_type_in_kernel(pm)
  simplify(pm)
  passes.add_asuka_replace_exp_and_log(pm)
  simplify(pm)
  passes.add_asuka_equivalent_transform(pm)
  simplify(pm)
  passes.add_asuka_to_mask(pm)
  simplify(pm)
  passes.add_asuka_parallelize(pm, True)
  simplify(pm)
  passes.add_asuka_equivalent_transform(pm)
  simplify(pm)
  passes.add_asuka_tiling(pm)
  simplify(pm)
  passes.add_asuka_dynamic_for(pm)
  simplify(pm)
  passes.add_asuka_recover_type_in_kernel(pm)
  simplify(pm)
  passes.add_asuka_convert_asuka_to_asukatriton(pm)
  simplify(pm)
  passes.add_asukatriton_squeeze_block(pm)
  simplify(pm)

  top_pm.run(op)


def module_to_py(module):
  py_str = passes.translate_module_to_py(module, True)
  return py_str


def kernel_to_py(kernel, add_import=True, add_benchmark=True):
  py_str = passes.translate_kernel_to_py(kernel, add_import, add_benchmark)
  return py_str
