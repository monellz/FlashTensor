from asuka.asuka_ffi import passes, ir

from .utils import get_pass_manager


def fission(op, context=None):
  top_pm, pm = get_pass_manager(op, context)
  passes.add_cse(pm)
  passes.add_asuka_simplify(pm)
  passes.add_asuka_lower_complex_reduce(pm)
  passes.add_cse(pm)
  top_pm.run(op)
