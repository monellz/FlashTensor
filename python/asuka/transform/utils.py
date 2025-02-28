from asuka.asuka_ffi import ir, passes


def get_pass_manager(op, context=None):
  if isinstance(op, ir.module):
    assert context is None
    context = op.context
    top_pm = passes.pass_manager(context)
    pm = top_pm.nest_any()
  else:
    assert context is not None
    top_pm = passes.pass_manager(context)
    pm = top_pm
  return top_pm, pm
