#include <chrono>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKAEQUIVALENTTRANSFORM
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

bool isAllPositiveConstant(arith::ConstantOp op) {
  auto val_attr = op.getValue();
  if (auto float_attr = dyn_cast<FloatAttr>(val_attr)) {
    return float_attr.getValueAsDouble() > 0.0;
  }
  if (auto int_attr = dyn_cast<IntegerAttr>(val_attr)) {
    return int_attr.getInt() > 0;
  }

  // dense
  if (auto dense_attr = dyn_cast<DenseElementsAttr>(val_attr)) {
    bool all_positive = true;
    for (auto e : dense_attr.getValues<Attribute>()) {
      assert(isa<FloatAttr>(e));
      auto _e = cast<FloatAttr>(e).getValueAsDouble();
      if (_e <= 0.0) {
        all_positive = false;
      }
    }
    return all_positive;
  } else {
    op->dump();
    llvm_unreachable("not supported");
  }
}

int64_t getMaxTensorSize(Operation *top_op) {
  int64_t max_size = 0;
  top_op->walk([&](Operation *op) {
    for (auto arg : op->getOperands()) {
      if (isa<RankedTensorType>(arg.getType())) {
        auto tensor_type = arg.getType().cast<RankedTensorType>();
        int64_t size = 1;
        for (int64_t i = 0; i < tensor_type.getRank(); ++i) {
          size *= tensor_type.getShape()[i];
        }
        max_size = std::max(max_size, size);
      }
    }
    for (auto res : op->getResults()) {
      if (isa<RankedTensorType>(res.getType())) {
        auto tensor_type = res.getType().cast<RankedTensorType>();
        int64_t size = 1;
        for (int64_t i = 0; i < tensor_type.getRank(); ++i) {
          size *= tensor_type.getShape()[i];
        }
        max_size = std::max(max_size, size);
      }
    }
  });
  return max_size;
}

int64_t AccessChanged(ArrayRef<Operation *> previous_ops, ArrayRef<Operation *> later_ops) {
  auto get_access = [](Operation *op) {
    int64_t out = 0;
    for (auto arg : op->getOperands()) {
      if (isa<RankedTensorType>(arg.getType())) {
        auto tensor_type = arg.getType().cast<RankedTensorType>();
        int64_t access = 1;
        for (int64_t i = 0; i < tensor_type.getRank(); ++i) {
          access *= tensor_type.getShape()[i];
        }
        out += access;
      }
    }
    for (auto res : op->getResults()) {
      if (isa<RankedTensorType>(res.getType())) {
        auto tensor_type = res.getType().cast<RankedTensorType>();
        int64_t access = 1;
        for (int64_t i = 0; i < tensor_type.getRank(); ++i) {
          access *= tensor_type.getShape()[i];
        }
        out += access;
      }
    }
    return out;
  };

  int64_t previous_access = 0;
  int64_t later_access = 0;
  for (auto op : previous_ops) {
    previous_access += get_access(op);
  }
  for (auto op : later_ops) {
    later_access += get_access(op);
  }
  return later_access - previous_access;
}

class TransformBase {
public:
  virtual std::string getName() = 0;
  virtual bool match(Operation *) = 0;
  virtual int64_t rewrite(Operation *) = 0;
};

// dot(mul(x, scalar), z) -> dot(x, mul(z, scalar))
class MulScalarDotLeft : public TransformBase {
public:
  std::string getName() override { return "dot(mul(x, scalar), z) -> dot(x, mul(z, scalar))"; }

  bool match(Operation *op) override {
    auto mul_op = dyn_cast<MulOp>(op);
    if (mul_op == nullptr)
      return false;
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto mul_res = mul_op.getResult();
    if (!mul_res.hasOneUse())
      return false;
    auto dot_op = dyn_cast<DotOp>(*mul_res.getUsers().begin());
    if (dot_op == nullptr)
      return false;
    if (dot_op.getLhs() != mul_res)
      return false;

    auto mul_rhs_shape = cast<RankedTensorType>(mul_rhs.getType()).getShape();
    if (mul_rhs_shape.size() == 1 && mul_rhs_shape[0] == 1) {
      return true;
    } else {
      return false;
    }
  }

  int64_t rewrite(Operation *op) override {
    auto mul_op = cast<MulOp>(op);
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto mul_res = mul_op.getResult();
    auto dot_op = cast<DotOp>(*mul_res.getUsers().begin());
    auto mul_rhs_shape = cast<RankedTensorType>(mul_rhs.getType()).getShape();
    assert(dot_op.getLhs() == mul_res);
    assert(mul_rhs_shape.size() == 1 && mul_rhs_shape[0] == 1);
    OpBuilder builder(dot_op);
    auto new_mul_op = builder.create<MulOp>(mul_op.getLoc(), dot_op.getRhs(), mul_rhs);
    auto new_dot_op = builder.create<DotOp>(dot_op.getLoc(), mul_lhs, new_mul_op.getResult());
    auto access = AccessChanged({&*mul_op, &*dot_op}, {&*new_mul_op, &*new_dot_op});

    dot_op.getResult().replaceAllUsesWith(new_dot_op.getResult());
    dot_op->erase();
    mul_op->erase();
    return access;
  }
};

// mul(dot(x, y), z) -> dot(mul(x, z), y)
class DotMulLeft : public TransformBase {
public:
  std::string getName() override { return "mul(dot(x, y), z) -> dot(mul(x, z), y)"; }

  bool match(Operation *op) override {
    auto mul_op = dyn_cast<MulOp>(op);
    if (mul_op == nullptr)
      return false;
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    if (mul_lhs.getDefiningOp() == nullptr)
      return false;
    if (!mul_lhs.hasOneUse())
      return false;

    auto dot_op = dyn_cast<DotOp>(mul_lhs.getDefiningOp());
    if (dot_op == nullptr)
      return false;
    auto mul_rhs_broadcasted_shape = cast<BroadcastableBinaryOpInterface>(&*mul_op).getRhsBroadcastedShape();
    if (mul_rhs_broadcasted_shape[mul_rhs_broadcasted_shape.size() - 1] != 1)
      return false;

    return true;
  }

  int64_t rewrite(Operation *op) override {
    auto mul_op = cast<MulOp>(op);
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto dot_op = cast<DotOp>(mul_lhs.getDefiningOp());
    auto dot_lhs = dot_op.getLhs();
    auto dot_rhs = dot_op.getRhs();

    OpBuilder builder(mul_op);
    auto new_mul_op = builder.create<MulOp>(mul_op.getLoc(), dot_lhs, mul_rhs);
    auto new_dot_op = builder.create<DotOp>(dot_op.getLoc(), new_mul_op.getResult(), dot_rhs);

    auto access = AccessChanged({&*dot_op, &*mul_op}, {&*new_mul_op, &*new_dot_op});
    mul_op.getResult().replaceAllUsesWith(new_dot_op.getResult());
    mul_op->erase();
    dot_op->erase();
    return access;
  }
};

// dot(mul(x, z), y) -> mul(dot(x, y), z)
class MulDotLeft : public TransformBase {
public:
  std::string getName() override { return "dot(mul(x, z), y) -> mul(dot(x, y), z)"; }
  bool match(Operation *op) override {
    auto mul_op = dyn_cast<MulOp>(op);
    if (mul_op == nullptr)
      return false;
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto mul_res = mul_op.getResult();
    if (!mul_res.hasOneUse())
      return false;
    auto dot_op = dyn_cast<DotOp>(*mul_res.getUsers().begin());
    if (dot_op == nullptr)
      return false;
    auto mul_rhs_broadcasted_shape = cast<BroadcastableBinaryOpInterface>(&*mul_op).getRhsBroadcastedShape();
    if (dot_op.getLhs() != mul_res || mul_rhs_broadcasted_shape[mul_rhs_broadcasted_shape.size() - 1] != 1)
      return false;
    return true;
  }
  int64_t rewrite(Operation *op) override {
    auto mul_op = cast<MulOp>(op);
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto mul_res = mul_op.getResult();
    auto dot_op = cast<DotOp>(*mul_res.getUsers().begin());
    auto dot_rhs = dot_op.getRhs();

    OpBuilder builder(dot_op);
    auto new_dot_op = builder.create<DotOp>(dot_op.getLoc(), mul_lhs, dot_rhs);
    auto new_mul_op = builder.create<MulOp>(mul_op.getLoc(), new_dot_op.getResult(), mul_rhs);

    auto access = AccessChanged({&*mul_op, &*dot_op}, {&*new_dot_op, &*new_mul_op});
    dot_op.getResult().replaceAllUsesWith(new_mul_op.getResult());
    dot_op->erase();
    mul_op->erase();
    return access;
  }
};

// div(dot(x, y), z) -> dot(div(x, z), y)
class DotDivLeft : public TransformBase {
public:
  std::string getName() override { return "div(dot(x, y), z) -> dot(div(x, z), y)"; }

  bool match(Operation *op) override {
    auto div_op = dyn_cast<DivOp>(op);
    if (div_op == nullptr)
      return false;
    auto div_lhs = div_op.getLhs();
    auto div_rhs = div_op.getRhs();
    if (div_lhs.getDefiningOp() == nullptr)
      return false;
    if (!div_lhs.hasOneUse())
      return false;

    auto dot_op = dyn_cast<DotOp>(div_lhs.getDefiningOp());
    if (dot_op == nullptr)
      return false;
    auto div_rhs_broadcasted_shape = cast<BroadcastableBinaryOpInterface>(&*div_op).getRhsBroadcastedShape();
    if (div_rhs_broadcasted_shape[div_rhs_broadcasted_shape.size() - 1] != 1)
      return false;

    return true;
  }

  int64_t rewrite(Operation *op) override {
    auto div_op = cast<DivOp>(op);
    auto div_lhs = div_op.getLhs();
    auto div_rhs = div_op.getRhs();
    auto dot_op = cast<DotOp>(div_lhs.getDefiningOp());
    auto dot_lhs = dot_op.getLhs();
    auto dot_rhs = dot_op.getRhs();

    OpBuilder builder(div_op);
    auto new_div_op = builder.create<DivOp>(div_op.getLoc(), dot_lhs, div_rhs);
    auto new_dot_op = builder.create<DotOp>(dot_op.getLoc(), new_div_op.getResult(), dot_rhs);

    auto access = AccessChanged({&*dot_op, &*div_op}, {&*new_div_op, &*new_dot_op});
    div_op.getResult().replaceAllUsesWith(new_dot_op.getResult());
    div_op->erase();
    dot_op->erase();
    return access;
  }
};

// dot(div(x, z), y) -> div(dot(x, y), z)
class DivDotLeft : public TransformBase {
public:
  std::string getName() override { return "dot(div(x, z), y) -> div(dot(x, y), z)"; }
  bool match(Operation *op) override {
    auto div_op = dyn_cast<DivOp>(op);
    if (div_op == nullptr)
      return false;
    auto div_lhs = div_op.getLhs();
    auto div_rhs = div_op.getRhs();
    auto div_res = div_op.getResult();
    if (!div_res.hasOneUse())
      return false;
    auto dot_op = dyn_cast<DotOp>(*div_res.getUsers().begin());
    if (dot_op == nullptr)
      return false;
    auto div_rhs_broadcasted_shape = cast<BroadcastableBinaryOpInterface>(&*div_op).getRhsBroadcastedShape();
    if (dot_op.getLhs() != div_res || div_rhs_broadcasted_shape[div_rhs_broadcasted_shape.size() - 1] != 1)
      return false;
    return true;
  }
  int64_t rewrite(Operation *op) override {
    auto div_op = cast<DivOp>(op);
    auto div_lhs = div_op.getLhs();
    auto div_rhs = div_op.getRhs();
    auto div_res = div_op.getResult();
    auto dot_op = cast<DotOp>(*div_res.getUsers().begin());
    auto dot_rhs = dot_op.getRhs();

    OpBuilder builder(dot_op);
    auto new_dot_op = builder.create<DotOp>(dot_op.getLoc(), div_lhs, dot_rhs);
    auto new_div_op = builder.create<DivOp>(div_op.getLoc(), new_dot_op.getResult(), div_rhs);

    auto access = AccessChanged({&*div_op, &*dot_op}, {&*new_dot_op, &*new_div_op});
    dot_op.getResult().replaceAllUsesWith(new_div_op.getResult());
    dot_op->erase();
    div_op->erase();
    return access;
  }
};

// mul(add(x, y), z) -> add(mul(x, z), mul(y, z))
class AddMulLeft : public TransformBase {
public:
  std::string getName() override { return "mul(add(x, y), z) -> add(mul(x, z), mul(y, z))"; }
  bool match(Operation *op) override {
    auto mul_op = dyn_cast<MulOp>(op);
    if (mul_op == nullptr)
      return false;
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    if (mul_lhs.getDefiningOp() == nullptr)
      return false;
    if (!mul_lhs.hasOneUse())
      return false;

    auto add_op = dyn_cast<AddOp>(mul_lhs.getDefiningOp());
    if (add_op == nullptr)
      return false;
    return true;
  }
  int64_t rewrite(Operation *op) override {
    auto mul_op = cast<MulOp>(op);
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto add_op = cast<AddOp>(mul_lhs.getDefiningOp());

    auto add_lhs = add_op.getLhs();
    auto add_rhs = add_op.getRhs();

    OpBuilder builder(mul_op);
    auto new_mul_op_0 = builder.create<MulOp>(mul_op.getLoc(), add_lhs, mul_rhs);
    auto new_mul_op_1 = builder.create<MulOp>(mul_op.getLoc(), add_rhs, mul_rhs);
    auto new_add_op = builder.create<AddOp>(add_op.getLoc(), new_mul_op_0.getResult(), new_mul_op_1.getResult());

    auto access = AccessChanged({&*add_op, &*mul_op}, {&*new_mul_op_0, &*new_mul_op_1, &*new_add_op});
    mul_op.getResult().replaceAllUsesWith(new_add_op.getResult());
    mul_op->erase();
    add_op->erase();
    return access;
  }
};

// add(mul(x, z), mul(y, z)) -> mul(add(x, y), z)
class MulMulAdd : public TransformBase {
public:
  std::string getName() override { return "add(mul(x, z), mul(y, z)) -> mul(add(x, y), z)"; }
  bool match(Operation *op) override {
    auto add_op = dyn_cast<AddOp>(op);
    if (add_op == nullptr)
      return false;
    auto add_lhs = add_op.getLhs();
    auto add_rhs = add_op.getRhs();

    if (add_lhs.getDefiningOp() == nullptr)
      return false;
    if (add_rhs.getDefiningOp() == nullptr)
      return false;

    auto lhs_mul_op = dyn_cast<MulOp>(add_lhs.getDefiningOp());
    if (lhs_mul_op == nullptr)
      return false;
    if (!add_lhs.hasOneUse())
      return false;
    auto rhs_mul_op = dyn_cast<MulOp>(add_rhs.getDefiningOp());
    if (rhs_mul_op == nullptr)
      return false;
    if (lhs_mul_op.getRhs() != rhs_mul_op.getRhs())
      return false;
    if (!add_rhs.hasOneUse())
      return false;
    return true;
  }
  int64_t rewrite(Operation *op) override {
    auto add_op = cast<AddOp>(op);
    auto add_lhs = add_op.getLhs();
    auto add_rhs = add_op.getRhs();
    auto lhs_mul_op = cast<MulOp>(add_lhs.getDefiningOp());
    auto rhs_mul_op = cast<MulOp>(add_rhs.getDefiningOp());

    OpBuilder builder(add_op);
    auto new_add_op = builder.create<AddOp>(add_op.getLoc(), lhs_mul_op.getLhs(), rhs_mul_op.getLhs());
    auto new_mul_op = builder.create<MulOp>(add_op.getLoc(), new_add_op.getResult(), lhs_mul_op.getRhs());

    auto access = AccessChanged({&*add_op, &*lhs_mul_op, &*rhs_mul_op}, {&*new_add_op, &*new_mul_op});
    add_op.getResult().replaceAllUsesWith(new_mul_op.getResult());
    add_op->erase();
    lhs_mul_op->erase();
    rhs_mul_op->erase();
    return access;
  }
};

// mul(trilu, positive) -> trilu
class MulTriluPositive : public TransformBase {
public:
  std::string getName() override { return "mul(trilu, positive) -> trilu"; }
  bool match(Operation *op) override {
    auto mul_op = dyn_cast<MulOp>(op);
    if (mul_op == nullptr)
      return false;
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    if (mul_lhs.getDefiningOp() == nullptr)
      return false;
    if (mul_rhs.getDefiningOp() == nullptr)
      return false;
    auto trilu_op = dyn_cast<TriluOp>(mul_lhs.getDefiningOp());
    if (trilu_op == nullptr)
      return false;
    auto const_op = dyn_cast<arith::ConstantOp>(mul_rhs.getDefiningOp());
    if (const_op == nullptr)
      return false;
    if (!isAllPositiveConstant(const_op))
      return false;
    return true;
  }
  int64_t rewrite(Operation *op) override {
    auto mul_op = cast<MulOp>(op);
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto trilu_op = cast<TriluOp>(mul_lhs.getDefiningOp());
    auto const_op = cast<arith::ConstantOp>(mul_rhs.getDefiningOp());

    auto access = AccessChanged({&*trilu_op, &*const_op, &*mul_op}, {&*trilu_op});
    mul_op.getResult().replaceAllUsesWith(trilu_op.getResult());
    mul_op->erase();
    return access;
  }
};

// mul(positive, trilu) -> trilu
class MulPositiveTrilu : public TransformBase {
public:
  std::string getName() override { return "mul(positive, trilu) -> trilu"; }
  bool match(Operation *op) override {
    auto mul_op = dyn_cast<MulOp>(op);
    if (mul_op == nullptr)
      return false;
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    if (mul_lhs.getDefiningOp() == nullptr)
      return false;
    if (mul_rhs.getDefiningOp() == nullptr)
      return false;
    auto const_op = dyn_cast<arith::ConstantOp>(mul_lhs.getDefiningOp());
    if (const_op == nullptr)
      return false;
    if (!isAllPositiveConstant(const_op))
      return false;
    auto trilu_op = dyn_cast<TriluOp>(mul_rhs.getDefiningOp());
    if (trilu_op == nullptr)
      return false;
    return true;
  }
  int64_t rewrite(Operation *op) override {
    auto mul_op = cast<MulOp>(op);
    auto mul_lhs = mul_op.getLhs();
    auto mul_rhs = mul_op.getRhs();
    auto const_op = cast<arith::ConstantOp>(mul_lhs.getDefiningOp());
    auto trilu_op = cast<TriluOp>(mul_rhs.getDefiningOp());

    auto access = AccessChanged({&*trilu_op, &*const_op, &*mul_op}, {&*trilu_op});
    mul_op.getResult().replaceAllUsesWith(trilu_op.getResult());
    mul_op->erase();
    return access;
  }
};

struct Action {
  TransformBase *transform;
  Operation *op;
  Action(TransformBase *trans, Operation *op) : transform(trans), op(op) {}
};

void search(Operation *kernel_op) {
  srand(0);
  OpBuilder builder(kernel_op);
  auto simu_start = std::chrono::high_resolution_clock::now();
  // llvm::errs() << "kernel: " << kernel_op.getName() << "\n";

  // register transform
  SmallVector<std::unique_ptr<TransformBase>> transforms;
  transforms.push_back(std::make_unique<DotDivLeft>());
  transforms.push_back(std::make_unique<DivDotLeft>());
  transforms.push_back(std::make_unique<DotMulLeft>());
  transforms.push_back(std::make_unique<MulDotLeft>());
  transforms.push_back(std::make_unique<AddMulLeft>());
  transforms.push_back(std::make_unique<MulMulAdd>());
  transforms.push_back(std::make_unique<MulTriluPositive>());
  // transforms.push_back(std::make_unique<MulPositiveTrilu>());

  transforms.push_back(std::make_unique<MulScalarDotLeft>());
  // dbg(transforms.size());

  int64_t max_tensor_size = getMaxTensorSize(kernel_op);

  int64_t best_access = 0;
  Operation *best_kernel_op = builder.clone(*kernel_op);
  Operation *cur_kernel_op = builder.clone(*kernel_op);
  int64_t cur_access = 0;

  double temp = 1000;
  int simu_iter = 0;
  while (temp > 0.001) {
    // cur_kernel_op->dump();
    // available actions
    SmallVector<Action> actions;
    cur_kernel_op->walk([&](Operation *op) {
      for (auto &t : transforms) {
        if (t->match(op)) {
          actions.push_back(Action(t.get(), op));
        }
      }
    });
    if (actions.size() == 0) {
      break;
    }
    // dbg(temp);
    // dbg(actions.size());
    // for (auto& act: actions) {
    //   dbg(act.transform->getName());
    // }

    // random select
    int selected_id = rand() % (int)actions.size();
    // dbg(actions[selected_id].transform->getName());

    auto &action = actions[selected_id];
    // action.op->dump();

    // Operation *next_kernel_op = cur_kernel_op->clone();
    Operation *next_kernel_op = builder.clone(*cur_kernel_op);
    // apply
    int64_t access_changed = action.transform->rewrite(action.op);
    double delta = (double)access_changed / (double)max_tensor_size;
    // dbg(access_changed, delta);
    // dbg(cur_access, best_access);

    bool accept = false;
    if (delta < 0.0) {
      accept = true;
    } else if (delta == 0.0) {
      accept = rand() / (double)RAND_MAX < 0.5;
    } else {
      accept = exp(-delta / temp) > (rand() / (double)RAND_MAX);
    }
    // dbg(accept);

    if (cur_access + access_changed < best_access) {
      best_access = cur_access + access_changed;
      // best_kernel_op = next_kernel_op->clone();
      best_kernel_op->erase();
      best_kernel_op = builder.clone(*next_kernel_op);
    }

    if (accept) {
      next_kernel_op->erase();
      next_kernel_op = nullptr;
      cur_access += access_changed;
    } else {
      cur_kernel_op->erase();
      cur_kernel_op = next_kernel_op;
    }

    temp *= 0.97;
    simu_iter += 1;
  }
  cur_kernel_op->erase();
  auto simu_end = std::chrono::high_resolution_clock::now();
  auto simu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(simu_end - simu_start).count();

  // dbg("done");
  // dbg(cur_access);
  // dbg(best_access);
  // dbg("post simulate");
  auto post_simu_start = std::chrono::high_resolution_clock::now();
  int post_simu_iter = 1000;
  for (int i = 0; i < post_simu_iter; ++i) {
    // available actions
    SmallVector<Action> actions;
    best_kernel_op->walk([&](Operation *op) {
      for (auto &t : transforms) {
        if (t->match(op)) {
          actions.push_back(Action(t.get(), op));
        }
      }
    });
    if (actions.size() == 0) {
      break;
    }
    int selected_id = rand() % (int)actions.size();
    auto &action = actions[selected_id];
    // Operation *next_best_kernel_op = best_kernel_op->clone();
    Operation *next_best_kernel_op = builder.clone(*best_kernel_op);
    // apply
    int64_t access_changed = action.transform->rewrite(action.op);
    if (access_changed < 0) {
      best_access += access_changed;
      next_best_kernel_op->erase();
      next_best_kernel_op = nullptr;
    } else {
      best_kernel_op->erase();
      best_kernel_op = next_best_kernel_op;
    }
  }
  auto post_simu_end = std::chrono::high_resolution_clock::now();
  auto post_simu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(post_simu_end - post_simu_start).count();
  // dbg("all done");
  // dbg(best_access);
  // dbg(kernel_op.getSymName());
  // dbg(simu_ms, post_simu_ms);
  // dbg(simu_iter, post_simu_iter);

  // kernel_op.getCallableRegion()->takeBody(best_kernel_op->getRegion(0));
  kernel_op->getRegion(0).takeBody(best_kernel_op->getRegion(0));
  best_kernel_op->erase();
}

class EquivalentTransformPass : public ::mlir::asuka::impl::AsukaEquivalentTransformBase<EquivalentTransformPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp k = getOperation();
    SmallVector<ParallelForOp> parallel_for_ops;
    k.getCallableRegion()->front().walk([&](ParallelForOp op) { parallel_for_ops.push_back(op); });
    if (parallel_for_ops.size() == 0) {
      search(k);
    } else {
      assert(parallel_for_ops.size() == 1);
      search(parallel_for_ops[0]);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createEquivalentTransformPass() { return std::make_unique<EquivalentTransformPass>(); }

} // namespace mlir::asuka