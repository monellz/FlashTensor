#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"
#include "asuka/Dialect/AsukaTriton/Transforms/Passes.h"

#include <queue>

#include "dbg.h"

namespace mlir::asuka::triton {

#define GEN_PASS_DEF_ASUKATRITONSQUEEZEBLOCK
#include "asuka/Dialect/AsukaTriton/Transforms/Passes.h.inc"

} // namespace mlir::asuka::triton

namespace mlir::asuka {
namespace triton {
namespace {

void infer_type(Operation *op) {
  auto type_infer = dyn_cast<InferTypeOpInterface>(op);
  if (type_infer) {
    llvm::SmallVector<::mlir::Type, 1> new_types;
    auto success =
        type_infer.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                    op->getPropertiesStorage(), op->getRegions(), new_types);
    assert(succeeded(success));
    for (size_t i = 0; i < new_types.size(); ++i) {
      op->getResult(i).setType(new_types[i]);
    }
  } else {
    op->dump();
    llvm_unreachable("op cannot infer result type");
  }
}

struct SqueezeBlockPattern : public OpRewritePattern<BlockPointerOfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BlockPointerOfOp block_ptr_of_op, PatternRewriter &rewriter) const override {
    SmallVector<int64_t> shape(block_ptr_of_op.getShape());
    SmallVector<int64_t> stride(block_ptr_of_op.getStride());
    SmallVector<int64_t> offset(block_ptr_of_op.getOffset());
    SmallVector<int64_t> block_shape(block_ptr_of_op.getBlockShape());
    SmallVector<int64_t> order(block_ptr_of_op.getOrder());

    size_t rank = shape.size();
    if (rank == 1) {
      return failure();
    }

    int target_dim = -1;
    for (size_t i = 0; i < rank; i++) {
      if (shape[i] == 1 && stride[i] == 1 && offset[i] == 0 && block_shape[i] == 1 && order[i] == 0) {
        target_dim = (int)i;
        break;
      }
    }
    if (target_dim == -1) {
      return failure();
    }

    if (block_ptr_of_op.getResult().hasOneUse()) {
      auto user = *block_ptr_of_op.getResult().getUsers().begin();
      if (isa<BlockStoreOp>(user)) {
        return failure();
      }
    }

    // this block can be squeezed
    shape.erase(shape.begin() + target_dim);
    stride.erase(stride.begin() + target_dim);
    offset.erase(offset.begin() + target_dim);
    block_shape.erase(block_shape.begin() + target_dim);
    order.erase(order.begin() + target_dim);
    for (size_t i = 0; i < rank - 1; i++) {
      order[i]--;
    }

    auto new_block_ptr_of_op = rewriter.replaceOpWithNewOp<BlockPointerOfOp>(
        block_ptr_of_op, block_ptr_of_op.getBasePointer(), block_ptr_of_op.getBaseOffset(), shape, stride, offset,
        block_shape, order);

    // bfs
    std::queue<Value> q;
    q.push(new_block_ptr_of_op.getResult());
    while (!q.empty()) {
      auto val = q.front();
      q.pop();
      for (auto &use : val.getUses()) {
        auto user = use.getOwner();
        auto use_op_idx = use.getOperandNumber();
        if (auto scf_for_op = dyn_cast<scf::ForOp>(user)) {
          assert(use_op_idx >= 3);
          auto iter_idx = use_op_idx - 3;
          // set block argument
          scf_for_op.getRegionIterArgs()[iter_idx].setType(val.getType());
          scf_for_op->getResult(iter_idx).setType(val.getType());
          auto val = scf_for_op.getRegionIterArgs()[iter_idx];
          q.push(val);
          q.push(scf_for_op->getResult(iter_idx));
        } else if (auto block_load_op = dyn_cast<BlockLoadOp>(user)) {
          infer_type(block_load_op);
          auto val = block_load_op.getResult();
          q.push(val);
        } else if (auto block_advance_op = dyn_cast<BlockAdvanceOp>(user)) {
          SmallVector<int64_t> offsets(block_advance_op.getOffsets());
          offsets.erase(offsets.begin() + target_dim);
          block_advance_op.setOffsets(offsets);
          infer_type(block_advance_op);
          auto val = block_advance_op.getResult();
          q.push(val);
        } else if (user->getNumResults() > 0) {
          // unsqueeze
          rewriter.setInsertionPoint(user);
          auto new_val = rewriter.create<::mlir::asuka::UnsqueezeOp>(user->getLoc(), use.get(), target_dim);
          user->setOperand(use_op_idx, new_val);
        }
      }
    }
    return success();
  }
};

class SqueezeBlockPass : public ::mlir::asuka::triton::impl::AsukaTritonSqueezeBlockBase<SqueezeBlockPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp k = getOperation();

    patterns.add<SqueezeBlockPattern>(context);
    if (applyPatternsAndFoldGreedily(k, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSqueezeBlockPass() { return std::make_unique<SqueezeBlockPass>(); }

} // namespace triton
} // namespace mlir::asuka
