#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKATILING
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

void infer_and_set_ret_type(mlir::Operation *op) {
  if (isa<InferTypeOpInterface>(op)) {
    auto type_infer = cast<InferTypeOpInterface>(op);
    llvm::SmallVector<::mlir::Type, 1> new_types;
    auto success =
        type_infer.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                    op->getPropertiesStorage(), op->getRegions(), new_types);
    assert(succeeded(success));
    for (size_t i = 0; i < new_types.size(); ++i) {
      op->getResult(i).setType(new_types[i]);
    }
  } else if (op->getNumResults() == 0) {
    return;
  } else {
    op->dump();
    llvm_unreachable("its not type_infer");
  }
}

void check_no_use_for_results(Operation *op) {
  for (int i = 0; i < op->getNumResults(); ++i) {
    if (!op->getResult(i).use_empty()) {
      op->getParentOp()->dump();
      llvm::errs() << "op: " << *op << "\n";
      llvm_unreachable("result has use");
    }
  }
}

struct TilingDotPattern : public OpRewritePattern<DotOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(DotOp op, mlir::PatternRewriter &rewriter) const override {
    // op->dump();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto res = op.getResult();
    auto lhs_type = cast<RankedTensorType>(lhs.getType());
    auto rhs_type = cast<RankedTensorType>(rhs.getType());
    auto res_type = cast<RankedTensorType>(res.getType());
    int m = lhs_type.getShape()[lhs_type.getRank() - 2];
    int k = lhs_type.getShape()[lhs_type.getRank() - 1];
    int n = rhs_type.getShape()[rhs_type.getRank() - 1];

    int block_size = 128;
    if (m > block_size && k <= block_size && n <= block_size) {
      // dbg("dot tiling m");
      assert(m % block_size == 0);
      SmallVector<Type> res_types({res_type});
      SmallVector<Value> block_args({lhs});
      SmallVector<int64_t> block_dims({lhs_type.getRank() - 2});

      auto block_for_op =
          rewriter.create<BlockForOp>(op.getLoc(), res_types, 0, m, block_size, block_args, block_dims, ValueRange());
      Block *entry = block_for_op.addEntryBlock(op.getLoc());
      rewriter.setInsertionPointToStart(entry);
      auto new_dot_op = rewriter.create<DotOp>(op.getLoc(), block_for_op.getBlockArgInEntry(0), rhs);
      rewriter.create<BlockYieldOp>(op.getLoc(), ValueRange({new_dot_op.getResult()}), ValueRange());
      rewriter.replaceAllUsesWith(op.getResult(), block_for_op.getResult(0));
      rewriter.eraseOp(op);
      // dbg("dot tiling m done");
      return success();
    } else if (m <= block_size && k > block_size && n <= block_size) {
      assert(k % block_size == 0);
      // dbg("dot tiling k");

      auto zero_const_op = rewriter.create<ZeroOp>(op.getLoc(), res_type.getShape(), res_type.getElementType());
      SmallVector<Type> res_types({zero_const_op.getResult().getType()});
      SmallVector<Value> block_args({lhs, rhs});
      SmallVector<int64_t> block_dims({lhs_type.getRank() - 1, rhs_type.getRank() - 2});
      SmallVector<Value> block_init_args({zero_const_op.getResult()});

      auto block_for_op = rewriter.create<BlockForOp>(op.getLoc(), res_types, 0, k, block_size, block_args, block_dims,
                                                      block_init_args);
      Block *entry = block_for_op.addEntryBlock(op.getLoc());
      rewriter.setInsertionPointToStart(entry);
      auto new_dot_op =
          rewriter.create<DotOp>(op.getLoc(), block_for_op.getBlockArgInEntry(0), block_for_op.getBlockArgInEntry(1));
      auto new_add_op = rewriter.create<AddOp>(op.getLoc(), block_for_op.getInitArgInEntry(0), new_dot_op.getResult());

      rewriter.create<BlockYieldOp>(op.getLoc(), ValueRange(), ValueRange({new_add_op.getResult()}));
      rewriter.replaceAllUsesWith(op.getResult(), block_for_op.getResult(0));
      rewriter.eraseOp(op);
      // dbg("dot tiling k done");
      return success();
    } else if (m <= block_size && k <= block_size && n > block_size) {
      assert(n % block_size == 0);
      // dbg("dot tiling n");
      SmallVector<Type> res_types({res_type});
      SmallVector<Value> block_args({rhs});
      SmallVector<int64_t> block_dims({rhs_type.getRank() - 1});

      auto block_for_op =
          rewriter.create<BlockForOp>(op.getLoc(), res_types, 0, n, block_size, block_args, block_dims, ValueRange());
      Block *entry = block_for_op.addEntryBlock(op.getLoc());
      rewriter.setInsertionPointToStart(entry);
      auto new_dot_op = rewriter.create<DotOp>(op.getLoc(), lhs, block_for_op.getBlockArgInEntry(0));
      rewriter.create<BlockYieldOp>(op.getLoc(), ValueRange({new_dot_op.getResult()}), ValueRange());
      rewriter.replaceAllUsesWith(op.getResult(), block_for_op.getResult(0));
      rewriter.eraseOp(op);
      // dbg("dot tiling n done");
      return success();
    }
    // no tiling
    return failure();
  }
};

BlockForOp update_same_dim_block_args_for_single_result_op(BlockForOp op, Operation *fused_op, int block_arg_idx,
                                                           SmallVector<Value> new_block_args,
                                                           PatternRewriter &rewriter) {
  assert(fused_op->getNumResults() == 1);
  assert(new_block_args.size() > 0);

  SmallVector<Type> new_for_res_types(op.getResultTypes());
  int yield_res_num = cast<BlockYieldOp>(op.getRegion().front().getTerminator()).getNumBlocks();

  SmallVector<Value> new_for_block_args;
  for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
    if (i == block_arg_idx) {
      for (int j = 0; j < (int)new_block_args.size(); ++j) {
        new_for_block_args.push_back(new_block_args[j]);
      }
    } else {
      new_for_block_args.push_back(op.getBlockArg(i));
    }
  }
  SmallVector<int64_t> new_for_block_dims;
  for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
    if (i == block_arg_idx) {
      for (int j = 0; j < (int)new_block_args.size(); ++j) {
        new_for_block_dims.push_back(op.getBlockDim(i));
      }
    } else {
      new_for_block_dims.push_back(op.getBlockDim(i));
    }
  }
  SmallVector<Value> new_for_init_args(op.getInitArgs());

  auto new_for_op = rewriter.create<BlockForOp>(op.getLoc(), new_for_res_types, op.getLowerBoundAttr().getInt(),
                                                op.getUpperBoundAttr().getInt(), op.getStepAttr().getInt(),
                                                new_for_block_args, new_for_block_dims, new_for_init_args);
  Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
  rewriter.setInsertionPointToStart(new_entry);

  IRMapping val_map;
  val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
  for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
    if (i == block_arg_idx) {
      for (int j = 0; j < (int)new_block_args.size(); ++j) {
        val_map.map(new_block_args[j], new_for_op.getBlockArgInEntry(i + j));
      }
    } else if (i > block_arg_idx) {
      val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i + new_block_args.size() - 1));
    } else {
      val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
    }
  }
  for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
    val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
  }
  auto new_fused_op = rewriter.clone(*fused_op, val_map);
  infer_and_set_ret_type(new_fused_op);
  val_map.map(op.getBlockArgInEntry(block_arg_idx), new_fused_op->getResult(0));

  Block *entry = &op.getRegion().front();
  entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
    if (auto yield_op = dyn_cast<BlockYieldOp>(original_op)) {
      SmallVector<Value> new_blocks;
      SmallVector<Value> new_iters;
      for (int i = 0; i < (int)yield_op.getNumBlocks(); ++i) {
        new_blocks.push_back(val_map.lookup(yield_op.getBlock(i)));
      }
      for (int i = 0; i < (int)yield_op.getNumIters(); ++i) {
        new_iters.push_back(val_map.lookup(yield_op.getIter(i)));
      }
      rewriter.create<BlockYieldOp>(op.getLoc(), new_blocks, new_iters);
      return WalkResult::advance();
    } else {
      auto new_op = rewriter.clone(*original_op, val_map);
      infer_and_set_ret_type(new_op);
      for (size_t i = 0; i < original_op->getNumResults(); ++i) {
        val_map.map(original_op->getResult(i), new_op->getResult(i));
      }
      if (isa<MaskOp>(original_op)) {
        return WalkResult::skip();
      } else {
        return WalkResult::advance();
      }
    }
  });

  rewriter.replaceOp(op, new_for_op);
  check_no_use_for_results(&*fused_op);
  rewriter.eraseOp(fused_op);
  return new_for_op;
}

BlockForOp update_single_block_arg_for_single_result_op(BlockForOp op, Operation *fused_op, int block_arg_idx,
                                                        Value new_block_arg, PatternRewriter &rewriter) {
  return update_same_dim_block_args_for_single_result_op(op, fused_op, block_arg_idx, {new_block_arg}, rewriter);
}

BlockForOp add_block_args_for_single_result_op(BlockForOp op, Operation *fused_op, int op_used_block_arg_idx,
                                               Value used_val, int block_arg_pos, SmallVector<Value> new_block_args,
                                               SmallVector<int64_t> new_block_dims, PatternRewriter &rewriter) {
  assert(fused_op->getNumResults() == 1);
  assert(new_block_args.size() > 0);

  // result is appened to the end of block args
  int yield_res_num = cast<BlockYieldOp>(op.getRegion().front().getTerminator()).getNumBlocks();
  int yield_iter_num = cast<BlockYieldOp>(op.getRegion().front().getTerminator()).getNumIters();
  SmallVector<Type> new_for_res_types(op.getResultTypes());
  new_for_res_types.insert(new_for_res_types.begin() + yield_res_num, fused_op->getResult(0).getType());
  // entry block arg is inserted by block_arg_pos
  SmallVector<Value> new_for_block_args(op.getBlockArgs());
  new_for_block_args.insert(new_for_block_args.begin() + block_arg_pos, new_block_args.begin(), new_block_args.end());
  SmallVector<int64_t> new_for_block_dims(op.getBlockDims());
  new_for_block_dims.insert(new_for_block_dims.begin() + block_arg_pos, new_block_dims.begin(), new_block_dims.end());
  SmallVector<Value> new_for_init_args(op.getInitArgs());

  auto new_for_op = rewriter.create<BlockForOp>(op.getLoc(), new_for_res_types, op.getLowerBoundAttr().getInt(),
                                                op.getUpperBoundAttr().getInt(), op.getStepAttr().getInt(),
                                                new_for_block_args, new_for_block_dims, new_for_init_args);
  Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
  rewriter.setInsertionPointToStart(new_entry);

  IRMapping val_map;
  val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
  for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
    if (i == block_arg_pos) {
      for (int j = 0; j < (int)new_block_args.size(); ++j) {
        val_map.map(new_block_args[j], new_for_op.getBlockArgInEntry(i + j));
      }
    }
    if (i >= block_arg_pos) {
      val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i + new_block_args.size()));
    } else {
      val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
    }
  }
  for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
    val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
  }
  val_map.map(used_val, val_map.lookup(op.getBlockArgInEntry(op_used_block_arg_idx)));
  auto new_fused_op = rewriter.clone(*fused_op, val_map);
  infer_and_set_ret_type(new_fused_op);

  Block *entry = &op.getRegion().front();
  entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
    if (auto yield_op = dyn_cast<BlockYieldOp>(original_op)) {
      SmallVector<Value> new_blocks;
      SmallVector<Value> new_iters;
      for (int i = 0; i < (int)yield_op.getNumBlocks(); ++i) {
        new_blocks.push_back(val_map.lookup(yield_op.getBlock(i)));
      }
      assert(yield_op.getNumBlocks() == yield_res_num);
      assert(yield_op.getNumIters() == yield_iter_num);
      for (int i = 0; i < (int)yield_op.getNumIters(); ++i) {
        new_iters.push_back(val_map.lookup(yield_op.getIter(i)));
      }
      new_blocks.push_back(new_fused_op->getResult(0));
      rewriter.create<BlockYieldOp>(op.getLoc(), new_blocks, new_iters);
      return WalkResult::advance();
    } else {
      auto new_op = rewriter.clone(*original_op, val_map);
      infer_and_set_ret_type(new_op);
      for (size_t i = 0; i < original_op->getNumResults(); ++i) {
        val_map.map(original_op->getResult(i), new_op->getResult(i));
      }
      if (isa<MaskOp>(original_op)) {
        return WalkResult::skip();
      } else {
        return WalkResult::advance();
      }
    }
  });

  for (int i = 0; i < yield_res_num; ++i) {
    rewriter.replaceAllUsesWith(op->getResult(i), new_for_op->getResult(i));
  }
  assert(new_for_op->getNumResults() == yield_res_num + yield_iter_num + 1);
  for (int i = 0; i < yield_iter_num; ++i) {
    // +1 for new result
    rewriter.replaceAllUsesWith(op->getResult(yield_res_num + i), new_for_op->getResult(yield_res_num + 1 + i));
  }
  // rewriter.replaceOp(op, new_for_op);
  rewriter.eraseOp(op);
  rewriter.replaceAllUsesWith(fused_op->getResult(0), new_for_op->getResult(yield_res_num));
  check_no_use_for_results(&*fused_op);
  rewriter.eraseOp(fused_op);
  return new_for_op;
}

struct FuseOtherUserOfArgPattern : public OpRewritePattern<BlockForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(BlockForOp op, mlir::PatternRewriter &rewriter) const override {
    int64_t block_size = op.getStepAttr().getInt();
    for (int block_arg_idx = 0; block_arg_idx < op.getNumBlockArgs(); ++block_arg_idx) {
      auto arg = op.getBlockArg(block_arg_idx);
      auto dim = op.getBlockDim(block_arg_idx);
      if (arg.use_empty() || arg.hasOneUse())
        continue;
      for (auto &arg_use : arg.getUses()) {
        auto user = arg_use.getOwner();
        if (user == op)
          continue;
        if (auto reduce_op = dyn_cast<ReduceOp>(user)) {
          auto reduce_dim = reduce_op.getReduceDimensionAttr().getInt();
          auto reduce_init = reduce_op.getInit();
          auto reduce_type = reduce_op.getReduceType();
          assert(reduce_init == nullptr);
          if (reduce_dim != dim) {
            op->getParentOp()->dump();
            llvm_unreachable("not supported different reduce dim");
          }
          if (reduce_type == ReduceType::ADD || reduce_type == ReduceType::ANY) {
            auto reduce_res_type = cast<RankedTensorType>(reduce_op.getResult().getType());
            auto init_const_op = rewriter.create<ZeroOp>(reduce_op.getLoc(), reduce_res_type.getShape(),
                                                         reduce_res_type.getElementType());

            SmallVector<Type> new_for_res_types(op.getResultTypes());
            new_for_res_types.push_back(init_const_op.getResult().getType());
            SmallVector<Value> new_for_block_args(op.getBlockArgs());
            SmallVector<int64_t> new_for_block_dims(op.getBlockDims());
            SmallVector<Value> new_for_init_args(op.getInitArgs());
            new_for_init_args.push_back(init_const_op.getResult());

            auto new_for_op = rewriter.create<BlockForOp>(
                op.getLoc(), new_for_res_types, op.getLowerBoundAttr().getInt(), op.getUpperBoundAttr().getInt(),
                block_size, new_for_block_args, new_for_block_dims, new_for_init_args);
            Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
            Block *entry = &op.getRegion().front();
            rewriter.setInsertionPointToStart(new_entry);

            auto original_init_arg_num = op.getNumInitArgs();
            auto new_reduce_op = rewriter.create<ReduceOp>(op.getLoc(), new_for_op.getBlockArgInEntry(block_arg_idx),
                                                           new_for_op.getInitArgInEntry(original_init_arg_num),
                                                           reduce_dim, reduce_type, reduce_op.getKeepDim());

            IRMapping val_map;
            val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
            for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
              val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
            }
            for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
              val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
            }
            int new_yield_res_num = 0;
            int new_yield_iter_num = 0;
            entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
              if (auto yield_op = dyn_cast<BlockYieldOp>(original_op)) {
                SmallVector<Value> new_blocks;
                SmallVector<Value> new_iters;
                for (int i = 0; i < (int)yield_op.getNumBlocks(); ++i) {
                  new_blocks.push_back(val_map.lookup(yield_op.getBlock(i)));
                }
                for (int i = 0; i < (int)yield_op.getNumIters(); ++i) {
                  new_iters.push_back(val_map.lookup(yield_op.getIter(i)));
                }
                new_iters.push_back(new_reduce_op.getResult());
                rewriter.create<BlockYieldOp>(op.getLoc(), new_blocks, new_iters);
                new_yield_res_num = (int)new_blocks.size();
                new_yield_iter_num = (int)new_iters.size();
              } else {
                auto new_op = rewriter.clone(*original_op, val_map);
                infer_and_set_ret_type(new_op);
                for (size_t i = 0; i < original_op->getNumResults(); ++i) {
                  val_map.map(original_op->getResult(i), new_op->getResult(i));
                }
                if (isa<MaskOp>(original_op)) {
                  return WalkResult::skip();
                }
              }
              return WalkResult::advance();
            });
            // replace use
            for (int i = 0; i < new_yield_res_num; ++i) {
              rewriter.replaceAllUsesWith(op->getResult(i), new_for_op->getResult(i));
            }
            for (int i = 0; i < new_yield_iter_num - 1; ++i) {
              rewriter.replaceAllUsesWith(op->getResult(new_yield_res_num + i),
                                          new_for_op->getResult(new_yield_res_num + i));
            }
            rewriter.replaceAllUsesWith(reduce_op.getResult(),
                                        new_for_op->getResult(new_yield_res_num + new_yield_iter_num - 1));

            // erase
            check_no_use_for_results(&*op);
            rewriter.eraseOp(op);
            check_no_use_for_results(&*reduce_op);
            rewriter.eraseOp(reduce_op);
            return success();
          } else {
            op->getParentOp()->dump();
            llvm_unreachable("not supported reduce type");
          }
        } else if (auto bin_op = dyn_cast<BroadcastableBinaryOpInterface>(user)) {
          auto lhs = bin_op.getLhs();
          auto rhs = bin_op.getRhs();
          auto lhs_broadcasted_shape = bin_op.getLhsBroadcastedShape();
          auto rhs_broadcasted_shape = bin_op.getRhsBroadcastedShape();
          assert(!(lhs_broadcasted_shape[dim] == 1 && rhs_broadcasted_shape[dim] == 1));

          if (lhs == arg) {
            if (rhs_broadcasted_shape[dim] == 1) {
              assert(lhs_broadcasted_shape[dim] > 1);
              assert(lhs_broadcasted_shape[dim] % block_size == 0);
              // just clone
              llvm_unreachable("not supported");
            } else {
              assert(lhs_broadcasted_shape[dim] == rhs_broadcasted_shape[dim]);
              // dbg("fuse bin into for (bin lhs used)", block_arg_idx);
              // op->getParentOp()->dump();
              auto new_for_op = add_block_args_for_single_result_op(op, bin_op, block_arg_idx, arg, block_arg_idx + 1,
                                                                    {rhs}, {dim}, rewriter);
              // dbg("fuse bin into for (bin lhs used) done");
              // new_for_op->getParentOp()->dump();
              return success();
            }
          } else {
            assert(rhs == arg);
            if (lhs_broadcasted_shape[dim] == 1) {
              assert(rhs_broadcasted_shape[dim] > 1);
              assert(rhs_broadcasted_shape[dim] % block_size == 0);
              // just clone
              llvm_unreachable("not supported");
            } else {
              assert(lhs_broadcasted_shape[dim] == rhs_broadcasted_shape[dim]);
              auto new_for_op = add_block_args_for_single_result_op(op, bin_op, block_arg_idx, arg, block_arg_idx,
                                                                    {lhs}, {dim}, rewriter);
              return success();
            }
          }
        } else {
          op->getParentOp()->dump();
          llvm::errs() << "user: " << *user << "\n";
          llvm_unreachable("not supported");
        }
      }
    }

    return failure();
  }
};

struct FuseNonForBeforeForPattern : public OpRewritePattern<BlockForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(BlockForOp op, mlir::PatternRewriter &rewriter) const override {
    int64_t block_size = op.getStepAttr().getInt();
    for (int block_arg_idx = 0; block_arg_idx < op.getNumBlockArgs(); ++block_arg_idx) {
      auto arg = op.getBlockArg(block_arg_idx);
      auto dim = op.getBlockDim(block_arg_idx);
      auto def_op = arg.getDefiningOp();
      if (def_op != nullptr) {
        if (auto bin_op = dyn_cast<BroadcastableBinaryOpInterface>(def_op)) {
          // dbg("fuse bin op");
          if (!bin_op.getResult().hasOneUse())
            continue;

          auto lhs = bin_op.getLhs();
          auto rhs = bin_op.getRhs();
          auto lhs_broadcasted_shape = bin_op.getLhsBroadcastedShape();
          auto rhs_broadcasted_shape = bin_op.getRhsBroadcastedShape();
          assert(!(lhs_broadcasted_shape[dim] == 1 && rhs_broadcasted_shape[dim] == 1));
          if (lhs_broadcasted_shape[dim] == 1 || rhs_broadcasted_shape[dim] == 1) {
            Value operand = nullptr;
            if (lhs_broadcasted_shape[dim] == 1) {
              operand = rhs;
              assert(rhs_broadcasted_shape[dim] > 1);
            } else {
              operand = lhs;
              assert(lhs_broadcasted_shape[dim] > 1);
            }

            update_single_block_arg_for_single_result_op(op, def_op, block_arg_idx, operand, rewriter);
            // dbg("fuse bin op done");
            return success();
          } else {
            assert(lhs_broadcasted_shape[dim] == rhs_broadcasted_shape[dim]);
            assert(lhs_broadcasted_shape[dim] > 1 && lhs_broadcasted_shape[dim] % block_size == 0);

            update_same_dim_block_args_for_single_result_op(op, def_op, block_arg_idx, {lhs, rhs}, rewriter);
            // dbg("fuse bin op done");
            return success();
          }
        } else if (isa<Exp2Op, ExpOp, LogOp, NegOp, TanhOp>(def_op)) {
          // dbg("fuse unary op");
          assert(def_op->getNumOperands() == 1);
          assert(def_op->getNumResults() == 1);
          auto operand = def_op->getOperand(0);
          auto res = def_op->getResult(0);
          if (!res.hasOneUse())
            continue;

          auto operand_type = cast<RankedTensorType>(operand.getType());
          auto operand_shape = operand_type.getShape();
          assert(operand_shape[dim] > block_size);
          assert(operand_shape[dim] % block_size == 0);

          // op->getParentOp()->dump();
          auto new_for_op = update_single_block_arg_for_single_result_op(op, def_op, block_arg_idx, operand, rewriter);
          // dbg("fuse unary op done");
          // new_for_op->getParentOp()->dump();
          return success();
        } else if (auto mask_op = dyn_cast<MaskOp>(def_op)) {
          // dbg("fuse mask op");
          // op->getParentOp()->dump();
          auto res = mask_op.getResult();
          auto res_type = cast<RankedTensorType>(res.getType());
          auto res_shape = res_type.getShape();
          auto sizes = mask_op.getSizes();
          if (!res.hasOneUse()) {
            // replicate mask
            rewriter.setInsertionPointAfterValue(mask_op);
            for (auto &use : res.getUses()) {
              auto user = use.getOwner();
              if (user == op)
                continue;
              auto new_mask_op = rewriter.clone(*mask_op);
              use.assign(new_mask_op->getResult(0));
            }
            return success();
          }
          assert(res.hasOneUse());
          assert(res_shape[dim] > block_size);
          assert(res_shape[dim] % block_size == 0);

          SmallVector<Type> new_for_res_types(op.getResultTypes());
          SmallVector<Value> new_for_block_args;
          for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
            if (i == block_arg_idx)
              continue;
            new_for_block_args.push_back(op.getBlockArg(i));
          }
          SmallVector<int64_t> new_for_block_dims;
          for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
            if (i == block_arg_idx)
              continue;
            new_for_block_dims.push_back(op.getBlockDim(i));
          }
          SmallVector<Value> new_for_init_args(op.getInitArgs());

          auto new_for_op = rewriter.create<BlockForOp>(op.getLoc(), new_for_res_types, op.getLowerBoundAttr().getInt(),
                                                        op.getUpperBoundAttr().getInt(), block_size, new_for_block_args,
                                                        new_for_block_dims, new_for_init_args);
          Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
          rewriter.setInsertionPointToStart(new_entry);

          IRMapping val_map;
          val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
          for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
            if (i == block_arg_idx)
              continue;
            if (i > block_arg_idx) {
              val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i - 1));
            } else {
              val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
            }
          }
          for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
            val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
          }

          // compute attrs
          SmallVector<Value> new_starts;
          SmallVector<int64_t> new_sizes;
          for (int i = 0; i < (int)sizes.size(); ++i) {
            auto original_start = mask_op.getStart(i);
            if (i != dim) {
              new_starts.push_back(original_start);
              new_sizes.push_back(sizes[i]);
            } else {
              auto new_start =
                  rewriter.create<arith::AddIOp>(op.getLoc(), original_start, new_for_op.getIterArgInEntry());
              new_starts.push_back(new_start);
              assert(sizes[i] > block_size);
              assert(sizes[i] % block_size == 0);
              new_sizes.push_back(block_size);
            }
          }
          auto new_mask_op = rewriter.create<MaskOp>(op.getLoc(), new_starts, new_sizes, mask_op.getElementType());
          new_mask_op.getRegion().takeBody(mask_op.getRegion());
          val_map.map(op.getBlockArgInEntry(block_arg_idx), new_mask_op.getResult());

          Block *entry = &op.getRegion().front();
          entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
            auto new_op = rewriter.clone(*original_op, val_map);
            infer_and_set_ret_type(new_op);
            for (size_t i = 0; i < original_op->getNumResults(); ++i) {
              val_map.map(original_op->getResult(i), new_op->getResult(i));
            }
            if (isa<MaskOp>(original_op)) {
              return WalkResult::skip();
            } else {
              return WalkResult::advance();
            }
          });

          rewriter.replaceOp(op, new_for_op);
          rewriter.eraseOp(mask_op);
          // dbg("fuse mask op done");
          // new_for_op->getParentOp()->dump();
          return success();
        } else if (isa<PermuteOp, DotOp, BlockForOp>(def_op)) {
          // pass
          continue;
        } else {
          op->getParentOp()->dump();
          llvm::errs() << "def_op: " << *def_op << "\n";
          llvm_unreachable("not supported");
        }
      }
    }
    return failure();
  }
};

int64_t getResBlockDim(BlockForOp op, int res_i) {
  auto yield_op = cast<BlockYieldOp>(op.getRegion().front().getTerminator());
  auto block = yield_op.getBlock(res_i);
  auto block_shape = cast<RankedTensorType>(block.getType()).getShape();
  auto res = op->getResult(res_i);
  auto res_shape = cast<RankedTensorType>(res.getType()).getShape();

  int block_size = op.getStepAttr().getInt();
  int total_size = op.getUpperBoundAttr().getInt() - op.getLowerBoundAttr().getInt();
  int num = total_size / block_size;
  assert(block_shape.size() == res_shape.size());
  for (int i = 0; i < (int)block_shape.size(); ++i) {
    int n = (int)res_shape[i] / block_shape[i];
    if (n == num)
      return i;
  }
  op->getParentOp()->dump();
  llvm::errs() << "for op:" << *op << "\n";
  llvm::errs() << "res_i: " << res_i << "\n";
  llvm_unreachable("cannot find res block dim for res i");
}

BlockForOp update_single_res_for_single_result_op(BlockForOp op, Operation *fused_op, int res_i, Value op_res,
                                                  Value op_yield_res, SmallVector<Value> extra_new_block_args,
                                                  SmallVector<int64_t> extra_new_dims, PatternRewriter &rewriter) {
  assert(fused_op->getNumResults() == 1);
  auto res = fused_op->getResult(0);

  SmallVector<Type> new_for_res_types(op.getResultTypes());
  new_for_res_types[res_i] = res.getType();
  SmallVector<Value> new_for_block_args(op.getBlockArgs());
  for (auto &arg : extra_new_block_args) {
    new_for_block_args.push_back(arg);
  }
  SmallVector<int64_t> new_for_block_dims(op.getBlockDims());
  for (auto &dim : extra_new_dims) {
    new_for_block_dims.push_back(dim);
  }
  SmallVector<Value> new_for_init_args(op.getInitArgs());

  auto new_for_op = rewriter.create<BlockForOp>(op.getLoc(), new_for_res_types, op.getLowerBoundAttr().getInt(),
                                                op.getUpperBoundAttr().getInt(), op.getStepAttr().getInt(),
                                                new_for_block_args, new_for_block_dims, new_for_init_args);
  Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
  rewriter.setInsertionPointToStart(new_entry);

  IRMapping val_map;
  val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
  for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
    val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
  }
  for (int i = 0; i < (int)extra_new_block_args.size(); ++i) {
    val_map.map(extra_new_block_args[i], new_for_op.getBlockArgInEntry(op.getNumBlockArgs() + i));
  }
  for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
    val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
  }

  Block *entry = &op.getRegion().front();
  entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
    if (auto yield_op = dyn_cast<BlockYieldOp>(original_op)) {
      val_map.map(op_res, val_map.lookup(op_yield_res));
      auto new_fused_op = rewriter.clone(*fused_op, val_map);
      infer_and_set_ret_type(new_fused_op);
      assert(new_fused_op->getNumResults() == 1);
      val_map.map(op_yield_res, new_fused_op->getResult(0));
      rewriter.clone(*yield_op, val_map);
    } else {
      auto new_op = rewriter.clone(*original_op, val_map);
      infer_and_set_ret_type(new_op);
      for (size_t i = 0; i < original_op->getNumResults(); ++i) {
        val_map.map(original_op->getResult(i), new_op->getResult(i));
      }
      if (isa<MaskOp>(original_op)) {
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });

  rewriter.replaceOp(op, new_for_op);
  rewriter.replaceAllUsesWith(fused_op->getResult(0), new_for_op->getResult(res_i));
  check_no_use_for_results(&*fused_op);
  rewriter.eraseOp(fused_op);

  return new_for_op;
}

BlockForOp update_single_res_for_single_result_op_with_reduce_ops(BlockForOp op, Operation *fused_op, int res_i,
                                                                  Value op_res, Value op_yield_res,
                                                                  SmallVector<Value> extra_new_block_args,
                                                                  SmallVector<int64_t> extra_new_dims,
                                                                  SmallVector<ReduceOp> extra_reduce_ops,
                                                                  PatternRewriter &rewriter) {
  assert(fused_op->getNumResults() == 1);
  auto res = fused_op->getResult(0);

  auto check_reduce_ops_func = [&](SmallVector<ReduceOp> reduce_ops) {
    if (reduce_ops.size() == 0)
      return;
    auto reduce_op = reduce_ops[0];
    auto reduce_dim = reduce_op.getReduceDimensionAttr().getInt();
    auto reduce_init = reduce_op.getInit();
    auto reduce_type = reduce_op.getReduceType();
    assert(reduce_init == nullptr);
    assert(reduce_type == ReduceType::ADD || reduce_type == ReduceType::ANY);
    for (auto reduce_op : reduce_ops) {
      assert(reduce_op.getReduceDimensionAttr().getInt() == reduce_dim);
      assert(reduce_op.getInit() == nullptr);
      assert(reduce_op.getReduceType() == reduce_type);
    }
  };
  check_reduce_ops_func(extra_reduce_ops);
  SmallVector<Value> init_consts;
  for (auto reduce_op : extra_reduce_ops) {
    auto reduce_res_type = cast<RankedTensorType>(reduce_op.getResult().getType());
    auto init_const_op =
        rewriter.create<ZeroOp>(reduce_op.getLoc(), reduce_res_type.getShape(), reduce_res_type.getElementType());
    init_consts.push_back(init_const_op.getResult());
  }

  SmallVector<Type> new_for_res_types(op.getResultTypes());
  new_for_res_types[res_i] = res.getType();
  for (auto init_const : init_consts) {
    new_for_res_types.push_back(init_const.getType());
  }

  SmallVector<Value> new_for_block_args(op.getBlockArgs());
  for (auto &arg : extra_new_block_args) {
    new_for_block_args.push_back(arg);
  }
  SmallVector<int64_t> new_for_block_dims(op.getBlockDims());
  for (auto &dim : extra_new_dims) {
    new_for_block_dims.push_back(dim);
  }
  SmallVector<Value> new_for_init_args(op.getInitArgs());
  for (auto init_const : init_consts) {
    new_for_init_args.push_back(init_const);
  }

  auto new_for_op = rewriter.create<BlockForOp>(op.getLoc(), new_for_res_types, op.getLowerBoundAttr().getInt(),
                                                op.getUpperBoundAttr().getInt(), op.getStepAttr().getInt(),
                                                new_for_block_args, new_for_block_dims, new_for_init_args);
  Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
  rewriter.setInsertionPointToStart(new_entry);

  IRMapping val_map;
  val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
  for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
    val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
  }
  for (int i = 0; i < (int)extra_new_block_args.size(); ++i) {
    val_map.map(extra_new_block_args[i], new_for_op.getBlockArgInEntry(op.getNumBlockArgs() + i));
  }
  for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
    val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
  }

  Block *entry = &op.getRegion().front();
  entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
    if (auto yield_op = dyn_cast<BlockYieldOp>(original_op)) {
      val_map.map(op_res, val_map.lookup(op_yield_res));
      auto new_fused_op = rewriter.clone(*fused_op, val_map);
      infer_and_set_ret_type(new_fused_op);
      assert(new_fused_op->getNumResults() == 1);

      SmallVector<Value> new_blocks;
      SmallVector<Value> new_iters;
      for (int i = 0; i < (int)yield_op.getNumBlocks(); ++i) {
        if (i == res_i) {
          new_blocks.push_back(new_fused_op->getResult(0));
        } else {
          new_blocks.push_back(val_map.lookup(yield_op.getBlock(i)));
        }
      }
      for (int i = 0; i < (int)yield_op.getNumIters(); ++i) {
        new_iters.push_back(val_map.lookup(yield_op.getIter(i)));
      }
      for (int i = 0; i < (int)extra_reduce_ops.size(); ++i) {
        auto new_reduce_op = rewriter.create<ReduceOp>(
            op.getLoc(), val_map.lookup(op_yield_res), new_for_op.getInitArgInEntry(op.getNumInitArgs() + i),
            extra_reduce_ops[i].getReduceDimensionAttr().getInt(), extra_reduce_ops[i].getReduceType(),
            extra_reduce_ops[i].getKeepDim());
        new_iters.push_back(new_reduce_op.getResult());
      }
      rewriter.create<BlockYieldOp>(op.getLoc(), new_blocks, new_iters);
    } else {
      auto new_op = rewriter.clone(*original_op, val_map);
      infer_and_set_ret_type(new_op);
      for (size_t i = 0; i < original_op->getNumResults(); ++i) {
        val_map.map(original_op->getResult(i), new_op->getResult(i));
      }
      if (isa<MaskOp>(original_op)) {
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });

  rewriter.replaceOp(op, new_for_op);
  rewriter.replaceAllUsesWith(fused_op->getResult(0), new_for_op->getResult(res_i));
  check_no_use_for_results(&*fused_op);
  rewriter.eraseOp(fused_op);
  for (int i = 0; i < (int)extra_reduce_ops.size(); ++i) {
    rewriter.replaceAllUsesWith(extra_reduce_ops[i].getResult(),
                                new_for_op->getResult(new_for_op->getNumResults() - extra_reduce_ops.size() + i));
    check_no_use_for_results(&*extra_reduce_ops[i]);
    rewriter.eraseOp(extra_reduce_ops[i]);
  }

  return new_for_op;
}

BlockForOp update_single_res_for_reduce_ops(BlockForOp op, SmallVector<Operation *> fused_reduce_ops, int res_i,
                                            Value op_res, Value op_yield_res, PatternRewriter &rewriter) {
  SmallVector<ReduceOp> reduce_ops;
  for (auto fused_op : fused_reduce_ops) {
    auto reduce_op = dyn_cast<ReduceOp>(fused_op);
    assert(reduce_op != nullptr);
    reduce_ops.push_back(reduce_op);
  }

  auto reduce_op = reduce_ops[0];
  auto reduce_dim = reduce_op.getReduceDimensionAttr().getInt();
  auto reduce_init = reduce_op.getInit();
  auto reduce_type = reduce_op.getReduceType();
  assert(reduce_init == nullptr);
  assert(reduce_type == ReduceType::ADD || reduce_type == ReduceType::ANY);
  for (auto reduce_op : reduce_ops) {
    assert(reduce_op.getReduceDimensionAttr().getInt() == reduce_dim);
    assert(reduce_op.getInit() == nullptr);
    assert(reduce_op.getReduceType() == reduce_type);
  }

  SmallVector<Value> init_consts;
  for (auto reduce_op : reduce_ops) {
    auto reduce_res_type = cast<RankedTensorType>(reduce_op.getResult().getType());
    auto init_const_op =
        rewriter.create<ZeroOp>(reduce_op.getLoc(), reduce_res_type.getShape(), reduce_res_type.getElementType());
    init_consts.push_back(init_const_op.getResult());
  }
  SmallVector<Type> new_for_res_types;
  for (int i = 0; i < (int)op.getResultTypes().size(); ++i) {
    if (i == res_i)
      continue;
    new_for_res_types.push_back(op.getResultTypes()[i]);
  }
  for (auto init_const : init_consts) {
    new_for_res_types.push_back(init_const.getType());
  }
  SmallVector<Value> new_for_block_args(op.getBlockArgs());
  SmallVector<int64_t> new_for_block_dims(op.getBlockDims());
  SmallVector<Value> new_for_init_args(op.getInitArgs());
  for (auto init_const : init_consts) {
    new_for_init_args.push_back(init_const);
  }

  auto new_for_op = rewriter.create<BlockForOp>(op.getLoc(), new_for_res_types, op.getLowerBoundAttr().getInt(),
                                                op.getUpperBoundAttr().getInt(), op.getStepAttr().getInt(),
                                                new_for_block_args, new_for_block_dims, new_for_init_args);
  Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
  Block *entry = &op.getRegion().front();
  rewriter.setInsertionPointToStart(new_entry);

  auto original_init_arg_num = op.getNumInitArgs();

  IRMapping val_map;
  val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
  for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
    val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
  }
  for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
    val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
  }

  entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
    if (auto yield_op = dyn_cast<BlockYieldOp>(original_op)) {
      SmallVector<Value> new_blocks;
      SmallVector<Value> new_iters;
      for (int i = 0; i < (int)yield_op.getNumBlocks(); ++i) {
        if (i != res_i) {
          new_blocks.push_back(val_map.lookup(yield_op.getBlock(i)));
        }
      }
      for (int i = 0; i < (int)yield_op.getNumIters(); ++i) {
        new_iters.push_back(val_map.lookup(yield_op.getIter(i)));
      }
      for (int i = 0; i < (int)reduce_ops.size(); ++i) {
        auto new_reduce_op = rewriter.create<ReduceOp>(op.getLoc(), val_map.lookup(op_yield_res),
                                                       new_for_op.getInitArgInEntry(original_init_arg_num + i),
                                                       reduce_dim, reduce_type, reduce_op.getKeepDim());
        new_iters.push_back(new_reduce_op.getResult());
      }
      rewriter.create<BlockYieldOp>(op.getLoc(), new_blocks, new_iters);
    } else {
      auto new_op = rewriter.clone(*original_op, val_map);
      infer_and_set_ret_type(new_op);
      for (size_t i = 0; i < original_op->getNumResults(); ++i) {
        val_map.map(original_op->getResult(i), new_op->getResult(i));
      }
      if (isa<MaskOp>(original_op)) {
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });

  for (int i = 0; i < (int)reduce_ops.size(); ++i) {
    rewriter.replaceAllUsesWith(reduce_ops[i].getResult(),
                                new_for_op->getResult(new_for_op->getNumResults() - reduce_ops.size() + i));
    check_no_use_for_results(&*reduce_ops[i]);
    rewriter.eraseOp(reduce_ops[i]);
  }
  int yield_res_num = cast<BlockYieldOp>(op.getRegion().front().getTerminator()).getNumBlocks();
  int yield_iter_num = cast<BlockYieldOp>(op.getRegion().front().getTerminator()).getNumIters();
  for (int i = 0; i < yield_res_num - 1; ++i) {
    if (i >= res_i) {
      rewriter.replaceAllUsesWith(op->getResult(i + 1), new_for_op->getResult(i));
    } else {
      rewriter.replaceAllUsesWith(op->getResult(i), new_for_op->getResult(i));
    }
  }
  assert(new_for_op->getNumResults() == yield_res_num - 1 + yield_iter_num + (int)reduce_ops.size());
  for (int i = 0; i < yield_iter_num; ++i) {
    rewriter.replaceAllUsesWith(op->getResult(yield_res_num + i), new_for_op->getResult(yield_res_num - 1 + i));
  }
  check_no_use_for_results(&*op);
  rewriter.eraseOp(op);

  return new_for_op;
}

struct FuseNonForAfterForPattern : public OpRewritePattern<BlockForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(BlockForOp op, mlir::PatternRewriter &rewriter) const override {
    auto for_yield_op = cast<BlockYieldOp>(op.getRegion().front().getTerminator());
    int block_size = op.getStepAttr().getInt();
    int yield_res_num = for_yield_op.getNumBlocks();
    for (int res_i = 0; res_i < yield_res_num; ++res_i) {
      auto dim = getResBlockDim(op, res_i);
      auto op_res = op->getResult(res_i);
      auto op_yield_res = for_yield_op.getBlock(res_i);
      if (!op_res.hasOneUse())
        continue;
      auto user = *op_res.getUsers().begin();
      if (auto bin_op = dyn_cast<BroadcastableBinaryOpInterface>(user)) {
        // dbg("fuse later bin op");
        auto lhs = bin_op.getLhs();
        auto lhs_broadcasted_shape = bin_op.getLhsBroadcastedShape();
        auto rhs = bin_op.getRhs();
        auto rhs_broadcasted_shape = bin_op.getRhsBroadcastedShape();
        auto res = bin_op.getResult();
        auto res_type = cast<RankedTensorType>(res.getType());
        auto res_shape = res_type.getShape();
        assert(res_shape[dim] > block_size);
        assert(res_shape[dim] % block_size == 0);

        if (lhs == op_res) {
          assert(lhs_broadcasted_shape[dim] == res_shape[dim]);
          if (rhs_broadcasted_shape[dim] > 1) {
            assert(rhs_broadcasted_shape[dim] == lhs_broadcasted_shape[dim]);
            auto new_for_op = update_single_res_for_single_result_op(op, &*bin_op, res_i, op_res, op_yield_res, {rhs},
                                                                     {dim}, rewriter);

            // dbg("fuse latter bin op with rhs");
            // new_for_op->getParentOp()->dump();
            return success();
          } else {
            assert(rhs_broadcasted_shape[dim] == 1);
            auto new_for_op =
                update_single_res_for_single_result_op(op, &*bin_op, res_i, op_res, op_yield_res, {}, {}, rewriter);
            // dbg("fuse latter bin op");
            return success();
          }
        } else {
          assert(rhs == op_res);
          op->getParentOp()->dump();
          llvm_unreachable("not supported");
        }
      } else if (isa<Exp2Op, ExpOp, LogOp, NegOp, TanhOp>(user)) {
        // dbg("fuse latter unary op before");
        // op->getParentOp()->dump();
        assert(user->getNumOperands() == 1);
        assert(user->getNumResults() == 1);
        auto operand = user->getOperand(0);
        auto res = user->getResult(0);
        auto new_for_op =
            update_single_res_for_single_result_op(op, &*user, res_i, op_res, op_yield_res, {}, {}, rewriter);
        // dbg("fuse latter unary op done");
        // new_for_op->getParentOp()->dump();
        return success();
      } else if (auto reduce_op = dyn_cast<ReduceOp>(user)) {
        // dbg("fuse later reduce op");
        // op->getParentOp()->dump();
        auto reduce_dim = reduce_op.getReduceDimensionAttr().getInt();
        auto reduce_init = reduce_op.getInit();
        auto reduce_type = reduce_op.getReduceType();
        assert(reduce_init == nullptr);
        if (reduce_dim != dim) {
          op->getParentOp()->dump();
          llvm_unreachable("not supported different reduce dim");
        }
        if (reduce_type == ReduceType::ADD || reduce_type == ReduceType::ANY) {
          auto new_for_op = update_single_res_for_reduce_ops(op, {user}, res_i, op_res, op_yield_res, rewriter);
          // dbg("fuse later reduce op done");
          // new_for_op->getParentOp()->dump();
          return success();
        } else {
          op->getParentOp()->dump();
          llvm_unreachable("not supported reduce type");
        }
      } else if (isa<BlockForOp>(user)) {
        // pass
        continue;
      } else {
        op->getParentOp()->dump();
        llvm::errs() << "user: " << *user << "\n";
        llvm::errs() << "res_i: " << res_i << ", dim: " << dim << "\n";
        llvm_unreachable("not supported");
      }
    }
    // TODO
    return failure();
  }
};

// TODO: better workflow: replicate the for op, and fuse different use into them, then fuse the for ops and simplify
// them
struct FuseMultipleUseOfResPattern : public OpRewritePattern<BlockForOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  template <typename Op>
  bool isAllSameTypeOp(ArrayRef<Operation *> ops) const {
    bool is_same = true;
    for (int i = 0; i < (int)ops.size(); ++i) {
      if (!isa<Op>(ops[i])) {
        is_same = false;
        break;
      }
    }
    return is_same;
  }
  mlir::LogicalResult matchAndRewrite(BlockForOp op, mlir::PatternRewriter &rewriter) const override {
    auto for_yield_op = cast<BlockYieldOp>(op.getRegion().front().getTerminator());
    int64_t block_size = op.getStepAttr().getInt();
    int yield_res_num = for_yield_op.getNumBlocks();
    for (int res_i = 0; res_i < yield_res_num; ++res_i) {
      auto dim = getResBlockDim(op, res_i);
      auto op_res = op->getResult(res_i);
      auto op_yield_res = for_yield_op.getBlock(res_i);
      if (op_res.hasOneUse())
        continue;
      assert(!op_res.use_empty());

      SmallVector<Operation *> users;
      for (auto user : op_res.getUsers()) {
        users.push_back(&*user);
      }
      assert(users.size() > 1);
      if (isAllSameTypeOp<ReduceOp>(users)) {
        auto is_same_attr_reduce_op = [&](ArrayRef<Operation *> reduce_ops) {
          assert(isa<ReduceOp>(reduce_ops[0]));
          auto reduce_dim = cast<ReduceOp>(reduce_ops[0]).getReduceDimensionAttr().getInt();
          auto reduce_init = cast<ReduceOp>(reduce_ops[0]).getInit();
          auto reduce_type = cast<ReduceOp>(reduce_ops[0]).getReduceType();
          for (auto op : reduce_ops) {
            auto reduce_op = cast<ReduceOp>(op);
            if (reduce_op.getReduceDimensionAttr().getInt() != reduce_dim)
              return false;
            if (reduce_op.getInit() != reduce_init)
              return false;
            if (reduce_op.getReduceType() != reduce_type)
              return false;
          }
          return true;
        };

        if (!is_same_attr_reduce_op(users)) {
          op->getParentOp()->dump();
          llvm_unreachable("not supported different reduce attr");
        }

        auto reduce_dim = cast<ReduceOp>(users[0]).getReduceDimensionAttr().getInt();
        auto reduce_init = cast<ReduceOp>(users[0]).getInit();
        assert(reduce_init == nullptr);
        if (reduce_dim != dim) {
          op->getParentOp()->dump();
          llvm_unreachable("not supported different reduce dim with for blocking dim");
        }
        auto reduce_type = cast<ReduceOp>(users[0]).getReduceType();
        if (reduce_type != ReduceType::ADD && reduce_type != ReduceType::ANY) {
          op->getParentOp()->dump();
          llvm_unreachable("not supported reduce type other than ADD and ANY");
        }

        auto new_for_op = update_single_res_for_reduce_ops(op, users, res_i, op_res, op_yield_res, rewriter);
        return success();
      } else if (users.size() == 2) {
        auto user0 = users[0];
        auto user1 = users[1];
        if (isa<BroadcastableBinaryOpInterface>(user0) && isa<ReduceOp>(user1)) {
          // pass
        } else if (isa<ReduceOp>(user0) && isa<BroadcastableBinaryOpInterface>(user1)) {
          std::swap(user0, user1);
        } else if (isa<MulOp>(user0) && isa<AddOp>(user1)) {
          // FIXME: harcode for keyformer
          llvm_unreachable("need fix for keyformer");
          // auto mul_op = cast<MulOp>(user0);
          // auto add_op = cast<AddOp>(user1);
          // assert(cast<RankedTensorType>(mul_op.getRhs().getType()).getShape().size() == 1);
          // assert(cast<RankedTensorType>(mul_op.getRhs().getType()).getShape()[0] == 1);

          // // log(arg) -> neg -> add
          // auto neg_op = add_op.getRhs().getDefiningOp<NegOp>();
          // assert(neg_op != nullptr);
          // assert(neg_op.hasOneUse());
          // auto log_op = neg_op.getOperand().getDefiningOp<LogOp>();
          // assert(log_op != nullptr);
          // assert(log_op.hasOneUse());
          // assert(log_op.getOperand().getDefiningOp() == nullptr);
          // // move
          // log_op->moveBefore(op);
          // neg_op->moveBefore(op);

          // // now we can fuse
          // SmallVector<Type> new_for_res_types(op.getResultTypes());
          // new_for_res_types[res_i] = mul_op.getResult().getType();
        } else {
          op->getParentOp()->dump();
          llvm::errs() << "unreachable" << "\n";
          return failure();
          llvm_unreachable("not supported");
        }
        // dbg("fuse bin op and reduce op into for");
        // op->getParentOp()->dump();
        // user0 is bin op, user1 is reduce op
        auto bin_op = cast<BroadcastableBinaryOpInterface>(user0);
        auto reduce_op = cast<ReduceOp>(user1);
        auto lhs = bin_op.getLhs();
        auto lhs_broadcasted_shape = bin_op.getLhsBroadcastedShape();
        auto rhs = bin_op.getRhs();
        auto rhs_broadcasted_shape = bin_op.getRhsBroadcastedShape();
        auto res = bin_op.getResult();
        auto res_type = cast<RankedTensorType>(res.getType());
        auto res_shape = res_type.getShape();
        auto reduce_dim = reduce_op.getReduceDimensionAttr().getInt();
        auto reduce_init = reduce_op.getInit();
        auto reduce_type = reduce_op.getReduceType();
        assert(reduce_init == nullptr);
        assert(reduce_type == ReduceType::ADD || reduce_type == ReduceType::ANY);
        assert(reduce_dim == dim);
        assert(res_shape[dim] > block_size);
        assert(res_shape[dim] % block_size == 0);

        if (lhs == op_res) {
          assert(lhs_broadcasted_shape[dim] == res_shape[dim]);
          if (rhs_broadcasted_shape[dim] > 1) {
            assert(rhs_broadcasted_shape[dim] == lhs_broadcasted_shape[dim]);
            auto new_for_op = update_single_res_for_single_result_op_with_reduce_ops(
                op, bin_op, res_i, op_res, op_yield_res, {rhs}, {dim}, {reduce_op}, rewriter);
            // dbg("fuse bin op and reduce op with rhs into for done");
            // new_for_op->getParentOp()->dump();
            return success();
          } else {
            assert(rhs_broadcasted_shape[dim] == 1);
            auto new_for_op = update_single_res_for_single_result_op_with_reduce_ops(
                op, bin_op, res_i, op_res, op_yield_res, {}, {}, {reduce_op}, rewriter);
            // dbg("fuse bin op and reduce op into for done");
            // new_for_op->getParentOp()->dump();
            return success();
          }
        } else {
          assert(rhs == op_res);
          op->getParentOp()->dump();
          llvm_unreachable("not supported");
        }
      } else {
        op->getParentOp()->dump();
        llvm_unreachable("not supported");
      }
    }
    return failure();
  }
};

struct FuseForBeforeForPattern : public OpRewritePattern<BlockForOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(BlockForOp op, mlir::PatternRewriter &rewriter) const override {
    int64_t block_size = op.getStepAttr().getInt();
    for (int block_arg_idx = 0; block_arg_idx < op.getNumBlockArgs(); ++block_arg_idx) {
      auto arg = op.getBlockArg(block_arg_idx);
      auto dim = op.getBlockDim(block_arg_idx);
      auto def_op = arg.getDefiningOp();
      if (def_op != nullptr && isa<BlockForOp>(def_op)) {
        auto prev_for_op = cast<BlockForOp>(def_op);
        if (prev_for_op->getNumResults() == 1) {
          // dbg("fuse for");
          // op->getParentOp()->dump();

          assert(prev_for_op.getStepAttr().getInt() == block_size);
          SmallVector<Type> new_res_types(op.getResultTypes());
          SmallVector<Value> new_block_args;
          for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
            if (i == block_arg_idx) {
              for (int j = 0; j < (int)prev_for_op.getNumBlockArgs(); ++j) {
                new_block_args.push_back(prev_for_op.getBlockArg(j));
              }
            } else {
              new_block_args.push_back(op.getBlockArg(i));
            }
          }
          SmallVector<int64_t> new_dims;
          for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
            if (i == block_arg_idx) {
              for (int j = 0; j < (int)prev_for_op.getNumBlockArgs(); ++j) {
                new_dims.push_back(prev_for_op.getBlockDim(j));
              }
            } else {
              new_dims.push_back(op.getBlockDim(i));
            }
          }
          SmallVector<Value> new_init_args(prev_for_op.getInitArgs());
          for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
            new_init_args.push_back(op.getInitArg(i));
          }

          auto new_for_op = rewriter.create<BlockForOp>(op.getLoc(), new_res_types, op.getLowerBoundAttr().getInt(),
                                                        op.getUpperBoundAttr().getInt(), op.getStepAttr().getInt(),
                                                        new_block_args, new_dims, new_init_args);
          Block *new_entry = new_for_op.addEntryBlock(op.getLoc());
          rewriter.setInsertionPointToStart(new_entry);

          IRMapping val_map;
          val_map.map(prev_for_op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
          val_map.map(op.getIterArgInEntry(), new_for_op.getIterArgInEntry());
          for (int i = 0; i < (int)op.getNumBlockArgs(); ++i) {
            if (i == block_arg_idx) {
              for (int j = 0; j < (int)prev_for_op.getNumBlockArgs(); ++j) {
                val_map.map(prev_for_op.getBlockArgInEntry(j), new_for_op.getBlockArgInEntry(i + j));
              }
            } else if (i > block_arg_idx) {
              val_map.map(op.getBlockArgInEntry(i),
                          new_for_op.getBlockArgInEntry(i + prev_for_op.getNumBlockArgs() - 1));
            } else {
              val_map.map(op.getBlockArgInEntry(i), new_for_op.getBlockArgInEntry(i));
            }
          }
          for (int i = 0; i < (int)prev_for_op.getNumInitArgs(); ++i) {
            val_map.map(prev_for_op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i));
          }
          for (int i = 0; i < (int)op.getNumInitArgs(); ++i) {
            val_map.map(op.getInitArgInEntry(i), new_for_op.getInitArgInEntry(i + prev_for_op.getNumInitArgs()));
          }

          Block *prev_entry = &prev_for_op.getRegion().front();
          prev_entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
            if (auto yield_op = dyn_cast<BlockYieldOp>(original_op)) {
              assert(yield_op.getNumBlocks() == 1);
              assert(yield_op.getNumIters() == 0);
              val_map.map(op.getBlockArgInEntry(block_arg_idx), val_map.lookup(yield_op.getBlock(0)));
              return WalkResult::advance();
            } else {
              auto new_op = rewriter.clone(*original_op, val_map);
              infer_and_set_ret_type(new_op);
              for (size_t i = 0; i < original_op->getNumResults(); ++i) {
                val_map.map(original_op->getResult(i), new_op->getResult(i));
              }
              if (isa<MaskOp>(original_op)) {
                return WalkResult::skip();
              } else {
                return WalkResult::advance();
              }
            }
          });
          Block *entry = &op.getRegion().front();
          entry->walk<WalkOrder::PreOrder>([&](Operation *original_op) {
            auto new_op = rewriter.clone(*original_op, val_map);
            infer_and_set_ret_type(new_op);
            for (size_t i = 0; i < original_op->getNumResults(); ++i) {
              val_map.map(original_op->getResult(i), new_op->getResult(i));
            }
            if (isa<MaskOp>(original_op)) {
              return WalkResult::skip();
            } else {
              return WalkResult::advance();
            }
          });

          rewriter.replaceOp(op, new_for_op);
          check_no_use_for_results(&*prev_for_op);
          rewriter.eraseOp(prev_for_op);
          // dbg("fuse for done");
          // new_for_op->getParentOp()->dump();
          return success();
        } else {
          op->getParentOp()->dump();
          llvm_unreachable("not supported for prev op with multiple results");
        }
      }
    }
    return failure();
  }
};

class TilingPass : public ::mlir::asuka::impl::AsukaTilingBase<TilingPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp k = getOperation();

    patterns.add<TilingDotPattern>(context);
    patterns.add<FuseNonForBeforeForPattern>(context);
    patterns.add<FuseNonForAfterForPattern>(context);
    patterns.add<FuseOtherUserOfArgPattern>(context);
    patterns.add<FuseForBeforeForPattern>(context);
    patterns.add<FuseMultipleUseOfResPattern>(context);
    if (applyPatternsAndFoldGreedily(k, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTilingPass() { return std::make_unique<TilingPass>(); }

} // namespace mlir::asuka