#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/AsukaTriton/IR/AsukaTritonDialect.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_CONVERTASUKATOASUKATRITON
#include "asuka/Conversion/AsukaToAsukaTriton/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

SmallVector<int64_t> getStrides(RankedTensorType type) {
  SmallVector<int64_t> strides(type.getRank(), 1);
  for (int i = (int)type.getRank() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * type.getShape()[i + 1];
  }
  return strides;
}

Value getConst(int val, OpBuilder &builder, Location loc) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(val)).getResult();
}

Value mul(Value a, Value b, OpBuilder &builder, Location loc) {
  return builder.create<arith::MulIOp>(loc, a, b).getResult();
}

Value add(Value a, Value b, OpBuilder &builder, Location loc) {
  return builder.create<arith::AddIOp>(loc, a, b).getResult();
}

struct ParallelForConversion : public OpConversionPattern<ParallelForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ParallelForOp para_for_op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kernel_op = para_for_op->getParentOfType<KernelOp>();
    assert(kernel_op);
    assert(kernel_op.hasParallelMap());
    auto para_maps = kernel_op.getParallelMaps();

    // build pointer outside device_kernel
    SmallVector<Value> dev_args;
    SmallVector<Value> dev_out_args;
    for (auto arg : para_for_op->getOperands()) {
      assert(isa<RankedTensorType>(arg.getType()));
      auto ptr_op = rewriter.create<triton::PointerOfOp>(para_for_op.getLoc(), arg);
      dev_args.push_back(ptr_op.getResult());
    }
    for (auto res : para_for_op->getResults()) {
      assert(isa<RankedTensorType>(res.getType()));
      auto tensor_type = cast<RankedTensorType>(res.getType());
      auto empty_ptr_op = rewriter.create<triton::EmptyPointerOp>(para_for_op.getLoc(), TypeAttr::get(tensor_type));
      dev_args.push_back(empty_ptr_op.getResult());
      dev_out_args.push_back(empty_ptr_op.getResult());
    }
    SmallVector<int64_t> grid;
    for (auto &map : para_maps) {
      grid.push_back(map.unit_num);
    }

    auto dev_kernel_op = rewriter.create<triton::DeviceKernelOp>(para_for_op.getLoc(), dev_args, grid);
    auto dev_block = &dev_kernel_op.getRegion().front();
    auto para_for_block = &para_for_op.getRegion().front();
    rewriter.setInsertionPointToStart(dev_block);

    IRMapping val_map;
    auto build_block_ptr_for_arg = [&](int idx, Type original_type, Value dev_ptr_in_entry) {
      auto arg_type = cast<RankedTensorType>(original_type);
      auto arg_strides = getStrides(arg_type);
      SmallVector<bool> recorded_strides(arg_strides.size(), true);
      auto base_ptr = dev_ptr_in_entry;
      Value base_offset = nullptr;
      auto loc = para_for_op.getLoc();
      for (auto en : llvm::enumerate(para_maps)) {
        auto map_i = en.index();
        auto iter_arg = dev_kernel_op.getIterArgInEntry(map_i);
        auto map = en.value();
        if (map.arg_dims[idx] >= 0) {
          // parallelized
          if (map.size_per_unit == 1) {
            // squeezed
            recorded_strides[map.arg_dims[idx]] = false;
          }
          // optimize: if unit_num == 1, no need to calculate offset
          // TODO: verify
          if (map.unit_num == 1) {
            continue;
          }
          auto cur_stride = arg_strides[map.arg_dims[idx]];
          auto cur_stride_val = getConst(cur_stride, rewriter, loc);
          auto size_per_unit_val = getConst(map.size_per_unit, rewriter, loc);
          auto block_off_val = mul(iter_arg, size_per_unit_val, rewriter, loc);
          auto off_val = mul(block_off_val, cur_stride_val, rewriter, loc);
          if (base_offset) {
            base_offset = add(base_offset, off_val, rewriter, loc);
          } else {
            base_offset = off_val;
          }
        }
      }
      if (base_offset == nullptr) {
        // zero
        base_offset = getConst(0, rewriter, loc);
      }

      auto block_arg = para_for_block->getArgument(grid.size() + idx);
      SmallVector<int64_t> shape(cast<RankedTensorType>(block_arg.getType()).getShape());
      SmallVector<int64_t> strides;
      for (auto [stride, recorded] : llvm::zip(arg_strides, recorded_strides)) {
        if (recorded) {
          strides.push_back(stride);
        }
      }
      // dbg(i, shape, arg_strides, recorded_strides, strides);
      assert(strides.size() == shape.size());
      // always 0, the corresponding offset is in base_offset
      SmallVector<int64_t> offsets(shape.size(), 0);
      // block_shape is the same as shape
      auto block_ptr_op = rewriter.create<triton::BlockPointerOfOp>(para_for_op.getLoc(), base_ptr, base_offset, shape,
                                                                    strides, offsets, shape);
      // then load
      auto load_op = rewriter.create<triton::BlockLoadOp>(para_for_op.getLoc(), block_ptr_op.getResult());
      val_map.map(block_arg, load_op.getResult());
    };

    auto build_block_ptr_for_res = [&](int idx, Type original_type, Type block_type, Value dev_ptr_in_entry) -> Value {
      auto res_type = cast<RankedTensorType>(original_type);
      auto res_strides = getStrides(res_type);
      SmallVector<bool> recorded_strides(res_strides.size(), true);
      auto base_ptr = dev_ptr_in_entry;
      Value base_offset = nullptr;
      auto loc = para_for_op.getLoc();
      for (auto en : llvm::enumerate(para_maps)) {
        auto map_i = en.index();
        auto iter_arg = dev_kernel_op.getIterArgInEntry(map_i);
        auto map = en.value();
        if (map.res_dims[idx] >= 0) {
          // parallelized
          if (map.size_per_unit == 1) {
            // squeezed
            recorded_strides[map.res_dims[idx]] = false;
          }
          // optimize: if unit_num == 1, no need to calculate offset
          // TODO: verify
          if (map.unit_num == 1) {
            continue;
          }
          auto cur_stride = res_strides[map.res_dims[idx]];
          auto cur_stride_val = getConst(cur_stride, rewriter, loc);
          auto size_per_unit_val = getConst(map.size_per_unit, rewriter, loc);
          auto block_off_val = mul(iter_arg, size_per_unit_val, rewriter, loc);
          auto off_val = mul(block_off_val, cur_stride_val, rewriter, loc);
          if (base_offset) {
            base_offset = add(base_offset, off_val, rewriter, loc);
          } else {
            base_offset = off_val;
          }
        }
      }
      if (base_offset == nullptr) {
        // zero
        base_offset = getConst(0, rewriter, loc);
      }

      SmallVector<int64_t> shape(cast<RankedTensorType>(block_type).getShape());
      SmallVector<int64_t> strides;
      for (auto [stride, recorded] : llvm::zip(res_strides, recorded_strides)) {
        if (recorded) {
          strides.push_back(stride);
        }
      }
      // dbg(i, shape, arg_strides, recorded_strides, strides);
      assert(strides.size() == shape.size());
      // always 0, the corresponding offset is in base_offset
      SmallVector<int64_t> offsets(shape.size(), 0);
      // block_shape is the same as shape
      auto block_ptr_op = rewriter.create<triton::BlockPointerOfOp>(para_for_op.getLoc(), base_ptr, base_offset, shape,
                                                                    strides, offsets, shape);
      return block_ptr_op.getResult();
    };

    // make block ptr for in_arg and mapping its result
    // FIXME: dev_block's args contains out_args(len not match)
    for (auto en : llvm::enumerate(
             llvm::zip(para_for_op->getOperands(), llvm::drop_begin(dev_block->getArguments(), grid.size())))) {
      auto i = en.index();
      auto [arg, dev_in_arg] = en.value();
      build_block_ptr_for_arg(i, arg.getType(), dev_in_arg);
    }
    // make block ptr for out_arg
    auto para_for_yield_op = cast<ParallelYieldOp>(para_for_block->getTerminator());
    SmallVector<Value> out_block_ptrs;
    for (auto en : llvm::enumerate(
             llvm::zip(para_for_op->getResults(), para_for_yield_op->getOperands(),
                       llvm::drop_begin(dev_block->getArguments(), grid.size() + para_for_op->getNumOperands())))) {
      auto i = en.index();
      auto [res, block_res, dev_out_arg] = en.value();
      auto block_ptr = build_block_ptr_for_res(i, res.getType(), block_res.getType(), dev_out_arg);
      out_block_ptrs.push_back(block_ptr);
    }
    // map iter
    for (size_t i = 0; i < grid.size(); ++i) {
      auto original_iter = para_for_op.getIterArgInEntry(i);
      auto new_iter = dev_kernel_op.getIterArgInEntry(i);

      auto size_per_unit = para_maps[i].size_per_unit;
      auto offset = getConst(size_per_unit, rewriter, para_for_op.getLoc());
      auto real_iter = mul(new_iter, offset, rewriter, para_for_op.getLoc());
      val_map.map(original_iter, real_iter);
    }

    // clone all other ops (no need for infer type due to no change)
    for (auto &op : *para_for_block) {
      if (auto yield_op = dyn_cast<ParallelYieldOp>(op)) {
        // store block
        for (auto [block_ptr, res] : llvm::zip(out_block_ptrs, yield_op->getOperands())) {
          rewriter.create<triton::BlockStoreOp>(para_for_op.getLoc(), block_ptr, val_map.lookup(res));
        }
      } else {
        auto new_op = rewriter.clone(op, val_map);
        for (auto [old_res, new_res] : llvm::zip(op.getResults(), new_op->getResults())) {
          val_map.map(old_res, new_res);
        }
      }
    }

    rewriter.setInsertionPointAfter(dev_kernel_op);
    for (auto [res, out_arg] : llvm::zip(para_for_op->getResults(), dev_out_args)) {
      auto tensor_from_op = rewriter.create<triton::TensorFromOp>(para_for_op.getLoc(), out_arg);
      rewriter.replaceAllUsesWith(res, tensor_from_op.getResult());
    }
    rewriter.eraseOp(para_for_op);
    return success();
  }
};

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

struct FusePermuteConversion : public OpConversionPattern<PermuteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(PermuteOp permute_op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto operand = permute_op.getOperand();
    auto block_load_op = operand.getDefiningOp<triton::BlockLoadOp>();
    if (block_load_op == nullptr)
      return failure();
    if (block_load_op.getResult().hasOneUse() == false)
      return failure();

    auto block_ptr_op = block_load_op.getSrcPointer().getDefiningOp<triton::BlockPointerOfOp>();
    if (block_ptr_op == nullptr)
      return failure();
    if (block_ptr_op.getResult().hasOneUse() == false)
      return failure();

    auto dims = permute_op.getDims();
    auto shape = block_ptr_op.getShape();
    auto stride = block_ptr_op.getStride();
    auto offset = block_ptr_op.getOffset();
    auto block_shape = block_ptr_op.getBlockShape();
    auto order = block_ptr_op.getOrder();
    assert(dims.size() == block_shape.size());
    SmallVector<int64_t> new_shape(shape.size());
    SmallVector<int64_t> new_stride(stride.size());
    SmallVector<int64_t> new_offset(offset.size());
    SmallVector<int64_t> new_block_shape(block_shape.size());
    SmallVector<int64_t> new_order(order.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      new_shape[dims[i]] = shape[i];
      new_stride[dims[i]] = stride[i];
      new_offset[dims[i]] = offset[i];
      new_block_shape[dims[i]] = block_shape[i];
      new_order[dims[i]] = order[i];
    }
    block_ptr_op.setShape(new_shape);
    block_ptr_op.setStride(new_stride);
    block_ptr_op.setOffset(new_offset);
    block_ptr_op.setBlockShape(new_block_shape);
    block_ptr_op.setOrder(new_order);
    // infer type
    infer_type(block_ptr_op);
    infer_type(block_load_op);
    // replace uses
    rewriter.replaceAllUsesWith(permute_op.getResult(), block_load_op.getResult());
    rewriter.eraseOp(permute_op);
    return success();
  }
};

struct DynamicBlockForConversion : public OpConversionPattern<DynamicBlockForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DynamicBlockForOp for_op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // match
    // need to be executed after ParallelForConversion
    auto para_for_op = for_op->getParentOfType<ParallelForOp>();
    if (para_for_op != nullptr)
      return failure();

    // all block arg must be result of block load
    SmallVector<triton::BlockLoadOp> block_load_ops;
    SmallVector<triton::BlockPointerOfOp> block_ptr_ops;
    SmallVector<int> block_dims;
    for (int i = 0; i < for_op.getNumBlockArgs(); ++i) {
      auto arg = for_op.getBlockArg(i);
      auto block_load_op = arg.getDefiningOp<triton::BlockLoadOp>();
      if (block_load_op == nullptr)
        return failure();
      if (block_load_op.getResult().hasOneUse() == false)
        return failure();
      auto block_ptr_op = block_load_op.getSrcPointer().getDefiningOp<triton::BlockPointerOfOp>();
      if (block_ptr_op == nullptr)
        return failure();
      if (block_ptr_op.getResult().hasOneUse() == false)
        return failure();

      block_load_ops.push_back(block_load_op);
      block_ptr_ops.push_back(block_ptr_op);

      auto dim = for_op.getBlockDim(i);
      block_dims.push_back(dim);
    }

    // matched
    // dynamic_block_for -> scf::for
    auto lb_v = for_op.getLowerBound();
    auto ub_v = for_op.getUpperBound();
    int step = for_op.getStepAttr().getInt();
    auto step_v = getConst(step, rewriter, for_op.getLoc());

    // update block_ptr_ops
    SmallVector<Value> block_ptrs;
    for (auto [block_ptr_op, block_load_op, block_dim] : llvm::zip(block_ptr_ops, block_load_ops, block_dims)) {
      auto block_shape = block_ptr_op.getBlockShape();
      SmallVector<int64_t> new_block_shape(block_shape);
      assert(block_shape[block_dim] > step);
      assert(block_shape[block_dim] % step == 0);
      new_block_shape[block_dim] = step;
      // dynamic_for may have dynamic lb, which will change the start of tensor blocking
      // FIXME: we need an valued offset

      // we already know block_ptr_op -> block_load_op -> for_op without any other use
      // we move the two ops right before for_op to make following operation valid
      // rewriter.moveOpBefore(block_load_op, for_op);
      // rewriter.moveOpBefore(block_ptr_op, block_load_op);
      block_load_op->moveBefore(for_op);
      block_ptr_op->moveBefore(block_load_op);

      auto stride = block_ptr_op.getStride();
      rewriter.setInsertionPoint(block_ptr_op);
      auto offset = mul(lb_v, getConst(stride[block_dim], rewriter, for_op.getLoc()), rewriter, for_op.getLoc());
      offset = add(offset, block_ptr_op.getBaseOffset(), rewriter, for_op.getLoc());
      // FIXME: how to reset the operand elegantly?
      block_ptr_op->setOperand(1, offset);

      block_ptr_op.setBlockShape(new_block_shape);
      infer_type(block_ptr_op);
      block_ptrs.push_back(block_ptr_op.getResult());
    }

    // create scf::for
    auto body_builder = [&](OpBuilder &builder, Location loc, Value iv, ValueRange iters) {
      SmallVector<Value> iter_args(iters);
      SmallVector<Value> yield_args;

      IRMapping val_map;
      // map iter
      val_map.map(for_op.getIterArgInEntry(), iv);
      // map block
      assert(iter_args.size() == block_ptrs.size() + for_op.getNumInitArgs());
      for (size_t i = 0; i < block_ptrs.size(); ++i) {
        auto block_ptr = iter_args[i];
        auto block_load_op = builder.create<triton::BlockLoadOp>(loc, block_ptr);
        val_map.map(for_op.getBlockArgInEntry(i), block_load_op.getResult());
      }
      // map init
      for (size_t i = block_ptrs.size(); i < iter_args.size(); ++i) {
        val_map.map(for_op.getInitArgInEntry(i - block_ptrs.size()), iter_args[i]);
      }

      // clone all other ops (no need for infer type due to no change)
      for (auto &op : for_op.getRegion().front()) {
        if (auto yield_op = dyn_cast<BlockYieldOp>(op)) {
          assert(yield_op.getNumBlocks() == 0);
          // build block advance
          SmallVector<Value> yield_args;
          for (size_t i = 0; i < block_ptrs.size(); ++i) {
            auto block_ptr = iter_args[i];
            auto rank = cast<triton::BlockPointerType>(block_ptr.getType()).getPointeeType().getRank();
            SmallVector<int64_t> offsets(rank, 0);
            offsets[block_dims[i]] = step;
            auto advance_op = builder.create<triton::BlockAdvanceOp>(loc, block_ptr, offsets);
            yield_args.push_back(advance_op.getResult());
          }

          assert(yield_args.size() + yield_op.getNumIters() == iter_args.size());
          for (int i = 0; i < yield_op.getNumIters(); ++i) {
            yield_args.push_back(val_map.lookup(yield_op.getIter(i)));
          }
          rewriter.create<scf::YieldOp>(loc, yield_args);
        } else {
          auto new_op = builder.clone(op, val_map);
          for (auto [old_res, new_res] : llvm::zip(op.getResults(), new_op->getResults())) {
            val_map.map(old_res, new_res);
          }
        }
      }
    };

    SmallVector<Value> scf_for_iters(block_ptrs);
    for (int i = 0; i < for_op.getNumInitArgs(); ++i) {
      scf_for_iters.push_back(for_op.getInitArg(i));
    }
    auto yield_op = cast<BlockYieldOp>(for_op.getBody()->getTerminator());
    // TODO: if num_block > 0, we need tl.store in for loop
    if (yield_op.getNumBlocks() > 0) {
      return failure();
    }
    assert(yield_op.getNumBlocks() == 0);

    rewriter.setInsertionPoint(for_op);
    auto scf_for = rewriter.create<scf::ForOp>(for_op.getLoc(), lb_v, ub_v, step_v, scf_for_iters, body_builder);

    // replace uses
    assert(for_op->getNumResults() == scf_for->getNumResults() - block_ptrs.size());
    for (auto [old_res, new_res] :
         llvm::zip(for_op->getResults(), llvm::drop_begin(scf_for->getResults(), block_ptrs.size()))) {
      rewriter.replaceAllUsesWith(old_res, new_res);
    }
    rewriter.eraseOp(for_op);
    for (auto block_load_op : block_load_ops) {
      rewriter.eraseOp(block_load_op);
    }

    return success();
  }
};

// FIXME: redundant code
struct BlockForConversion : public OpConversionPattern<BlockForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockForOp for_op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // match
    // need to be executed after ParallelForConversion
    auto para_for_op = for_op->getParentOfType<ParallelForOp>();
    if (para_for_op != nullptr)
      return failure();

    // all block arg must be result of block load
    SmallVector<triton::BlockLoadOp> block_load_ops;
    SmallVector<triton::BlockPointerOfOp> block_ptr_ops;
    SmallVector<int> block_dims;
    for (int i = 0; i < for_op.getNumBlockArgs(); ++i) {
      auto arg = for_op.getBlockArg(i);
      auto block_load_op = arg.getDefiningOp<triton::BlockLoadOp>();
      if (block_load_op == nullptr)
        return failure();
      if (block_load_op.getResult().hasOneUse() == false)
        return failure();
      auto block_ptr_op = block_load_op.getSrcPointer().getDefiningOp<triton::BlockPointerOfOp>();
      if (block_ptr_op == nullptr)
        return failure();
      if (block_ptr_op.getResult().hasOneUse() == false)
        return failure();

      block_load_ops.push_back(block_load_op);
      block_ptr_ops.push_back(block_ptr_op);

      auto dim = for_op.getBlockDim(i);
      block_dims.push_back(dim);
    }

    // matched
    auto lb = for_op.getLowerBoundAttr().getInt();
    auto ub = for_op.getUpperBoundAttr().getInt();
    int step = for_op.getStepAttr().getInt();
    auto step_v = getConst(step, rewriter, for_op.getLoc());

    // update block_ptr_ops
    SmallVector<Value> block_ptrs;
    for (auto [block_ptr_op, block_load_op, block_dim] : llvm::zip(block_ptr_ops, block_load_ops, block_dims)) {
      auto block_shape = block_ptr_op.getBlockShape();
      SmallVector<int64_t> new_block_shape(block_shape);
      assert(block_shape[block_dim] > step);
      assert(block_shape[block_dim] % step == 0);
      new_block_shape[block_dim] = step;
      block_ptr_op.setBlockShape(new_block_shape);
      infer_type(block_ptr_op);
      block_ptrs.push_back(block_ptr_op.getResult());
    }

    // create scf::for
    auto body_builder = [&](OpBuilder &builder, Location loc, Value iv, ValueRange iters) {
      SmallVector<Value> iter_args(iters);
      SmallVector<Value> yield_args;

      IRMapping val_map;
      // map iter
      val_map.map(for_op.getIterArgInEntry(), iv);
      // map block
      assert(iter_args.size() == block_ptrs.size() + for_op.getNumInitArgs());
      for (size_t i = 0; i < block_ptrs.size(); ++i) {
        auto block_ptr = iter_args[i];
        auto block_load_op = builder.create<triton::BlockLoadOp>(loc, block_ptr);
        val_map.map(for_op.getBlockArgInEntry(i), block_load_op.getResult());
      }
      // map init
      for (size_t i = block_ptrs.size(); i < iter_args.size(); ++i) {
        val_map.map(for_op.getInitArgInEntry(i - block_ptrs.size()), iter_args[i]);
      }

      // clone all other ops (no need for infer type due to no change)
      for (auto &op : for_op.getRegion().front()) {
        if (auto yield_op = dyn_cast<BlockYieldOp>(op)) {
          assert(yield_op.getNumBlocks() == 0);
          // build block advance
          SmallVector<Value> yield_args;
          for (size_t i = 0; i < block_ptrs.size(); ++i) {
            auto block_ptr = iter_args[i];
            auto rank = cast<triton::BlockPointerType>(block_ptr.getType()).getPointeeType().getRank();
            SmallVector<int64_t> offsets(rank, 0);
            offsets[block_dims[i]] = step;
            auto advance_op = builder.create<triton::BlockAdvanceOp>(loc, block_ptr, offsets);
            yield_args.push_back(advance_op.getResult());
          }

          assert(yield_args.size() + yield_op.getNumIters() == iter_args.size());
          for (int i = 0; i < yield_op.getNumIters(); ++i) {
            yield_args.push_back(val_map.lookup(yield_op.getIter(i)));
          }
          rewriter.create<scf::YieldOp>(loc, yield_args);
        } else {
          auto new_op = builder.clone(op, val_map);
          for (auto [old_res, new_res] : llvm::zip(op.getResults(), new_op->getResults())) {
            val_map.map(old_res, new_res);
          }
        }
      }
    };

    SmallVector<Value> scf_for_iters(block_ptrs);
    for (int i = 0; i < for_op.getNumInitArgs(); ++i) {
      scf_for_iters.push_back(for_op.getInitArg(i));
    }
    auto yield_op = cast<BlockYieldOp>(for_op.getBody()->getTerminator());
    // TODO: if num_block > 0, we need tl.store in for loop
    assert(yield_op.getNumBlocks() == 0);

    rewriter.setInsertionPoint(for_op);
    auto lb_v = getConst(lb, rewriter, for_op.getLoc());
    auto ub_v = getConst(ub, rewriter, for_op.getLoc());
    auto scf_for = rewriter.create<scf::ForOp>(for_op.getLoc(), lb_v, ub_v, step_v, scf_for_iters, body_builder);

    // replace uses
    assert(for_op->getNumResults() == scf_for->getNumResults() - block_ptrs.size());
    for (auto [old_res, new_res] :
         llvm::zip(for_op->getResults(), llvm::drop_begin(scf_for->getResults(), block_ptrs.size()))) {
      rewriter.replaceAllUsesWith(old_res, new_res);
    }
    rewriter.eraseOp(for_op);
    for (auto block_load_op : block_load_ops) {
      rewriter.eraseOp(block_load_op);
    }

    return success();
  }
};

class ConvertAsukaToAsukaTriton : public ::mlir::asuka::impl::ConvertAsukaToAsukaTritonBase<ConvertAsukaToAsukaTriton> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    KernelOp k = getOperation();
    ConversionTarget target(*context);

    // clang-format off
    target.addLegalDialect<::mlir::asuka::AsukaDialect,
                           ::mlir::asuka::triton::AsukaTritonDialect,
                           ::mlir::arith::ArithDialect,
                           ::mlir::scf::SCFDialect>();
    // clang-format on

    // some patterns need ir structure for matching, so we need to run them in different pass theorectically
    // TODO: we use different applyPartialConversion to run them in one pass, is it undefined behavior?
    target.addIllegalOp<ParallelForOp>();
    RewritePatternSet patterns0(context);
    patterns0.add<ParallelForConversion>(context);
    if (failed(applyPartialConversion(k, target, std::move(patterns0))))
      return signalPassFailure();

    target.addDynamicallyLegalOp<PermuteOp>([](PermuteOp permute_op) {
      auto operand = permute_op.getOperand();
      auto block_load_op = operand.getDefiningOp<triton::BlockLoadOp>();
      if (block_load_op == nullptr)
        return true;
      if (block_load_op.getResult().hasOneUse() == false)
        return true;
      auto block_ptr_op = block_load_op.getSrcPointer().getDefiningOp<triton::BlockPointerOfOp>();
      if (block_ptr_op == nullptr)
        return true;
      if (block_ptr_op.getResult().hasOneUse() == false)
        return true;
      return false;
    });
    RewritePatternSet patterns1(context);
    patterns1.add<FusePermuteConversion>(context);
    if (failed(applyPartialConversion(k, target, std::move(patterns1))))
      return signalPassFailure();

    target.addIllegalOp<DynamicBlockForOp, BlockForOp>();
    RewritePatternSet patterns2(context);
    patterns2.add<DynamicBlockForConversion>(context);
    patterns2.add<BlockForConversion>(context);
    if (failed(applyPartialConversion(k, target, std::move(patterns2))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertAsukaToAsukaTritonPass() {
  return std::make_unique<ConvertAsukaToAsukaTriton>();
}

} // namespace mlir::asuka