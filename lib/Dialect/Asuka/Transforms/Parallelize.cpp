#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "asuka/Analysis/Parallelism.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKAPARALLELIZE
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

class ParallelizePass : public ::mlir::asuka::impl::AsukaParallelizeBase<ParallelizePass> {
public:
  using AsukaParallelizeBase::AsukaParallelizeBase;

  void set_parallel_maps(KernelOp kernel_op, DenseSet<int> &allocated_batch_ids, ParallelismAnalysis &ana,
                         SmallVector<KernelOp::ParallelMap> &parallel_maps, DenseSet<int> &squeezed_batch_ids,
                         DenseMap<int, int> &batch_id_to_map_id) const {
    for (auto father : allocated_batch_ids) {
      KernelOp::ParallelMap map;
      for (auto arg : kernel_op.getArguments()) {
        auto info = ana.getInfo(arg);
        auto shape = cast<RankedTensorType>(arg.getType()).getShape();
        assert(shape.size() == info.getRank());
        int arg_dim = -1;
        for (int i = 0; i < info.getRank(); ++i) {
          if (info.info[i].kind == ParaType::Kind::kBatch && ana.batch_set.find(info.info[i].batch_id) == father) {
            if (map.unit_num <= 0) {
              map.unit_num = shape[i];
              map.size_per_unit = 1;
            } else {
              assert(map.unit_num == shape[i]);
              assert(map.size_per_unit == 1);
            }
            assert(arg_dim == -1);
            arg_dim = i;
          }
          if (info.info[i].kind == ParaType::Kind::kReUse && ana.batch_set.find(info.info[i].batch_id) == father) {
            if (map.unit_num <= 0) {
              // heuristic
              if (shape[i] > 128) {
                int size_per_unit = 128;
                assert(shape[i] % size_per_unit == 0);
                map.unit_num = shape[i] / size_per_unit;
                map.size_per_unit = size_per_unit;
              } else {
                // too small size
                // only batch
                map.size_per_unit = 1;
                map.unit_num = (int)shape[i];
              }
            } else {
              assert(shape[i] % map.size_per_unit == 0);
              assert(shape[i] / map.size_per_unit == map.unit_num);
            }
            assert(arg_dim == -1);
            arg_dim = i;
          }
        }
        map.arg_dims.push_back(arg_dim);
      }

      auto ret_op = cast<ReturnOp>(kernel_op.getCallableRegion()->front().getTerminator());
      for (auto res : ret_op.getOperands()) {
        auto info = ana.getInfo(res);
        auto shape = cast<RankedTensorType>(res.getType()).getShape();
        assert(shape.size() == info.getRank());
        int res_dim = -1;
        for (int i = 0; i < info.getRank(); ++i) {
          if (info.info[i].kind == ParaType::Kind::kBatch && ana.batch_set.find(info.info[i].batch_id) == father) {
            assert(map.unit_num = shape[i]);
            assert(map.size_per_unit = 1);
            assert(res_dim == -1);
            res_dim = i;
          }
          if (info.info[i].kind == ParaType::Kind::kReUse && ana.batch_set.find(info.info[i].batch_id) == father) {
            assert(map.unit_num = shape[i] / map.size_per_unit);
            assert(map.size_per_unit == shape[i] / map.unit_num);
            assert(res_dim == -1);
            res_dim = i;
          }
        }
        map.res_dims.push_back(res_dim);
      }

      parallel_maps.push_back(map);
      if (map.size_per_unit == 1) {
        squeezed_batch_ids.insert(father);
      }
      batch_id_to_map_id[father] = (int)parallel_maps.size() - 1;
      // map.show();
    }

    kernel_op.setParallelMaps(parallel_maps);
  }

  LogicalResult matchAndRewrite(KernelOp kernel_op, OpBuilder &builder) const {
    if (kernel_op.hasParallelMap()) {
      return failure();
    }
    ParallelismAnalysis ana;
    ana.initialize(kernel_op);
    ana.run(kernel_op);

    // FIXME: verbose flag?
    /*
    for (auto arg : kernel_op.getArguments()) {
      arg.dump();
      std::string str;
      llvm::raw_string_ostream os(str);
      ana.getInfo(arg).print(os, ana.batch_set);
      llvm::errs() << "\t" << str << "\n";
    }
    kernel_op.getCallableRegion()->front().walk<WalkOrder::PreOrder>([&](Operation *op) {
      op->dump();
      for (auto res : op->getResults()) {
        llvm::errs() << "\t";
        std::string str;
        llvm::raw_string_ostream os(str);
        ana.getInfo(res).print(os, ana.batch_set);
        llvm::errs() << "\t" << str << "\n";
      }
      if (isa<MaskOp>(op))
        return WalkResult::skip();
      else
        return WalkResult::advance();
    });
    */

    DenseSet<int> allocated_batch_ids;

    for (auto arg : kernel_op.getArguments()) {
      auto info = ana.getInfo(arg);
      for (size_t i = 0; i < info.getRank(); ++i) {
        auto &type = info.info[i];
        if (type.kind == ParaType::Kind::kBatch) {
          if (!allocated_batch_ids.contains(ana.batch_set.find(type.batch_id))) {
            allocated_batch_ids.insert(ana.batch_set.find(type.batch_id));
          }
        } else if (type.kind == ParaType::Kind::kReUse) {
          auto shape = cast<RankedTensorType>(arg.getType()).getShape();
          // heuristic
          // for locality, inner dim should not be parallelized
          bool is_innermost_dim = i == (int)(info.getRank()) - 1;
          if (!is_innermost_dim && !allocated_batch_ids.contains(ana.batch_set.find(type.batch_id))) {
            allocated_batch_ids.insert(ana.batch_set.find(type.batch_id));
          }
        }
      }
    }

    SmallVector<KernelOp::ParallelMap> parallel_maps;
    // size_per_unit = 1 && is_batch/reuse
    // we need to squeeze it
    DenseSet<int> squeezed_batch_ids;
    DenseMap<int, int> batch_id_to_map_id;
    set_parallel_maps(kernel_op, allocated_batch_ids, ana, parallel_maps, squeezed_batch_ids, batch_id_to_map_id);

    // FIXME: if not partition now, we can never partition it. maybe we need another pass
    if (!partition) {
      return success();
    }

    // build
    Block *build_block = builder.createBlock(kernel_op.getCallableRegion());
    builder.setInsertionPointToStart(build_block);
    auto arg_types = kernel_op.getArgumentTypes();
    auto res_types = kernel_op.getResultTypes();
    for (auto arg_type : arg_types) {
      build_block->addArgument(arg_type, kernel_op.getLoc());
    }
    auto para_for_op = builder.create<ParallelForOp>(kernel_op.getLoc(), res_types, build_block->getArguments());
    Block *entry = builder.createBlock(&para_for_op.getRegion());

    // prepare block argument types
    SmallVector<Type> para_arg_types(arg_types);
    SmallVector<Type> para_res_types(res_types);
    for (auto &map : parallel_maps) {
      for (size_t i = 0; i < map.arg_dims.size(); ++i) {
        auto type = cast<RankedTensorType>(para_arg_types[i]);
        SmallVector<int64_t> shape(type.getShape());
        if (map.arg_dims[i] < 0)
          continue;
        shape[map.arg_dims[i]] = map.size_per_unit;
        para_arg_types[i] = RankedTensorType::get(shape, type.getElementType());
      }
      for (size_t i = 0; i < map.res_dims.size(); ++i) {
        auto type = cast<RankedTensorType>(para_res_types[i]);
        SmallVector<int64_t> shape(type.getShape());
        if (map.res_dims[i] < 0)
          continue;
        shape[map.res_dims[i]] = map.size_per_unit;
        para_res_types[i] = RankedTensorType::get(shape, type.getElementType());
      }
    }
    // squeeze dim
    for (int arg_i = 0; arg_i < kernel_op.getNumArguments(); ++arg_i) {
      auto arg = kernel_op.getArgument(arg_i);
      auto info = ana.getInfo(arg);
      auto type = cast<RankedTensorType>(arg.getType());
      auto para_shape = cast<RankedTensorType>(para_arg_types[arg_i]).getShape();
      SmallVector<int64_t> new_shape;
      for (int i = 0; i < (int)para_shape.size(); ++i) {
        auto &para_type = info.info[i];
        if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
            squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
          assert(para_shape[i] == 1);
        } else {
          new_shape.push_back(para_shape[i]);
        }
      }
      // reset shape
      para_arg_types[arg_i] = RankedTensorType::get(new_shape, type.getElementType());
    }
    for (int res_i = 0; res_i < kernel_op.getNumResults(); ++res_i) {
      auto res = kernel_op.getCallableRegion()->front().getTerminator()->getOperand(res_i);
      auto info = ana.getInfo(res);
      auto type = cast<RankedTensorType>(res.getType());
      auto para_shape = cast<RankedTensorType>(para_res_types[res_i]).getShape();
      SmallVector<int64_t> new_shape;
      for (int i = 0; i < (int)para_shape.size(); ++i) {
        auto &para_type = info.info[i];
        if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
            squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
          assert(para_shape[i] == 1);
        } else {
          new_shape.push_back(para_shape[i]);
        }
      }
      // reset shape
      para_res_types[res_i] = RankedTensorType::get(new_shape, type.getElementType());
    }

    // add parallel id and other arguments
    SmallVector<Value> para_id_args;
    SmallVector<Value> para_args;
    for (int i = 0; i < (int)parallel_maps.size(); ++i) {
      auto index_type = builder.getIndexType();
      auto id_arg = entry->addArgument(index_type, kernel_op.getLoc());
      para_id_args.push_back(id_arg);
    }
    for (auto arg_type : para_arg_types) {
      auto para_arg = entry->addArgument(arg_type, kernel_op.getLoc());
      para_args.push_back(para_arg);
    }

    // clone op in original block
    IRMapping val_map;
    for (auto [original_arg, arg] : llvm::zip(kernel_op.getCallableRegion()->front().getArguments(), para_args)) {
      val_map.map(original_arg, arg);
    }
    Block *original_block = &kernel_op.getCallableRegion()->front();
    original_block->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto const_op = dyn_cast<arith::ConstantOp>(op)) {
        auto res = const_op.getResult();
        bool parallelized = false;
        if (isa<RankedTensorType>(res.getType())) {
          auto info = ana.getInfo(res);
          for (int i = 0; i < (int)info.getRank(); ++i) {
            auto para_type = info.info[i];
            if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
                squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
              parallelized = true;
            }
          }
        }
        if (!parallelized) {
          auto new_op = builder.clone(*op);
          for (auto [res, mapped_res] : llvm::zip(op->getResults(), new_op->getResults())) {
            val_map.map(res, mapped_res);
          }
        } else {
          llvm_unreachable("not support");
        }
      } else if (auto mask_op = dyn_cast<MaskOp>(op)) {
        auto new_op = builder.clone(*op, val_map);
        auto new_mask_op = cast<MaskOp>(new_op);

        auto res = mask_op.getResult();
        SmallVector<int64_t> sizes(mask_op.getSizes());
        auto info = ana.getInfo(res);
        // TODO: need program_id
        for (int i = 0; i < (int)info.getRank(); ++i) {
          auto para_type = info.info[i];
          auto batch_id = para_type.batch_id <= 0 ? -1 : ana.batch_set.find(para_type.batch_id);
          if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
              allocated_batch_ids.contains(batch_id)) {
            // need partition
            int map_id = batch_id_to_map_id[batch_id];
            new_mask_op->setOperand(i, para_id_args[map_id]);
            sizes[i] = parallel_maps[map_id].size_per_unit;
          }
        }
        new_mask_op.setSizes(sizes);
        auto type_infer = cast<InferTypeOpInterface>(new_op);
        llvm::SmallVector<::mlir::Type, 1> new_types;
        auto success = type_infer.inferReturnTypes(new_op->getContext(), new_op->getLoc(), new_op->getOperands(),
                                                   new_op->getAttrDictionary(), new_op->getPropertiesStorage(),
                                                   new_op->getRegions(), new_types);
        assert(succeeded(success));
        for (size_t i = 0; i < new_types.size(); ++i) {
          new_op->getResult(i).setType(new_types[i]);
        }
        for (auto [res, mapped_res] : llvm::zip(op->getResults(), new_op->getResults())) {
          val_map.map(res, mapped_res);
        }

        // do not walk into its inner region
        return WalkResult::skip();
      } else if (auto permute_op = dyn_cast<PermuteOp>(op)) {
        auto arg = permute_op.getOperand();
        auto permute_dims = permute_op.getDims();
        auto info = ana.getInfo(arg);
        DenseMap<int, int> dim_map;
        for (int i = 0, j = 0; i < (int)info.getRank(); ++i) {
          auto para_type = info.info[i];
          if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
              squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
            dim_map[i] = -1;
          } else {
            dim_map[i] = j++;
          }
        }

        SmallVector<int64_t> new_permute_dims;
        for (auto dim : permute_dims) {
          if (dim_map[dim] >= 0) {
            new_permute_dims.push_back(dim_map[dim]);
          }
        }

        auto new_op = builder.create<PermuteOp>(op->getLoc(), val_map.lookup(arg), new_permute_dims);
        val_map.map(permute_op.getResult(), new_op.getResult());
      } else if (auto reduce_op = dyn_cast<ReduceOp>(op)) {
        auto arg = reduce_op.getOperand();
        int64_t dim = reduce_op.getReduceDimensionAttr().getInt();
        if (dim < 0) {
          dim += cast<RankedTensorType>(reduce_op.getOperand().getType()).getRank();
        }
        auto reduce_type = reduce_op.getReduceType();
        bool keep_dim = reduce_op.getKeepDim();
        auto info = ana.getInfo(arg);
        DenseMap<int, int> dim_map;
        for (int i = 0, j = 0; i < (int)info.getRank(); ++i) {
          auto para_type = info.info[i];
          if ((para_type.kind == ParaType::Kind::kBatch || para_type.kind == ParaType::Kind::kReUse) &&
              squeezed_batch_ids.contains(ana.batch_set.find(para_type.batch_id))) {
            dim_map[i] = -1;
          } else {
            dim_map[i] = j++;
          }
        }
        // reduce_dim cannot be parallelized
        assert(dim_map[dim] >= 0);

        auto new_op = builder.create<ReduceOp>(op->getLoc(), val_map.lookup(arg), dim_map[dim], reduce_type, keep_dim);
        val_map.map(reduce_op.getResult(), new_op.getResult());
      } else if (isa<InferTypeOpInterface>(op) && !isa<TriluOp>(op)) {
        auto new_op = builder.clone(*op);
        for (size_t i = 0; i < new_op->getNumOperands(); ++i) {
          new_op->setOperand(i, val_map.lookup(new_op->getOperand(i)));
        }
        auto type_infer = cast<InferTypeOpInterface>(new_op);
        llvm::SmallVector<::mlir::Type, 1> new_types;
        auto success = type_infer.inferReturnTypes(new_op->getContext(), new_op->getLoc(), new_op->getOperands(),
                                                   new_op->getAttrDictionary(), new_op->getPropertiesStorage(),
                                                   new_op->getRegions(), new_types);
        assert(succeeded(success));
        for (size_t i = 0; i < new_types.size(); ++i) {
          new_op->getResult(i).setType(new_types[i]);
        }
        for (auto [res, mapped_res] : llvm::zip(op->getResults(), new_op->getResults())) {
          val_map.map(res, mapped_res);
        }
      } else if (auto return_op = dyn_cast<ReturnOp>(op)) {
        // check type
        SmallVector<Value> yield_args;
        for (size_t i = 0; i < return_op.getNumOperands(); ++i) {
          auto res = return_op.getOperand(i);
          auto new_res = val_map.lookup(res);
          auto type = cast<RankedTensorType>(new_res.getType());
          auto para_type = cast<RankedTensorType>(para_res_types[i]);
          assert(type.getRank() == para_type.getRank());
          for (int j = 0; j < (int)type.getRank(); ++j) {
            // assert(type.getShape()[j] == para_type.getShape()[j]);
            if (type.getShape()[j] != para_type.getShape()[j]) {
              kernel_op->dump();
              llvm_unreachable("result shape not verified");
            }
          }
          yield_args.push_back(new_res);
        }
        builder.create<ParallelYieldOp>(op->getLoc(), yield_args);

        // write last return
        builder.setInsertionPointAfter(para_for_op);
        builder.create<ReturnOp>(op->getLoc(), para_for_op.getResults());
      } else {
        op->dump();
        llvm_unreachable("not supported");
      }
      return WalkResult::advance();
    });

    // erase original block
    // FIXME: any api we can use to erase block?
    assert(original_block->use_empty());
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(*original_block))) {
      assert(op.use_empty());
      op.erase();
    }
    original_block->erase();

    // kernel_op->dump();
    return success();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    KernelOp kernel_op = getOperation();
    // dbg(partition);
    OpBuilder builder(context);
    if (failed(matchAndRewrite(kernel_op, builder))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createParallelizePass() { return std::make_unique<ParallelizePass>(); }
std::unique_ptr<mlir::Pass> createParallelizePass(const AsukaParallelizeOptions &options) {
  return std::make_unique<ParallelizePass>(options);
}

} // namespace mlir::asuka