#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"
#include "asuka/Dialect/Asuka/Transforms/Passes.h"

#include "asuka/Analysis/Parallelism.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

#include "dbg.h"

namespace mlir::asuka {

#define GEN_PASS_DEF_ASUKAANNOTATEPARALLELISM
#include "asuka/Dialect/Asuka/Transforms/Passes.h.inc"

} // namespace mlir::asuka

namespace mlir::asuka {

namespace {

class AnnotateParallelismPass : public ::mlir::asuka::impl::AsukaAnnotateParallelismBase<AnnotateParallelismPass> {
public:
  using AsukaAnnotateParallelismBase::AsukaAnnotateParallelismBase;
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
      if (verbose) {
        map.show();
      }
    }

    kernel_op.setParallelMaps(parallel_maps);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    KernelOp kernel_op = getOperation();
    // llvm::errs() << "kernel: " << kernel_op.getName() << "\n";
    if (kernel_op.hasParallelMap()) {
      signalPassFailure();
      return;
    }
    ParallelismAnalysis ana;
    ana.initialize(kernel_op);
    ana.run(kernel_op, verbose);

    if (verbose) {
      for (auto arg : kernel_op.getArguments()) {
        arg.dump();
        std::string str;
        llvm::raw_string_ostream os(str);
        ana.getInfo(arg).print(os, ana.batch_set);
        llvm::errs() << "\t" << str << "\n";
      }
      kernel_op.getCallableRegion()->front().walk([&](Operation *op) {
        op->dump();
        for (auto res : op->getResults()) {
          llvm::errs() << "\t";
          std::string str;
          llvm::raw_string_ostream os(str);
          ana.getInfo(res).print(os, ana.batch_set);
          llvm::errs() << "\t" << str << "\n";
        }
      });
    }

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
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAnnotateParallelismPass() { return std::make_unique<AnnotateParallelismPass>(); }
std::unique_ptr<mlir::Pass> createAnnotateParallelismPass(const AsukaAnnotateParallelismOptions &options) {
  return std::make_unique<AnnotateParallelismPass>(options);
}

} // namespace mlir::asuka