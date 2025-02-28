#include "asuka/Analysis/Parallelism.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"

#include "dbg.h"

namespace mlir {

ParaInfo ParaInfo::from_val(Value val) {
  auto rank = cast<RankedTensorType>(val.getType()).getRank();
  return ParaInfo(rank);
}

void ParallelismAnalysis::initialize(asuka::KernelOp kernel_op) {
  this->clear();
  for (auto arg : kernel_op.getArguments()) {
    val_info[arg] = ParaInfo::from_val(arg);
  }

  kernel_op.getCallableRegion()->front().walk([&](Operation *op) {
    for (auto res : op->getResults()) {
      if (isa<RankedTensorType>(res.getType())) {
        val_info[res] = ParaInfo::from_val(res);
      }
    }
  });
}

void ParallelismAnalysis::run(asuka::KernelOp kernel_op, bool verbose) {
  auto block = &kernel_op.getCallableRegion()->front();

  bool changed = true;
  int iter = 0;
  while (changed) {
    if (iter > 1000) {
      dbg(iter);
      llvm_unreachable("too many iterations");
    }
    changed = false;
    iter += 1;
    // preorder to avoid walking into mask_op
    block->walk<WalkOrder::PreOrder>([&](Operation *op) {
      // llvm::errs() << "op: " << op->getName() << "\n";
      if (auto convert_op = dyn_cast<asuka::ConvertOp>(op)) {
        auto arg = convert_op.getOperand();
        auto res = convert_op.getResult();

        auto arg_old_info = val_info[arg];
        auto res_old_info = val_info[res];

        auto arg_new_info = ParaInfo::join(arg_old_info, res_old_info, batch_set);
        auto res_new_info = ParaInfo::join(res_old_info, arg_old_info, batch_set);

        if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) {
          changed = true;
          val_info[arg] = arg_new_info;
        }
        if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) {
          changed = true;
          val_info[res] = res_new_info;
        }
      } else if (auto permute_op = dyn_cast<asuka::PermuteOp>(op)) {
        auto arg = permute_op.getOperand();
        auto res = permute_op.getResult();
        auto dims = permute_op.getDims();

        auto arg_old_info = val_info[arg];
        auto res_old_info = val_info[res];

        auto arg_new_info = ParaInfo::join(arg_old_info, res_old_info.permute_from(dims), batch_set);
        auto res_new_info = ParaInfo::join(res_old_info, arg_old_info.permute_by(dims), batch_set);

        if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) {
          changed = true;
          val_info[arg] = arg_new_info;
        }

        if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) {
          changed = true;
          val_info[res] = res_new_info;
        }
      } else if (auto bin_op = dyn_cast<asuka::BroadcastableBinaryOpInterface>(op)) {
        auto lhs = bin_op.getLhs();
        auto rhs = bin_op.getRhs();
        auto res = bin_op.getResult();

        auto lhs_broadcasted_shape = bin_op.getLhsBroadcastedShape();
        auto rhs_broadcasted_shape = bin_op.getRhsBroadcastedShape();
        auto res_shape = bin_op.getExpectedResultShape();

        ParaInfo res_new_info(res_shape.size());
        ParaInfo lhs_new_info(lhs_broadcasted_shape.size());
        ParaInfo rhs_new_info(rhs_broadcasted_shape.size());

        auto lhs_old_info = val_info[lhs];
        auto rhs_old_info = val_info[rhs];
        auto res_old_info = val_info[res];
        for (int i = 0; i < (int)res_shape.size(); ++i) {
          auto lhs_d = lhs_broadcasted_shape[i];
          auto rhs_d = rhs_broadcasted_shape[i];
          auto batch_id = batch_set.alloc_batch();

          if (lhs_d == 1 && rhs_d > 1) {
            lhs_new_info.info[i] = ParaType(ParaType::kNonPara);
            rhs_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);
            res_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);
          } else if (lhs_d > 1 && rhs_d == 1) {
            lhs_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);
            rhs_new_info.info[i] = ParaType(ParaType::kNonPara);
            res_new_info.info[i] = ParaType(ParaType::kReUse, batch_id);
          } else {
            assert(lhs_d == rhs_d);
            lhs_new_info.info[i] = ParaType(ParaType::kBatch, batch_id);
            rhs_new_info.info[i] = ParaType(ParaType::kBatch, batch_id);
            res_new_info.info[i] = ParaType(ParaType::kBatch, batch_id);
          }
        }
        lhs_new_info.join_(lhs_old_info, batch_set);
        rhs_new_info.join_(rhs_old_info, batch_set);
        res_new_info.join_(res_old_info, batch_set);
        // aggregate information forward and backward
        // forward and backward
        for (int i = 0; i < (int)res_shape.size(); ++i) {
          auto lhs_d = lhs_broadcasted_shape[i];
          auto rhs_d = rhs_broadcasted_shape[i];
          // forward
          if (lhs_d == 1 && rhs_d > 1) {
            res_new_info.info[i].join_(rhs_new_info.info[i], batch_set);
          } else if (lhs_d > 1 && rhs_d == 1) {
            res_new_info.info[i].join_(lhs_new_info.info[i], batch_set);
          } else {
            res_new_info.info[i].join_(lhs_new_info.info[i], batch_set);
            res_new_info.info[i].join_(rhs_new_info.info[i], batch_set);
          }
          // backward
          lhs_new_info.info[i].join_(res_new_info.info[i], batch_set);
          rhs_new_info.info[i].join_(res_new_info.info[i], batch_set);
        }

        lhs_new_info = lhs_new_info.slice_like(lhs_old_info);
        rhs_new_info = rhs_new_info.slice_like(rhs_old_info);

        if (!ParaInfo::equal(lhs_new_info, lhs_old_info, batch_set)) {
          changed = true;
          val_info[lhs] = lhs_new_info;
        }
        if (!ParaInfo::equal(rhs_new_info, rhs_old_info, batch_set)) {
          changed = true;
          val_info[rhs] = rhs_new_info;
        }
        if (!ParaInfo::equal(res_new_info, res_old_info, batch_set)) {
          changed = true;
          val_info[res] = res_new_info;
        }
      } else if (auto dot_op = dyn_cast<asuka::DotOp>(op)) {
        auto lhs = dot_op.getLhs();
        auto rhs = dot_op.getRhs();
        auto res = dot_op.getResult();

        auto lhs_old_info = val_info[lhs];
        auto rhs_old_info = val_info[rhs];
        auto res_old_info = val_info[res];

        size_t rank = res_old_info.getRank();

        ParaInfo res_new_info(rank);
        ParaInfo lhs_new_info(rank);
        ParaInfo rhs_new_info(rank);
        assert(rank >= 2);
        // alloc batch
        for (int i = 0; i < (int)rank - 2; ++i) {
          int batch_id = batch_set.alloc_batch();
          lhs_new_info.set(i, ParaType(ParaType::kBatch, batch_id));
          rhs_new_info.set(i, ParaType(ParaType::kBatch, batch_id));
          res_new_info.set(i, ParaType(ParaType::kBatch, batch_id));
        }
        int row_batch_id = batch_set.alloc_batch();
        int col_batch_id = batch_set.alloc_batch();
        lhs_new_info.set(-2, ParaType(ParaType::kReUse, row_batch_id));
        res_new_info.set(-2, ParaType(ParaType::kReUse, row_batch_id));
        rhs_new_info.set(-1, ParaType(ParaType::kReUse, col_batch_id));
        res_new_info.set(-1, ParaType(ParaType::kReUse, col_batch_id));
        lhs_new_info.set(-1, ParaType(ParaType::kNonPara));
        rhs_new_info.set(-2, ParaType(ParaType::kNonPara));

        lhs_new_info.join_(lhs_old_info, batch_set);
        rhs_new_info.join_(rhs_old_info, batch_set);
        res_new_info.join_(res_old_info, batch_set);

        // aggregate information forward and backward
        // forward
        for (int i = 0; i < (int)rank - 2; ++i) {
          res_new_info.info[i].join_(lhs_new_info.info[i], batch_set);
          res_new_info.info[i].join_(lhs_new_info.info[i], batch_set);
        }
        res_new_info.info[rank - 2].join_(lhs_new_info.info[rank - 2], batch_set);
        res_new_info.info[rank - 1].join_(rhs_new_info.info[rank - 1], batch_set);
        // backward
        for (int i = 0; i < (int)rank - 2; ++i) {
          lhs_new_info.info[i].join_(res_new_info.info[i], batch_set);
          rhs_new_info.info[i].join_(res_new_info.info[i], batch_set);
        }
        lhs_new_info.info[rank - 2].join_(res_new_info.info[rank - 2], batch_set);
        rhs_new_info.info[rank - 1].join_(res_new_info.info[rank - 1], batch_set);

        // check
        if (!ParaInfo::equal(res_new_info, res_old_info, batch_set)) {
          changed = true;
          val_info[res] = res_new_info;
        }
        if (!ParaInfo::equal(lhs_new_info, lhs_old_info, batch_set)) {
          changed = true;
          val_info[lhs] = lhs_new_info;
        }
        if (!ParaInfo::equal(rhs_new_info, rhs_old_info, batch_set)) {
          changed = true;
          val_info[rhs] = rhs_new_info;
        }
      } else if (isa<asuka::Exp2Op, asuka::ExpOp, asuka::LogOp, asuka::NegOp, asuka::TanhOp>(op)) {
        assert(op->getNumOperands() == 1);
        assert(op->getNumResults() == 1);
        auto arg = op->getOperand(0);
        auto res = op->getResult(0);

        auto arg_old_info = val_info[arg];
        auto res_old_info = val_info[res];

        // aggregate information forward and backward
        // forward and backward
        auto new_info = ParaInfo::join(arg_old_info, res_old_info, batch_set);

        if (!ParaInfo::equal(arg_old_info, new_info, batch_set)) {
          changed = true;
          val_info[arg] = new_info;
        }
        if (!ParaInfo::equal(res_old_info, new_info, batch_set)) {
          changed = true;
          val_info[res] = new_info;
        }
      } else if (auto reduce_op = dyn_cast<asuka::ReduceOp>(op)) {
        if (reduce_op.getInit() == nullptr) {
          auto arg = reduce_op.getOperand();
          auto res = reduce_op.getResult();
          int64_t dim = reduce_op.getReduceDimensionAttr().getValue().getSExtValue();
          bool keep_dim = reduce_op.getKeepDim();
          if (dim < 0) {
            dim += (int64_t)cast<RankedTensorType>(arg.getType()).getRank();
          }

          auto arg_old_info = val_info[arg];
          auto res_old_info = val_info[res];

          ParaInfo arg_new_info(arg_old_info.getRank());
          ParaInfo res_new_info(res_old_info.getRank());
          arg_new_info.set(dim, ParaType(ParaType::kNonPara));
          if (keep_dim) {
            res_new_info.set(dim, ParaType(ParaType::kNonPara));
          }

          arg_new_info.join_(arg_old_info, batch_set);
          res_new_info.join_(res_old_info, batch_set);

          // aggregate information forward and backward
          // forward and backward
          for (int arg_i = 0, res_i = 0; arg_i < arg_new_info.getRank(); ++arg_i) {
            if (arg_i == dim && !keep_dim)
              continue;
            arg_new_info.info[arg_i].join_(res_new_info.info[res_i], batch_set);
            res_new_info.info[res_i].join_(arg_new_info.info[arg_i], batch_set);
            res_i += 1;
          }

          // check
          if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) {
            changed = true;
            val_info[arg] = arg_new_info;
          }
          if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) {
            changed = true;
            val_info[res] = res_new_info;
          }
        } else {
          llvm_unreachable("not supported");
        }
      } else if (auto reshape_op = dyn_cast<asuka::ReshapeOp>(op)) {
        auto arg = reshape_op.getOperand();
        auto res = reshape_op.getResult();
        auto arg_shape = cast<RankedTensorType>(arg.getType()).getShape();
        auto res_shape = cast<RankedTensorType>(res.getType()).getShape();

        auto arg_old_info = val_info[arg];
        auto res_old_info = val_info[res];

        ParaInfo arg_new_info(arg_old_info.getRank());
        ParaInfo res_new_info(res_old_info.getRank());

        if (arg_shape.size() == res_shape.size() + 1) {
          // [..., x, y, ...] -> [..., z,, ...]
          int arg_i = 0, res_i = 0;
          while (arg_shape[arg_i] == res_shape[res_i]) {
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }
          assert(arg_shape[arg_i] * arg_shape[arg_i + 1] == res_shape[res_i]);
          auto b0 = batch_set.alloc_batch();
          auto b1 = batch_set.alloc_batch();
          auto b2 = batch_set.alloc_batch();
          assert(b0 >= 0 && b1 >= 0 && b2 >= 0);
          arg_new_info.info[arg_i] = ParaType(ParaType::kBatch, b0);
          arg_new_info.info[arg_i + 1] = ParaType(ParaType::kBatch, b1);
          res_new_info.info[res_i] = ParaType(ParaType::kBatch, b2);

          arg_new_info.info[arg_i].join_(arg_old_info.info[arg_i], batch_set);
          arg_new_info.info[arg_i + 1].join_(arg_old_info.info[arg_i + 1], batch_set);
          res_new_info.info[res_i].join_(res_old_info.info[res_i], batch_set);

          if (res_old_info.info[res_i].kind == ParaType::kNonPara ||
              arg_old_info.info[arg_i].kind == ParaType::kNonPara ||
              arg_old_info.info[arg_i + 1].kind == ParaType::kNonPara) {
            arg_new_info.info[arg_i].kind = ParaType::kNonPara;
            arg_new_info.info[arg_i + 1].kind = ParaType::kNonPara;
            res_new_info.info[res_i].kind = ParaType::kNonPara;
          }
          arg_i += 2;
          res_i += 1;
          while (arg_i < (int)arg_shape.size()) {
            assert(arg_shape[arg_i] == res_shape[res_i]);
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }
        } else if (arg_shape.size() + 1 == res_shape.size()) {
          // [..., x, ...] -> [..., y, z, ...]
          int arg_i = 0, res_i = 0;
          while (arg_shape[arg_i] == res_shape[res_i]) {
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }
          assert(arg_shape[arg_i] == res_shape[res_i + 1] * res_shape[res_i]);
          auto b0 = batch_set.alloc_batch();
          auto b1 = batch_set.alloc_batch();
          auto b2 = batch_set.alloc_batch();
          assert(b0 >= 0 && b1 >= 0 && b2 >= 0);
          arg_new_info.info[arg_i] = ParaType(ParaType::kBatch, b0);
          res_new_info.info[res_i] = ParaType(ParaType::kBatch, b1);
          res_new_info.info[res_i + 1] = ParaType(ParaType::kBatch, b2);

          arg_new_info.info[arg_i].join_(arg_old_info.info[arg_i], batch_set);
          res_new_info.info[res_i].join_(res_old_info.info[res_i], batch_set);
          res_new_info.info[res_i + 1].join_(res_old_info.info[res_i + 1], batch_set);

          if (res_old_info.info[res_i].kind == ParaType::kNonPara ||
              arg_old_info.info[arg_i].kind == ParaType::kNonPara ||
              res_old_info.info[res_i + 1].kind == ParaType::kNonPara) {
            arg_new_info.info[arg_i].kind = ParaType::kNonPara;
            res_new_info.info[res_i].kind = ParaType::kNonPara;
            res_new_info.info[res_i + 1].kind = ParaType::kNonPara;
          }
          arg_i += 1;
          res_i += 2;
          while (arg_i < (int)arg_shape.size()) {
            assert(arg_shape[arg_i] == res_shape[res_i]);
            arg_new_info.info[arg_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            res_new_info.info[res_i] = ParaType::join(arg_old_info.info[arg_i], res_old_info.info[res_i], batch_set);
            arg_i++;
            res_i++;
          }

          // int arg_i = 0, res_i = 0;
          // while (arg_shape[arg_i] == res_shape[res_i]) {
          //   arg_new_info.info[arg_i].join_(res_old_info.info[res_i], batch_set);
          //   res_new_info.info[res_i].join_(arg_old_info.info[arg_i], batch_set);
          //   arg_i++;
          //   res_i++;
          // }
          // assert(arg_shape[arg_i] ==  res_shape[res_i] * res_shape[res_i + 1]);
          // if (res_new_info.info[res_i].kind == ParaType::kNonPara || res_new_info.info[res_i + 1].kind ==
          // ParaType::kNonPara || arg_new_info.info[arg_i].kind == ParaType::kNonPara) {
          //   arg_new_info.info[arg_i] = ParaType::kNonPara;
          //   res_new_info.info[res_i] = ParaType::kNonPara;
          //   res_new_info.info[res_i + 1] = ParaType::kNonPara;
          // } else {
          //   auto b0 = batch_set.alloc_batch();
          //   auto b1 = batch_set.alloc_batch();
          //   auto b2 = batch_set.alloc_batch();
          //   arg_new_info.info[arg_i].join_(ParaType(ParaType::kBatch, b0), batch_set);
          //   res_new_info.info[res_i].join_(ParaType(ParaType::kBatch, b2), batch_set);
          //   res_new_info.info[res_i + 1].join_(ParaType(ParaType::kBatch, b1), batch_set);
          // }
          // arg_i += 1;
          // res_i += 2;
          // while (arg_i < (int)arg_shape.size()) {
          //   assert(arg_shape[arg_i] == res_shape[res_i]);
          //   arg_new_info.info[arg_i].join_(res_old_info.info[res_i], batch_set);
          //   res_new_info.info[res_i].join_(arg_old_info.info[arg_i], batch_set);
          //   arg_i++;
          //   res_i++;
          // }
        } else {
          op->dump();
          llvm_unreachable("reshape not fully supported");
        }
        if (!ParaInfo::equal(arg_old_info, arg_new_info, batch_set)) {
          changed = true;
          val_info[arg] = arg_new_info;
        }
        if (!ParaInfo::equal(res_old_info, res_new_info, batch_set)) {
          changed = true;
          val_info[res] = res_new_info;
        }
      } else if (isa<asuka::MaskOp>(op)) {
        // pass
        return WalkResult::skip();
      } else if (op->getNumOperands() == 0 || isa<asuka::ReturnOp>(op)) {
        // pass
        return WalkResult::skip();
      } else {
        op->dump();
        llvm_unreachable("not supported");
      }
      return WalkResult::advance();
    });

    if (verbose) {
      dbg(iter);
      for (auto arg : kernel_op.getArguments()) {
        arg.dump();
        std::string str;
        llvm::raw_string_ostream os(str);
        val_info[arg].print(os, batch_set);
        llvm::errs() << "\t" << str << "\n";
      }
      kernel_op.getCallableRegion()->front().walk([&](Operation *op) {
        op->dump();
        for (auto res : op->getResults()) {
          llvm::errs() << "\t";
          std::string str;
          llvm::raw_string_ostream os(str);
          val_info[res].print(os, batch_set);
          llvm::errs() << "\t" << str << "\n";
        }
      });
    }
  }
  if (verbose) {
    dbg(iter);
  }
}

/*
void ParallelismAnalysis::visitOperation(
  Operation *op,
  ArrayRef<const dataflow::Lattice<ParallelismInfo> *> operands,
  ArrayRef<dataflow::Lattice<ParallelismInfo> *> results) {
  dbg("into visit Operation");
  llvm::errs() << op->getName() << "\n";
  llvm::errs() << "arg_num: " << operands.size() << "\n";
  for (auto* arg: operands) {
    auto v = arg->getValue();
    std::string arg_type;
    llvm::raw_string_ostream os(arg_type);
    v.print(os);
    llvm::errs() << "\t" << arg_type << "\n";
  }
  llvm::errs() << "res_num: " << results.size() << "\n";
  for (auto* res: results) {
    auto v = res->getValue();
    std::string res_type;
    llvm::raw_string_ostream os(res_type);
    v.print(os);
    llvm::errs() << "\t" << res_type << "\n";
  }

  if (auto permute_op = dyn_cast<asuka::PermuteOp>(op)) {
    assert(operands.size() == 1);
    assert(results.size() == 1);
    assert(operands[0]->getValue().isInitialized());

    auto dims = permute_op.getDims();
    ParallelismInfo new_para(operands[0]->getValue().getRank());
    for (auto en: llvm::enumerate(dims)) {
      new_para.setParallelismType(en.index(), operands[0]->getValue().getParallelismType(en.value()));
    }
    propagateIfChanged(results[0], results[0]->join(new_para));
  } else if (auto dot_op = dyn_cast<asuka::DotOp>(op)) {
    assert(operands.size() == 2);
    assert(results.size() == 1);
    auto res_rank = cast<RankedTensorType>(dot_op.getResult()).getRank();

    ParallelismInfo common(res_rank, NoReuse);
    common.setParallelismType()

    // new_para.setParallelismType()

  } else if (operands.size() == 0) {
    for (auto [res_lattice, res]: llvm::zip(results, op->getResults())) {
      if (!res_lattice->getValue().isInitialized()) {
        propagateIfChanged(res_lattice, res_lattice->join(ParallelismInfo::getPessimisticValueState(res)));
      }
    }
  }
}
*/

} // namespace mlir