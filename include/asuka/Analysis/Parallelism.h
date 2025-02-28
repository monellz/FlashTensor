#ifndef ASUKA_ANALYSIS_PARALLELISM_H
#define ASUKA_ANALYSIS_PARALLELISM_H

#include <variant>

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

#include "asuka/Dialect/Asuka/IR/AsukaDialect.h"

#include "dbg.h"

namespace mlir {

struct BatchSet {
  std::vector<int> father;
  int total_batch = 0;
  int find(int x) {
    assert(x < (int)father.size());
    assert(x >= 0);
    return father[x] == x ? x : father[x] = find(father[x]);
  }
  int alloc_batch() {
    int size = (int)father.size();
    father.push_back(size);
    total_batch += 1;
    return size;
  }
  void merge(int x, int y) {
    int fx = find(x);
    int fy = find(y);
    if (fx != fy) {
      father[fx] = fy;
      total_batch -= 1;
    }
  }
};

struct ParaType {
  enum Kind { kInit = 0, kBatch = 1, kReUse = 2, kNonPara = 3 };
  Kind kind;
  int batch_id; // for batch/reuse type, -1 for others

  ParaType() : kind(kInit), batch_id(-1) {}
  ParaType(Kind kind, int batch_id = -1) : kind(kind), batch_id(batch_id) {
    // assert((kind == kBatch || kind == kReUse) == (batch_id >= 0));
    if (kind == kBatch || kind == kReUse) {
      assert(batch_id >= 0);
    }
  }

  static ParaType join(const ParaType &lhs, const ParaType &rhs, BatchSet &batch_set) {
    ParaType ret(std::max(lhs.kind, rhs.kind), std::max(lhs.batch_id, rhs.batch_id));
    if (ret.kind == kBatch || ret.kind == kReUse) {
      ret.batch_id = std::max(lhs.batch_id, rhs.batch_id);
      if (lhs.batch_id >= 0 && rhs.batch_id >= 0) {
        batch_set.merge(lhs.batch_id, rhs.batch_id);
      }
    }
    return ret;
  }

  ParaType join(const ParaType &other, BatchSet &batch_set) {
    ParaType ret(std::max(kind, other.kind), std::max(batch_id, other.batch_id));
    ret.join_(other, batch_set);
    return ret;
  }

  void join_(const ParaType &other, BatchSet &batch_set) {
    // inplace join
    this->kind = std::max(kind, other.kind);
    if (kind == kBatch || kind == kReUse) {
      batch_id = std::max(batch_id, other.batch_id);
      if (batch_id >= 0 && other.batch_id >= 0) {
        batch_set.merge(batch_id, other.batch_id);
      }
    }
  }

  static bool equal(const ParaType &lhs, const ParaType &rhs, BatchSet &batch_set) {
    if (lhs.kind != rhs.kind) {
      return false;
    }
    if ((lhs.kind == kBatch || lhs.kind == kReUse) && batch_set.find(lhs.batch_id) != batch_set.find(rhs.batch_id)) {
      return false;
    }
    return true;
  }

  bool equal(const ParaType &other, BatchSet &batch_set) { return equal(*this, other, batch_set); }

  void print(raw_ostream &os) const {
    switch (kind) {
    case kInit:
      os << "Init";
      break;
    case kBatch:
      os << "Batch(" << batch_id << ")";
      break;
    case kReUse:
      os << "ReUse(" << batch_id << ")";
      break;
    case kNonPara:
      os << "NonPara";
      break;
    default:
      os << "Unknown";
      break;
    }
  }

  void print(raw_ostream &os, BatchSet &batch_set) {
    switch (kind) {
    case kInit:
      os << "Init";
      break;
    case kBatch:
      os << "Batch(" << batch_set.find(batch_id) << ")";
      break;
    case kReUse:
      os << "ReUse(" << batch_set.find(batch_id) << ")";
      break;
    case kNonPara:
      os << "NonPara";
      break;
    default:
      os << "Unknown";
      break;
    }
  }
};

struct ParaInfo {
  SmallVector<ParaType> info;

  ParaInfo() = default;
  ParaInfo(size_t rank) : info(SmallVector<ParaType>(rank, ParaType())) {}
  static ParaInfo from_val(Value val);
  size_t getRank() const { return info.size(); }

  ParaInfo permute_by(ArrayRef<int64_t> dims) const {
    ParaInfo ret(getRank());
    for (int i = 0; i < (int)dims.size(); ++i) {
      ret.info[i] = info[dims[i]];
    }
    return ret;
  }
  ParaInfo permute_from(ArrayRef<int64_t> dims) const {
    ParaInfo ret(getRank());
    for (int i = 0; i < (int)dims.size(); ++i) {
      ret.info[dims[i]] = info[i];
    }
    return ret;
  }

  void print(raw_ostream &os) const {
    if (info.size() > 0) {
      os << "[";
      for (auto type : info) {
        type.print(os);
        os << ",";
      }
      os << "]";
    } else {
      os << "[Uninit]";
    }
  }

  void print(raw_ostream &os, BatchSet &batch_set) {
    if (info.size() > 0) {
      os << "[";
      for (auto type : info) {
        type.print(os, batch_set);
        os << ",";
      }
      os << "]";
    } else {
      os << "[Uninit]";
    }
  }

  ParaInfo slice_like(const ParaInfo &other) {
    assert(other.getRank() <= getRank());
    ParaInfo ret(other.getRank());
    // right align
    int off = getRank() - other.getRank();
    for (int i = 0; i < (int)other.getRank(); ++i) {
      ret.info[i] = info[i + off];
    }
    return ret;
  }

  void join_(const ParaInfo &other, BatchSet &batch_set) {
    assert(getRank() > 0 && other.getRank() > 0);
    // right align
    if (getRank() < other.getRank()) {
      int off = other.getRank() - getRank();
      for (size_t i = 0; i < getRank(); ++i) {
        info[i].join_(other.info[i + off], batch_set);
      }
    } else {
      int off = getRank() - other.getRank();
      for (size_t i = off; i < getRank(); ++i) {
        info[i].join_(other.info[i - off], batch_set);
      }
    }
  }

  // the rank is the same as lhs
  static ParaInfo join(const ParaInfo &lhs, const ParaInfo &rhs, BatchSet &batch_set) {
    assert(lhs.getRank() != 0 || rhs.getRank() != 0);
    ParaInfo ret = lhs;
    if (ret.getRank() < rhs.getRank()) {
      int off = rhs.getRank() - ret.getRank();
      for (size_t i = 0; i < ret.getRank(); ++i) {
        ret.info[i].join_(rhs.info[i + off], batch_set);
      }
    } else {
      int off = ret.getRank() - rhs.getRank();
      for (size_t i = off; i < ret.getRank(); ++i) {
        ret.info[i].join_(rhs.info[i - off], batch_set);
      }
    }
    return ret;
  }

  static bool equal(const ParaInfo &lhs, const ParaInfo &rhs, BatchSet &batch_set) {
    assert(lhs.getRank() == rhs.getRank());
    for (auto [t0, t1] : llvm::zip(lhs.info, rhs.info)) {
      if (!ParaType::equal(t0, t1, batch_set)) {
        return false;
      }
    }
    return true;
  }

  void set(int idx, ParaType type) {
    if (idx < 0) {
      idx += (int)getRank();
    }
    assert(idx >= 0 && idx < (int)getRank());
    info[idx] = type;
  }
};

class ParallelismAnalysis {
public:
  void initialize(asuka::KernelOp kernel_op);
  void run(asuka::KernelOp kernel_op, bool verbose = false);
  void clear() { val_info.clear(); }
  void dump() {
    for (auto &pair : val_info) {
      auto &val = pair.first;
      auto &info = pair.second;
      std::string str;
      llvm::raw_string_ostream os(str);
      info.print(os, batch_set);

      val.dump();
      llvm::errs() << "\t" << str << "\n";
    }
  }

  ParaInfo getInfo(Value val) { return val_info[val]; }

  BatchSet batch_set;

private:
  DenseMap<Value, ParaInfo> val_info;
};

} // namespace mlir

#endif // ASUKA_ANALYSIS_PARALLELISM_H