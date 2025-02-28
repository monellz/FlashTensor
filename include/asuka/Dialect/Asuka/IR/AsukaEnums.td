#ifndef ASUKA_ENUMS
#define ASUKA_ENUMS


include "asuka/Dialect/Asuka/IR/AsukaDialect.td"

include "mlir/IR/EnumAttr.td" 

def ASUKA_RNG_DISTRIBUTION_UNIT_UNIFORM : I32EnumAttrCase<"UNIT_UNIFORM", 1>;
def ASUKA_RNG_DISTRIBUTION_UNIT_GUMBEL : I32EnumAttrCase<"UNIT_GUMBEL", 2>;
def Asuka_RngDistribution : I32EnumAttr<"RngDistribution",
    "XLA PRNG distribution to be used.",
    [
      ASUKA_RNG_DISTRIBUTION_UNIT_UNIFORM,
      ASUKA_RNG_DISTRIBUTION_UNIT_GUMBEL
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::asuka";
}

def Asuka_RngDistributionAttr : EnumAttr<Asuka_Dialect, Asuka_RngDistribution, "rng_distribution">;



def ASUKA_REDUCE_ADD : I32EnumAttrCase<"ADD", 1>;
def ASUKA_REDUCE_MUL : I32EnumAttrCase<"MUL", 2>;
def ASUKA_REDUCE_ANY : I32EnumAttrCase<"ANY", 3>;
def Asuka_ReduceType : I32EnumAttr<"ReduceType",
    "Reduce type",
    [
      ASUKA_REDUCE_ADD,
      ASUKA_REDUCE_MUL,
      ASUKA_REDUCE_ANY
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::asuka";
}


def Asuka_ReduceTypeAttr : EnumAttr<Asuka_Dialect, Asuka_ReduceType, "reduce_type">;



def ASUKA_CMP_GT : I32EnumAttrCase<"GT", 1>;
def ASUKA_CMP_GE : I32EnumAttrCase<"GE", 2>;
def Asuka_CmpType: I32EnumAttr<"CmpType",
    "Compare Type",
    [
      ASUKA_CMP_GT,
      ASUKA_CMP_GE
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::asuka";
}
def Asuka_CmpTypeAttr : EnumAttr<Asuka_Dialect, Asuka_CmpType, "cmp_type">;

#endif // ASUKA_ENUMS