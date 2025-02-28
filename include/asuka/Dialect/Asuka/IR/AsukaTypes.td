#ifndef ASUKA_TYPES
#define ASUKA_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "asuka/Dialect/Asuka/IR/AsukaDialect.td"


// Floating-point Type
def AS_Float : AnyTypeOf<[F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64], "floating-point">;
def AS_FloatTensor : RankedTensorOf<[AS_Float]>;
def AS_FloatLike : AnyTypeOf<[AS_Float, AS_FloatTensor]>;

// Boolean Type
def AS_Bool : I<8>;
def AS_BoolTensor : RankedTensorOf<[AS_Bool]>;
def AS_BoolLike : AnyTypeOf<[AS_Bool, AS_BoolTensor]>;

// Integer Type
def I4 : I<4>;
def AS_Int : AnyTypeOf<[I1, I4, I8, I16, I32, I64], "integer">;
def AS_IntTensor : RankedTensorOf<[AS_Int]>;
def AS_IntLike : AnyTypeOf<[AS_Int, AS_IntTensor]>;

def AS_FloatIntLike : AnyTypeOf<[AS_IntLike, AS_FloatLike]>;
def AS_FloatIntTensor : AnyTypeOf<[AS_IntTensor, AS_FloatTensor]>;


// TODO: new type
def AS_ShapedOnlyTensor : RankedTensorOf<[Index]>;
def AS_TypedTensor : AnyTypeOf<[AS_IntTensor, AS_FloatTensor]>;

def AS_AnyTensor : AnyTypeOf<[AS_ShapedOnlyTensor, AS_TypedTensor]>;

/*

class AsukaTypeDef<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<Asuka_Dialect, name, traits> {

  let mnemonic = typeMnemonic;
}


class AS_ShapeType : AsukaTypeDef<"Shape", "shape"> {
  let summary = "";
  let description = [{}];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Attribute":$encoding
  );

  let builders = [
    TypeBuilder<(ins "ArrayRef<int64_t>":$shape, CArg<"Attribute", "{}">:$encoding), [{
      return Base::get($_ctxt, shape, encoding);
    }]>
  ];

  let assemblyFormat = "`<` $shape `,` $encoding `>`";

  let skipDefaultBuilders = 1;
  // let genVerifyDecl = 1;
}


def AS_Tensor : AynTypeOf<[AS_IntTensor, AS_FloatTensor, AS_ShapeType]>;
*/

#endif // ASUKA_TYPES
