module {
  func.func @KeyFormer(%arg0: tensor<4096x4096xf16>, %arg1: tensor<4096x4096xf16>, %arg2: tensor<4096x4096xf16>) -> (tensor<4096x4096xf16>, tensor<32x4096xf32>) {
    %cst = arith.constant dense<11.313708498984761> : tensor<1xf64>
    %cst_0 = arith.constant dense<1.500000e+00> : tensor<1xf64>
    %0 = stablehlo.reshape %arg0 : (tensor<4096x4096xf16>) -> tensor<4096x32x128xf16>
    %1 = stablehlo.reshape %arg1 : (tensor<4096x4096xf16>) -> tensor<4096x32x128xf16>
    %2 = stablehlo.reshape %arg2 : (tensor<4096x4096xf16>) -> tensor<4096x32x128xf16>
    %3 = stablehlo.transpose %0, dims = [1, 0, 2] : (tensor<4096x32x128xf16>) -> tensor<32x4096x128xf16>
    %4 = stablehlo.transpose %1, dims = [1, 0, 2] : (tensor<4096x32x128xf16>) -> tensor<32x4096x128xf16>
    %5 = stablehlo.transpose %2, dims = [1, 0, 2] : (tensor<4096x32x128xf16>) -> tensor<32x4096x128xf16>
    %6 = stablehlo.transpose %4, dims = [0, 2, 1] : (tensor<32x4096x128xf16>) -> tensor<32x128x4096xf16>
    %7 = asuka.dot %3, %6, batch_dims = [0] x [0], reduce_dims = [2] x [1] : (tensor<32x4096x128xf16>, tensor<32x128x4096xf16>) -> tensor<32x4096x4096xf16>
    %8 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xf32>
    %9 = stablehlo.convert %7 : (tensor<32x4096x4096xf16>) -> tensor<32x4096x4096xf32>
    %10 = asuka.div %9, %8 : (tensor<32x4096x4096xf32>, tensor<1xf32>) -> tensor<32x4096x4096xf32>
    %11 = asuka.softmax %10, dim = 2 : (tensor<32x4096x4096xf32>) -> tensor<32x4096x4096xf32>
    %12 = asuka.rng distribution =  UNIT_GUMBEL : tensor<32x4096x4096xf32>
    %13 = asuka.add %10, %12 : (tensor<32x4096x4096xf32>, tensor<32x4096x4096xf32>) -> tensor<32x4096x4096xf32>
    %14 = stablehlo.convert %cst_0 : (tensor<1xf64>) -> tensor<1xf32>
    %15 = asuka.div %13, %14 : (tensor<32x4096x4096xf32>, tensor<1xf32>) -> tensor<32x4096x4096xf32>
    %16 = asuka.softmax %15, dim = 2 : (tensor<32x4096x4096xf32>) -> tensor<32x4096x4096xf32>
    %17 = asuka.reduce %16, dim = 1, op =  ADD : (tensor<32x4096x4096xf32>) -> tensor<32x4096xf32>
    %18 = stablehlo.convert %11 : (tensor<32x4096x4096xf32>) -> tensor<32x4096x4096xf16>
    %19 = asuka.dot %18, %5, batch_dims = [0] x [0], reduce_dims = [2] x [1] : (tensor<32x4096x4096xf16>, tensor<32x4096x128xf16>) -> tensor<32x4096x128xf16>
    %20 = stablehlo.transpose %19, dims = [1, 0, 2] : (tensor<32x4096x128xf16>) -> tensor<4096x32x128xf16>
    %21 = stablehlo.reshape %20 : (tensor<4096x32x128xf16>) -> tensor<4096x4096xf16>
    return %21, %17 : tensor<4096x4096xf16>, tensor<32x4096xf32>
  }
}

