module {
  func.func @Attn(%arg0: tensor<4096x4096xf16>, %arg1: tensor<4096x4096xf16>, %arg2: tensor<4096x4096xf16>) -> tensor<4096x4096xf16> {
    %cst = arith.constant dense<11.313708498984761> : tensor<1xf64>
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
    %12 = stablehlo.convert %11 : (tensor<32x4096x4096xf32>) -> tensor<32x4096x4096xf16>
    %13 = asuka.dot %12, %5, batch_dims = [0] x [0], reduce_dims = [2] x [1] : (tensor<32x4096x4096xf16>, tensor<32x4096x128xf16>) -> tensor<32x4096x128xf16>
    %14 = stablehlo.transpose %13, dims = [1, 0, 2] : (tensor<32x4096x128xf16>) -> tensor<4096x32x128xf16>
    %15 = stablehlo.reshape %14 : (tensor<4096x32x128xf16>) -> tensor<4096x4096xf16>
    return %15 : tensor<4096x4096xf16>
  }
}

