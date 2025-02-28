module {
  func.func @EA(%arg0: tensor<64x784x128xf32>, %arg1: tensor<384x128xf32>, %arg2: tensor<384xf32>, %arg3: tensor<128x128xf32>, %arg4: tensor<128xf32>) -> tensor<64x784x128xf32> {
    %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<384x128xf32>) -> tensor<128x384xf32>
    %1 = asuka.dot %arg0, %0, batch_dims = [0] x [], reduce_dims = [2] x [0] : (tensor<64x784x128xf32>, tensor<128x384xf32>) -> tensor<64x784x384xf32>
    %2 = asuka.add %1, %arg2 : (tensor<64x784x384xf32>, tensor<384xf32>) -> tensor<64x784x384xf32>
    %3:3 = asuka.split %2, dim = 2 : (tensor<64x784x384xf32>) -> (tensor<64x784x128xf32>, tensor<64x784x128xf32>, tensor<64x784x128xf32>)
    %4 = stablehlo.reshape %3#0 : (tensor<64x784x128xf32>) -> tensor<64x784x4x32xf32>
    %5 = stablehlo.reshape %3#1 : (tensor<64x784x128xf32>) -> tensor<64x784x4x32xf32>
    %6 = stablehlo.reshape %3#2 : (tensor<64x784x128xf32>) -> tensor<64x784x4x32xf32>
    %7 = stablehlo.transpose %4, dims = [0, 2, 1, 3] : (tensor<64x784x4x32xf32>) -> tensor<64x4x784x32xf32>
    %8 = stablehlo.transpose %5, dims = [0, 2, 1, 3] : (tensor<64x784x4x32xf32>) -> tensor<64x4x784x32xf32>
    %9 = stablehlo.transpose %6, dims = [0, 2, 1, 3] : (tensor<64x784x4x32xf32>) -> tensor<64x4x784x32xf32>
    %10 = asuka.softmax %7, dim = 3 : (tensor<64x4x784x32xf32>) -> tensor<64x4x784x32xf32>
    %11 = asuka.softmax %8, dim = 2 : (tensor<64x4x784x32xf32>) -> tensor<64x4x784x32xf32>
    %12 = stablehlo.transpose %11, dims = [0, 1, 3, 2] : (tensor<64x4x784x32xf32>) -> tensor<64x4x32x784xf32>
    %13 = asuka.dot %12, %9, batch_dims = [0, 1] x [0, 1], reduce_dims = [3] x [2] : (tensor<64x4x32x784xf32>, tensor<64x4x784x32xf32>) -> tensor<64x4x32x32xf32>
    %14 = asuka.dot %10, %13, batch_dims = [0, 1] x [0, 1], reduce_dims = [3] x [2] : (tensor<64x4x784x32xf32>, tensor<64x4x32x32xf32>) -> tensor<64x4x784x32xf32>
    %15 = stablehlo.transpose %14, dims = [0, 2, 1, 3] : (tensor<64x4x784x32xf32>) -> tensor<64x784x4x32xf32>
    %16 = stablehlo.reshape %15 : (tensor<64x784x4x32xf32>) -> tensor<64x784x128xf32>
    %17 = stablehlo.transpose %arg3, dims = [1, 0] : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %18 = asuka.dot %16, %17, batch_dims = [0] x [], reduce_dims = [2] x [0] : (tensor<64x784x128xf32>, tensor<128x128xf32>) -> tensor<64x784x128xf32>
    %19 = asuka.add %18, %arg4 : (tensor<64x784x128xf32>, tensor<128xf32>) -> tensor<64x784x128xf32>
    return %19 : tensor<64x784x128xf32>
  }
}

