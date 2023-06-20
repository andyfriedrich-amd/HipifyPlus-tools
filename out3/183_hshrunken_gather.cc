// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.

#include "contrib_ops/rocm/tensor/shrunken_gather.h"
#include "contrib_ops/cpu/tensor/shrunken_gather.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

ONNX_OPERATOR_KERNEL_EX(
    ShrunkenGather,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{
                                    DataTypeImpl::GetTensorType<int32_t>(),
                                    DataTypeImpl::GetTensorType<int64_t>()}),
    ShrunkenGather);

Status ShrunkenGather::ComputeInternal(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));
  ShrunkenGatherCommon::CheckInput(p.input_tensor, p.indices_tensor, p.axis);
  return onnxruntime::rocm::Gather::ComputeInternal(context);
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime

#endif
