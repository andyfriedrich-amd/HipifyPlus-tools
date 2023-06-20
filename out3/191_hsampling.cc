// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_execution_provider.h"
#include "contrib_ops/rocm/transformers/sampling.h"
#include "contrib_ops/rocm/transformers/generation_device_helper.h"
#include "contrib_ops/rocm/transformers/dump_rocm_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

ONNX_OPERATOR_KERNEL_EX(
    Sampling,
    kMSDomain,
    1,
    kRocmExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)    // 'input_ids' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 1)    // 'max_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 2)    // 'min_length' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 3)    // 'repetition_penalty' needs to be on CPU
        .InputMemoryType(OrtMemTypeCPUInput, 6)    // 'custom_attention_mask' needs to be on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)  // 'sequences' output on CPU
        .OutputMemoryType(OrtMemTypeCPUOutput, 1)  // 'logits_to_debug' output on CPU
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    Sampling);

transformers::HipTensorConsoleDumper g_rocm_dumper_sampling;

Sampling::Sampling(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::Sampling(info) {
  SetDeviceHelpers(GenerationCudaDeviceHelper::ReorderPastState,
                   GenerationCudaDeviceHelper::AddToFeeds,
                   GenerationCudaDeviceHelper::TopK,
                   GenerationCudaDeviceHelper::DeviceCopy<float>,
                   GenerationCudaDeviceHelper::GreedySearchProcessLogits<float>,
                   GenerationCudaDeviceHelper::GreedySearchProcessLogits<MLFloat16>,
                   GenerationCudaDeviceHelper::InitGreedyState<float>,
                   GenerationCudaDeviceHelper::InitGreedyState<MLFloat16>);

  SetDeviceHelpers_Gpt(GenerationCudaDeviceHelper::UpdateGptFeeds<float>,
                       GenerationCudaDeviceHelper::UpdateGptFeeds<MLFloat16>);

  SetConsoleDumper(&g_rocm_dumper_sampling);

  gpu_device_prop_ = &reinterpret_cast<const ROCMExecutionProvider*>(info.GetExecutionProvider())->GetDeviceProp();

  gpu_device_arch_ = static_cast<const hipDeviceProp_t*>(gpu_device_prop_)->major * 100 +
                     static_cast<const hipDeviceProp_t*>(gpu_device_prop_)->minor * 10;
}

Status Sampling::ComputeInternal(OpKernelContext* context) const {
  return onnxruntime::contrib::transformers::Sampling::Compute(context);
}

Status Sampling::Compute(OpKernelContext* context) const {
  auto s = ComputeInternal(context);

  if (s.IsOK()) {
    auto err = hipGetLastError();
    if (err != hipSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "ROCM error ", hipGetErrorName(err), ":", hipGetErrorString(err));
    }
  }

  return s;
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
