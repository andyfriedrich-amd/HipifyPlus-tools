

#pragma once
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
namespace onnxruntime {
namespace cuda {
template <bool is_dropout>
class IdentityOp final : public CudaKernel {
 public:
 IdentityOp(const OpKernelInfo& info) : CudaKernel(info) {
 }
 Status ComputeInternal(OpKernelContext* context) const override {
  auto X_ml_type = context->InputType(0);
  if (X_ml_type->IsTensorType()) {
   const Tensor* X = context->Input<Tensor>(0);
   if (nullptr == X) {
    return Status(common::ONNXRUNTIME, common::FAIL, "IdentityOp cuda: input count mismatch.");
   }
   const TensorShape& shape = X->Shape();
   Tensor* Y = context->Output(0, shape);
   if (nullptr == Y) {
    return Status(common::ONNXRUNTIME, common::FAIL, "IdentityOp cuda: failed to allocate output tensor.");
   }
   auto X_type = X->DataType();
   const void* source = X->DataRaw(X_type);
   void* target = Y->MutableDataRaw(X_type);
   
   if (target != source) {
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, X->Shape().Size() * X->DataType()->Size(), cudaMemcpyDeviceToDevice, Stream(context)));
   }
   if (is_dropout) {
    Tensor* mask = context->Output(1, shape);
    
    if (mask != nullptr) {
     
     
     
     
     void* mask_data = mask->MutableDataRaw();
     
     
     CUDA_RETURN_IF_ERROR(cudaMemsetAsync(mask_data, 0, mask->SizeInBytes(), Stream(context)));
    }
   }
  } else if (X_ml_type->IsTensorSequenceType()) {
   const TensorSeq* X = context->Input<TensorSeq>(0);
   ORT_ENFORCE(X != nullptr, "IdentityOp cuda: input tensor is missing.");
   TensorSeq* Y = context->Output<TensorSeq>(0);
   ORT_ENFORCE(Y != nullptr, "IdentityOp cuda: failed to allocate output tensor sequence.");
   if (X == Y) {
    return Status::OK();
   }
   auto X_type = X->DataType();
   Y->SetType(X_type);
   AllocatorPtr alloc;
   auto status = context->GetTempSpaceAllocator(&alloc);
   if (!status.IsOK()) {
    return Status(common::ONNXRUNTIME, common::FAIL, "IdentityOp cuda: unable to get an allocator.");
   }
   auto X_size = X->Size();
   Y->Reserve(X_size);
   for (size_t i = 0; i < X_size; ++i) {
    const Tensor& source_tensor = X->Get(i);
    std::unique_ptr<Tensor> target_tensor = Tensor::Create(source_tensor.DataType(), source_tensor.Shape(), alloc);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target_tensor->MutableDataRaw(), source_tensor.DataRaw(), source_tensor.SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(context)));
    Y->Add(std::move(*target_tensor));
   }
  } else {
   return Status(common::ONNXRUNTIME, common::FAIL, "IdentityOp cuda: unsupported input type.");
  }
  return Status::OK();
 }
};
} 
} 