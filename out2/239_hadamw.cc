

#include <memory>
#include <utility>
#include "core/providers/shared_library/provider_api.h"
#include "orttraining/training_ops/rocm/optimizer/adamw/adamw.h"
#include "orttraining/training_ops/rocm/optimizer/adamw/adamw_impl.h"
#include "orttraining/training_ops/rocm/optimizer/common.h"
namespace onnxruntime {
namespace rocm {
ONNX_OPERATOR_KERNEL_EX(
  AdamWOptimizer, kMSDomain, 1, kRocmExecutionProvider, (*KernelDefBuilder::Create())
    .InputMemoryType(OrtMemTypeCPUInput, 0)
    .InputMemoryType(OrtMemTypeCPUInput, 1)
    .InputMemoryType(OrtMemTypeCPUInput, 6)
    .OutputMemoryType(OrtMemTypeCPUOutput, 0)
    .Alias(2, 1) 
    .Alias(4, 2) 
    .Alias(5, 3) 
    .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
    .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>())
    .TypeConstraint("S_WEIGHT", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
    .TypeConstraint("S_GRAD", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
    .TypeConstraint("S_MOMENT", DataTypeImpl::AllFixedSizeSequenceTensorTypes()), AdamWOptimizer);
Status AdamWOptimizer::ComputeInternal(OpKernelContext* ctx) const {
 AdamWOptimizerBase::Prepare p;
 ORT_RETURN_IF_ERROR(PrepareForCompute(ctx, p));
 int64_t* updated_flag_ptr = p.updated_flag->template MutableData<int64_t>();
 
 const Tensor* update_signal = ctx->Input<Tensor>(6);
 if (update_signal == nullptr || *update_signal->template Data<bool>()) {
  typedef typename ToHipType<float>::MappedType HipT_FLOAT;
  typedef AdamWMTAFunctor<HipT_FLOAT, HipT_FLOAT, HipT_FLOAT> TFunctor;
  TFunctor functor;
  const float* lr_ptr = p.learning_rate->template Data<float>();
  const int64_t* step_ptr = p.step->template Data<int64_t>();
  ORT_ENFORCE(lr_ptr && step_ptr);
  launch_multi_tensor_functor<MTA_ADAMW_GROUP_SIZE, TFunctor>(
    Stream(ctx), MTA_ADAMW_CHUNK_SIZE, p.grouped_tensor_sizes, p.grouped_tensor_pointers, functor, alpha_, beta_, epsilon_, *lr_ptr, weight_decay_, adam_mode_, correct_bias_, *step_ptr);
  *updated_flag_ptr = 1;
 } else {
  *updated_flag_ptr = 0;
 }
 if (p.updated_weights != nullptr) {
  ORT_RETURN_IF_ERROR(CopyIfNotSameROCMBuffer(ctx, p.num_of_weights, p.weights, p.updated_weights));
 }
 if (p.updated_momentums_1 != nullptr) {
  ORT_RETURN_IF_ERROR(CopyIfNotSameROCMBuffer(ctx, p.num_of_weights, p.momentums_1, p.updated_momentums_1));
 }
 if (p.updated_momentums_2 != nullptr) {
  ORT_RETURN_IF_ERROR(CopyIfNotSameROCMBuffer(ctx, p.num_of_weights, p.momentums_2, p.updated_momentums_2));
 }
 return Status::OK();
}
} 
} 