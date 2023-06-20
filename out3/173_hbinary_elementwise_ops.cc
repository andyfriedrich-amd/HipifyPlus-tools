

#include "contrib_ops/rocm/math/binary_elementwise_ops.h"
#include "contrib_ops/rocm/math/binary_elementwise_ops_impl.h"
using namespace onnxruntime::common;
namespace onnxruntime {
namespace contrib {
namespace rocm {
#define CONTRIB_BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(x, ver, T)              ONNX_OPERATOR_TYPED_KERNEL_EX(                                 x, kMSDomain, ver, T, kRocmExecutionProvider, (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), x<T>);
#define CONTRIB_BINARY_ELEMENTWISE_COMPUTE(x, T)                              template <>                                               Status x<T>::ComputeInternal(OpKernelContext* context) const {                       BinaryElementwisePreparation prepare;                                  ORT_RETURN_IF_ERROR(Prepare(context, &prepare));                             Impl_##x<typename ToHipType<T>::MappedType>(                                Stream(context), prepare.output_rank_or_simple_broadcast, &prepare.lhs_padded_strides, reinterpret_cast<const typename ToHipType<T>::MappedType*>(prepare.lhs_tensor->Data<T>()), &prepare.rhs_padded_strides, reinterpret_cast<const typename ToHipType<T>::MappedType*>(prepare.rhs_tensor->Data<T>()), &prepare.fdm_output_strides, prepare.fdm_H, prepare.fdm_C, reinterpret_cast<typename ToHipType<T>::MappedType*>(prepare.output_tensor->MutableData<T>()), prepare.output_tensor->Shape().Size());                               return Status::OK();                                          }
#define CONTRIB_BINARY_OP_TYPED(name, ver, T)            CONTRIB_BINARY_ELEMENTWISE_REGISTER_KERNEL_TYPED(name, ver, T)  CONTRIB_BINARY_ELEMENTWISE_COMPUTE(name, T)














#define CONTRIB_BINARY_OP_HFD(name, ver)      CONTRIB_BINARY_OP_TYPED(name, ver, MLFloat16)  CONTRIB_BINARY_OP_TYPED(name, ver, float)    CONTRIB_BINARY_OP_TYPED(name, ver, double)    CONTRIB_BINARY_OP_TYPED(name, ver, BFloat16)
CONTRIB_BINARY_OP_HFD(BiasGelu, 1)
} 
} 
} 