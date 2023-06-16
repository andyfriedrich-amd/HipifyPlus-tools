// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"
#include "orttraining/core/graph/horovod_adapters.h"
#include "orttraining/core/graph/optimizer_config.h"

namespace onnxruntime {
namespace rocm {

class HorovodAllReduce final : public RocmKernel {
 public:
  HorovodAllReduce(const OpKernelInfo& info) : RocmKernel(info) {
    unique_name = "AllReduceNode_" + info.node().Name();
    int64_t reduce_op;
    // bugbug
    int64_t adasum_type = training::AdasumReductionType::None;
    info.GetAttrOrDefault("reduce_op", &reduce_op, static_cast<int64_t>(hvd::ReduceOp::SUM));
    info.GetAttrOrDefault("reduce_algo", &adasum_type, static_cast<int64_t>(training::AdasumReductionType::None));
    reduce_op_ = GetReduceOp(reduce_op);
    adasum_type_ = static_cast<training::AdasumReductionType>(adasum_type);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  std::string unique_name;
  hvd::ReduceOp reduce_op_;
  training::AdasumReductionType adasum_type_;
};

class HorovodBarrier final : public RocmKernel {
 public:
  HorovodBarrier(const OpKernelInfo& info) : RocmKernel(info) {
    // bugbug
    int64_t adasum_type = training::AdasumReductionType::None;
    info.GetAttrOrDefault("reduce_algo", &adasum_type, static_cast<int64_t>(training::AdasumReductionType::None));
    adasum_type_ = static_cast<training::AdasumReductionType>(adasum_type);
  }
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  training::AdasumReductionType adasum_type_;
};

}  // namespace rocm
}  // namespace onnxruntime
