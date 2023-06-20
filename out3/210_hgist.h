// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
class GistBinarizeEncoderOp final : public RocmKernel {
 public:
  GistBinarizeEncoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistBinarizeDecoderOp final : public RocmKernel {
 public:
  GistBinarizeDecoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack1EncoderOp final : public RocmKernel {
 public:
  static constexpr int GIST_PACK1_FACTOR = 8;
  GistPack1EncoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack1DecoderOp final : public RocmKernel {
 public:
  static constexpr int GIST_PACK1_FACTOR = 8;
  GistPack1DecoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack8EncoderOp final : public RocmKernel {
 public:
  GistPack8EncoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack8DecoderOp final : public RocmKernel {
 public:
  GistPack8DecoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack16EncoderOp final : public RocmKernel {
 public:
  GistPack16EncoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack16DecoderOp final : public RocmKernel {
 public:
  GistPack16DecoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPackMsfp15EncoderOp final : public RocmKernel {
 public:
  GistPackMsfp15EncoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPackMsfp15DecoderOp final : public RocmKernel {
 public:
  GistPackMsfp15DecoderOp(const OpKernelInfo& info) : RocmKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace rocm
}  // namespace onnxruntime
