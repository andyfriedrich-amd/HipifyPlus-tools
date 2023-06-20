// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/platform/ort_mutex.h"
#include "core/providers/rocm/rocm_pch.h"

namespace onnxruntime {

using CaptureId_t = unsigned long long;

struct ROCMGraph {
  ROCMGraph(){};
  ROCMGraph(hipStream_t stream);
  ~ROCMGraph();

  void SetStream(hipStream_t stream);
  void CaptureBegin();
  void CaptureEnd();
  Status Replay();
  void Reset();

 private:
  hipGraph_t graph_ = NULL;
  hipGraphExec_t graph_exec_ = NULL;

  bool has_graph_ = false;
  bool has_graph_exec_ = false;

  hipStream_t stream_ = nullptr;  // Does not own the stream
};

}  // namespace onnxruntime
