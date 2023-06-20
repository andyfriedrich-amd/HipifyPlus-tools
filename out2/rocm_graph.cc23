// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/rocm_graph.h"

#include "core/providers/rocm/rocm_common.h"
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>

namespace onnxruntime {

ROCMGraph::ROCMGraph(hipStream_t stream) : stream_(stream) {
}

void ROCMGraph::SetStream(hipStream_t stream) {
  stream_ = stream;
}

void ROCMGraph::CaptureBegin() {
  ORT_ENFORCE(!has_graph_exec_,
              "This rocm graph has already captured a graph. "
              "Create a new instance to capture a new graph.");

  HIP_CALL_THROW(hipStreamSynchronize(stream_));
  // For now rocm graph can only work with a single thread. In the future, we
  // will support multiple threads. For multiple threads with multiple graphs
  // and streams, `hipStreamCaptureModeGlobal` needs to be changed to
  // `hipStreamCaptureModeThreadLocal`
  HIP_CALL_THROW(hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal));
}

void ROCMGraph::CaptureEnd() {
  HIP_CALL_THROW(hipStreamEndCapture(stream_, &graph_));
  if (graph_ == NULL) {
    ORT_THROW("ROCMGraph::CaptureEnd: graph_ is NULL");
  }

  has_graph_ = true;
  HIP_CALL_THROW(hipGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  has_graph_exec_ = true;
  HIP_CALL_THROW(hipGraphDestroy(graph_));
  has_graph_ = false;
}

Status ROCMGraph::Replay() {
  // Although this function is not thread safe, the lock is not needed here because
  // ROCM EP maintains a separate rocm graph per thread
  LOGS_DEFAULT(INFO) << "Replaying ROCM graph on stream " << stream_;
  HIP_RETURN_IF_ERROR(hipGraphLaunch(graph_exec_, stream_));
  HIP_RETURN_IF_ERROR(hipStreamSynchronize(stream_));
  return Status::OK();
}

void ROCMGraph::Reset() {
  if (has_graph_) {
    HIP_CALL_THROW(hipGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    HIP_CALL_THROW(hipGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

ROCMGraph::~ROCMGraph() {
  Reset();
}

}  // namespace onnxruntime
