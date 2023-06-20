// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include <stdio.h>
#include <assert.h>
#include <hip/hip_runtime.h>

namespace onnxruntime {
namespace rocm {
static constexpr int GIST_PACK1_FACTOR = 8;
template <typename T>
void GistBinarizeEncoderImpl(
    hipStream_t stream,
    const T* input_data,
    bool* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistBinarizeDecoderImpl(
    hipStream_t stream,
    const bool* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack1EncoderImpl(
    hipStream_t stream,
    const T* input_data,
    uint8_t* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack1DecoderImpl(
    hipStream_t stream,
    const uint8_t* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack8EncoderImpl(
    hipStream_t stream,
    const T* input_data,
    uint8_t* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack8DecoderImpl(
    hipStream_t stream,
    const uint8_t* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack16EncoderImpl(
    hipStream_t stream,
    const T* input_data,
    half* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack16DecoderImpl(
    hipStream_t stream,
    const half* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPackMsfp15EncoderImpl(
    hipStream_t stream,
    const T* input_data,
    uint8_t* output_data,
    const size_t pre_axis_size,
    const size_t axis_size,
    const size_t tile_size);

template <typename T>
void GistPackMsfp15DecoderImpl(
    hipStream_t stream,
    const uint8_t* input_data,
    T* output_data,
    const size_t pre_axis_size,
    const size_t axis_size,
    const size_t tile_size);

}  // namespace rocm
}  // namespace onnxruntime
