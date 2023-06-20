





#ifndef NDEBUG
#include <iostream>
#include "core/providers/rocm/test/all_tests.h"
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/rocm_allocator.h"
#include "core/providers/rocm/rocm_stream_handle.h"
namespace onnxruntime {
namespace rocm {
namespace test {


bool TestDeferredRelease() {
 
 ROCMExecutionProviderInfo info;
 ROCMExecutionProvider ep(info);
 
 onnxruntime::AllocatorManager allocator_manager;
 ep.RegisterAllocator(allocator_manager);
 AllocatorPtr gpu_alloctor = ep.GetAllocator(OrtMemType::OrtMemTypeDefault);
 
 
 AllocatorPtr cpu_pinned_alloc = ep.GetAllocator(OrtMemTypeCPU);
 
 
 RocmStream stream(nullptr, gpu_alloctor->Info().device, cpu_pinned_alloc, false, true, nullptr, nullptr);
 
 const size_t n_bytes = 10 * 1000000;
 const int64_t n_allocs = 64;
 ORT_THROW_IF_ERROR(ep.OnRunStart());
 for (size_t i = 0; i < n_allocs; ++i) {
  
  auto pinned_buffer = ep.AllocateBufferOnCPUPinned<void>(n_bytes);
  
  stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
 }
 
 AllocatorStats stats;
 cpu_pinned_alloc->GetStats(&stats);
 ORT_ENFORCE(stats.num_allocs == n_allocs);
 ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
 ORT_THROW_IF_ERROR(ep.OnRunEnd(true));
 return true;
}
bool TestDeferredReleaseWithoutArena() {
 
 ROCMExecutionProviderInfo info;
 ROCMExecutionProvider ep(info);
 
 onnxruntime::AllocatorManager allocator_manager;
 OrtDevice pinned_device{OrtDevice::CPU, OrtDevice::MemType::CUDA_PINNED, DEFAULT_CPU_ALLOCATOR_DEVICE_ID};
 
 AllocatorCreationInfo pinned_memory_info(
   [](OrtDevice::DeviceId device_id) {
    return std::make_unique<ROCMPinnedAllocator>(device_id, CUDA_PINNED);
   }, pinned_device.Id(), false );
 auto rocm_pinned_alloc = CreateAllocator(pinned_memory_info);
 allocator_manager.InsertAllocator(rocm_pinned_alloc);
 
 
 ep.RegisterAllocator(allocator_manager);
 AllocatorPtr gpu_alloctor = ep.GetAllocator(OrtMemType::OrtMemTypeDefault);
 
 
 AllocatorPtr cpu_pinned_alloc = ep.GetAllocator(OrtMemTypeCPU);
 
 
 RocmStream stream(nullptr, gpu_alloctor->Info().device, cpu_pinned_alloc, false, true, nullptr, nullptr);
 
 const size_t n_bytes = 10 * 1000000;
 const int64_t n_allocs = 64;
 ORT_THROW_IF_ERROR(ep.OnRunStart());
 for (size_t i = 0; i < n_allocs; ++i) {
  
  auto pinned_buffer = ep.AllocateBufferOnCPUPinned<void>(n_bytes);
  
  stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
 }
 ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
 ORT_THROW_IF_ERROR(ep.OnRunEnd(true));
 return true;
}
} 
} 
} 
#endif