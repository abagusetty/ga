#ifndef _DEV_MEM_HANDLE_H
#define _DEV_MEM_HANDLE_H


#if defined(ENABLE_DEVICE)

#if defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
// #include <hip/hip_runtime_api.h>
typedef struct {
  hipIpcMemHandle_t handle;
} devMemHandle_t;
#endif // ENABLE_HIP

#if defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include "cublas_v2.h"
typedef struct {
  cudaIpcMemHandle_t handle;
} devMemHandle_t;
#endif

#if defined(ENABLE_SYCL)
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#include <sycl/backend/level_zero.hpp>

#include <sycl/sycl.hpp>

#include <oneapi/mkl/blas.hpp>
typedef struct {
  ze_device_handle_t zeDevice;
  ze_context_handle_t zeContext;
  ze_ipc_mem_handle_t handle;
} devMemHandle_t;
#endif

#endif // ENABLE_DEVICE


#endif /*_DEV_MEM_HANDLE_H*/
