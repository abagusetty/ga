#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "dev_mem_handle.h"

#include "comex.h"

/* avoid name mangling by the SYCL compiler */
//extern "C" {

extern int MPI_Wrapper_world_rank();
extern void MPI_Wrapper_abort(int err);

/* return the number of GPUs visible from this processor*/
int numDevices()
{
  int ngpus=0;
  syclGetDeviceCount(&ngpus);
  if (ngpus == 0) {
    int rank = MPI_Wrapper_world_rank();
    std::cout << "p[" << rank << "] Error encountered by syclGetDeviceCount\n";
  }

  return ngpus;
}

/* set the GPU device for this processor
 * id: id of device
 */
void setDevice(int id)
{
  syclSetDevice(id);
}

/* allocate a unified memory buffer
 * buf: pointer to buffer
 * size: size of allocation in bytes
 */
void mallocDevice(void **buf, size_t size)
{
  *buf = sycl::malloc_device(size, *(sycl_get_queue()));
}

/* free unified memory
 * buf: pointer to memory allocation
 */
void freeDevice(void *buf)
{
  sycl::free(buf, *(sycl_get_queue()));
}

/* is pointer located on host?
 * return 1 data is located on host, 0 otherwise
 * ptr: pointer to data
 */
int isHostPointer(void *ptr)
{
  /* Returns usm::alloc::unknown if ptr does not point within a valid USM allocation from syclContext. */
  if( sycl::get_pointer_type(ptr, sycl_get_queue()->get_context()) == sycl::usm::alloc::unknown ) {
    return 1;
  };

  return 0;
}

/* return local ID of device hosting buffer. Return -1
 * if buffer is on host
 * ptr: pointer to data
 */
int getDeviceID(void *ptr)
{
  if ( isHostPointer(ptr) ) {
    return -1;
  }
  else { // USM pointers
    auto syclDev = sycl::get_pointer_device(ptr, sycl_get_queue()->get_context());
    return syclGetDeviceID(&syclDev);
  }
}

/* copy data from host buffer to unified memory
 * devptr: pointer to allocation on device
 * hostptr: pointer to allocation on host
 * bytes: number of bytes to copy
 */
void copyToDevice(void *devptr, void *hostptr, int bytes)
{
  sycl_get_queue()->memcpy( devptr, hostptr, bytes ).wait();
}

/* copy data from unified memory to host buffer
 * hostptr: pointer to allocation on host
 * devptr: pointer to allocation on device
 * bytes: number of bytes to copy
 */
void copyToHost(void *hostptr, void *devptr, int bytes)
{
  sycl_get_queue()->memcpy( hostptr, devptr, bytes ).wait();
}

/* copy data between buffers on same device
 * dstptr: destination pointer
 * srcptr: source pointer
 * bytes: number of bytes to copy
 */
void copyDevToDev(void *dstptr, void *srcptr, int bytes)
{
  sycl_get_queue()->copy( dstptr, srcptr, bytes ).wait();
}

/* copy data between buffers on different devices
 * dstptr: destination pointer
 * dstID: device ID of destination
 * srcptr: source pointer
 * srcID: device ID of source
 * bytes: number of bytes to copy
 */
void copyPeerToPeer(void *dstptr, int dstID, void *srcptr, int srcID, int bytes)
{
  ze_result_t ierr;
  ierr = cudaMemcpyPeer(dstptr,dstID,srcptr,srcID,bytes);
  zeErrCheck(ierr);
  if (ierr != ZE_RESULT_SUCCESS) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemcpyPeer dev to dev msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
  }
}

/**
 * set values on the device to a specific value
 * ptr: pointer to device memory that needs to be set
 * val: integer representation of the value of each byte
 * size: number of bytes that should be set
 */
void deviceMemset(void *ptr, int val, size_t bytes)
{
  sycl_get_queue()->memset( ptr, val, bytes ).wait();
}

void deviceIaxpy(int *dst, int *src, const int *scale, int n)
{
  auto event = sycl_get_queue()->parallel_for<class iaxpy_kernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
      dst[idx] = dst[idx] + scale*src[idx];
    });
  event.wait();
}

void deviceLaxpy(long *dst, long *src, const long *scale, int n)
{
  auto event = sycl_get_queue()->parallel_for<class laxpy_kernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
      dst[idx] = dst[idx] + scale*src[idx];
    });
  event.wait();
}

void deviceAddInt(int *ptr, const int inc)
{
  int tmp;
  copyToHost(&tmp,ptr,sizeof(int));
  tmp += inc;
  copyToDevice(ptr,&tmp,sizeof(int));
}

void deviceAddLong(long *ptr, const long inc)
{
  long tmp;
  copyToHost(&tmp,ptr,sizeof(long));
  tmp += inc;
  copyToDevice(ptr,&tmp,sizeof(long));
}

int deviceGetMemHandle(devMemHandle_t *handle, void *memory)
{
  ze_result_t ierr;
  ierr = zeMemGetIpcHandle(handle->zeContext, memory, &(handle->handle));
  if (ierr != ZE_RESULT_SUCCESS) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    std::cout << "p[" << rank << "] zeMemOpenIpcHandle msg: " << std::hex<<err<<std::dec << std::endl;
    MPI_Wrapper_abort(err);
  }
  return ierr;
}

int deviceOpenMemHandle(void **memory, devMemHandle_t handle)
{
  ze_result_t ierr;
  ierr = zeMemOpenIpcHandle(handle.zeContext, handle.zeDevice, handle.handle, 0, memory);
  if (ierr != ZE_RESULT_SUCCESS) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    std::cout << "p[" << rank << "] zeMemOpenIpcHandle msg: " << std::hex<<err<<std::dec << std::endl;
    MPI_Wrapper_abort(err);
  }
  return ierr;
}

int deviceCloseMemHandle(void *memory, devMemHandle_t handle)
{
  return zeMemCloseIpcHandle(handle.zeContext, memory);
}

#define MAXDIM 7
struct strided_kernel_arg {
  void *dst;
  void *src;
  int dst_strides[MAXDIM];  /* smallest strides are first */
  int src_strides[MAXDIM];
  int dims[MAXDIM];         /* dimensions of block being transferred */
  int stride_levels;        /* dimension of array minus 1 */
  int elem_size;            /* size of array elements */
  int totalCopyElems;       /* total constructs to copy */
  int elements_per_block;   /* number of elements copied by each thread */
  int op;                   /* accumulate operation (if applicable) */
  char scale[64];           /* accumulate scale parameter */
};

#define TTHREADS 1024
void parallelMemcpy(void *src,         /* starting pointer of source data */
                    int *src_stride,   /* strides of source data */
                    void *dst,         /* starting pointer of destination data */
                    int *dst_stride,   /* strides of destination data */
                    int *count,        /* dimensions of data block to be transfered */
                    int stride_levels) /* number of stride levels */
{
  int src_on_host = isHostPointer(src);
  int dst_on_host = isHostPointer(dst);
  void *msrc;
  void *mdst;
  int total_elems;
  int nblocks;
  strided_kernel_arg arg;
  int i;

  /* if src or dst is on host, map pointer to device */
  if (src_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*src_stride[stride_levels-1];
    cudaHostRegister(src, total, cudaHostRegisterMapped);
    /* Register the host pointer */
    cudaHostGetDevicePointer (&msrc, src, 0);
  } else {
    msrc = src;
  }
  if (dst_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*dst_stride[stride_levels-1];
    cudaHostRegister(dst, total, cudaHostRegisterMapped);
    /* Register the host pointer */
    cudaHostGetDevicePointer (&mdst, dst, 0);
  } else {
    mdst = dst;
  }

  /* total_elems = count[0]/elem_size; */
  total_elems = 1;
  for (i=0; i<stride_levels; i++) total_elems *= count[i+1];

  if(total_elems < TTHREADS){
    nblocks = 1;
  } else {
    nblocks = int(ceil(((float)total_elems)/((float)TTHREADS)));
  }

  arg.dst = mdst;
  arg.src = msrc;
  arg.elem_size = count[0];
  for (i=0; i<stride_levels; i++) {
    arg.dst_strides[i] = dst_stride[i]/arg.elem_size;
    arg.src_strides[i] = src_stride[i]/arg.elem_size;
  }
  for (i=0; i<=stride_levels; i++) arg.dims[i] = count[i];
  arg.dims[0] = 1;
  arg.stride_levels = stride_levels;
  /* arg.elem_size = elem_size; */
  arg.totalCopyElems = total_elems;
  arg.elements_per_block = 1;

  auto event = sycl_get_queue()->parallel_for
    <class strided_memcpy_kernel>(sycl::nd_range<1>(nblocks*TTHREADS, TTHREADS), [=](sycl::nd_item<1> item) {
        int index = item.get_local_id(0);
        int stride = item.get_group(0);

        int i;
        int idx[MAXDIM];
        int currElem = 0;
        int elements_per_block = arg.elements_per_block;
        int bytes_per_thread = arg.elem_size*elements_per_block;
        int stride_levels = arg.stride_levels;
        int src_block_offset; /* Offset based on chunk_index */
        int dst_block_offset; /* Offset based on chunk_index */

        /* Determine location of chunk_index in array based
           on the thread id and the block id */
        index = index + stride * item.get_local_range(0);
        /* If the thread index is bigger than the total transfer
           entities then this thread does not participate in the
           copy */
        if(index >= arg.totalCopyElems) {
          return;
        }
        /* Find the indices that mark the location of this element within
           the block of data that will be moved */
        index *= elements_per_block;
        // Calculate the index starting points
        for (i=0; i<=stride_levels; i++) {
          idx[i] = index%arg.dims[i];
          index = (index-idx[i])/arg.dims[i];
        }
        /* Calculate the block offset for this thread */
        src_block_offset = bytes_per_thread*idx[0];
        dst_block_offset = bytes_per_thread*idx[0];
        for (i=0; i<stride_levels; i++) {
          src_block_offset += arg.src_strides[i]*idx[i+1]*bytes_per_thread;
          dst_block_offset += arg.dst_strides[i]*idx[i+1]*bytes_per_thread;
        }

        /* Start copying element by element
           TODO: Make it sure that it is continuous and replace the loop
           with a single memcpy */
        copy((char*)arg.dst + dst_block_offset + currElem * bytes_per_thread,
             (char*)arg.src + src_block_offset + currElem * bytes_per_thread,
             elements_per_block*bytes_per_thread);
        /* Synchronize the threads before returning  */
        sycl::group_barrier(item.get_group());
      });


  /*
  cudaDeviceSynchoronize();
  */
  if (src_on_host) {
    cudaHostUnregister(src);
  }
  if (dst_on_host) {
    cudaHostUnregister(dst);
  }


}

void parallelAccumulate(int op,        /* accumulate operation */
                    void *src,         /* starting pointer of source data */
                    int *src_stride,   /* strides of source data */
                    void *dst,         /* starting pointer of destination data */
                    int *dst_stride,   /* strides of destination data */
                    int *count,        /* dimensions of data block to be transfered */
                    int stride_levels, /* number of stride levels */
                    void *scale)       /* scale factor in accumulate */
{
  int src_on_host = isHostPointer(src);
  int dst_on_host = isHostPointer(dst);
  void *msrc;
  void *mdst;
  int total_elems;
  int elem_size;
  int nblocks;
  strided_kernel_arg arg;
  int i;

  /* if src or dst is on host, map pointer to device */
  if (src_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*src_stride[stride_levels-1];
    cudaHostRegister(src, total, cudaHostRegisterMapped);
    /* Register the host pointer */
    cudaHostGetDevicePointer (&msrc, src, 0);
  } else {
    msrc = src;
  }
  if (dst_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*dst_stride[stride_levels-1];
    cudaHostRegister(dst, total, cudaHostRegisterMapped);
    /* Register the host pointer */
    cudaHostGetDevicePointer (&mdst, dst, 0);
  } else {
    mdst = dst;
  }

  /* total_elems = count[0]/elem_size; */
  if (op == COMEX_ACC_INT) {
    elem_size = sizeof(int);
    *((int*)arg.scale) = *((int*)scale);
  } else if (op == COMEX_ACC_LNG) {
    elem_size = sizeof(long);
    *((long*)arg.scale) = *((long*)scale);
  } else if (op == COMEX_ACC_FLT) {
    elem_size = sizeof(float);
    *((float*)arg.scale) = *((float*)scale);
  } else if (op == COMEX_ACC_DBL) {
    elem_size = sizeof(double);
    *((double*)arg.scale) = *((double*)scale);
  } else if (op == COMEX_ACC_CPL) {
    elem_size = 2*sizeof(float);
    *((float*)arg.scale) = *((float*)scale);
    *(((float*)arg.scale)+1) = *(((float*)scale)+1);
  } else if (op == COMEX_ACC_DCP) {
    elem_size = 2*sizeof(double);
    *((double*)arg.scale) = *((double*)scale);
    *(((double*)arg.scale)+1) = *(((double*)scale)+1);
  }

  total_elems = count[0]/elem_size;
  for (i=0; i<stride_levels; i++) total_elems *= count[i+1];

  if(total_elems < TTHREADS){
    nblocks = 1;
  } else {
    nblocks = int(ceil(((float)total_elems)/((float)TTHREADS)));
  }

  arg.src = msrc;
  arg.dst = mdst;
  arg.elem_size = elem_size;
  arg.op = op;
  for (i=0; i<stride_levels; i++) {
    arg.dst_strides[i] = dst_stride[i];
    arg.src_strides[i] = src_stride[i];
  }
  for (i=0; i<=stride_levels; i++) arg.dims[i] = count[i];
  arg.dims[0] = count[0]/elem_size;
  arg.stride_levels = stride_levels;
  /* arg.elem_size = elem_size; */
  arg.totalCopyElems = total_elems;
  arg.elements_per_block = 1;

  auto event = sycl_get_queue()->parallel_for
    <class strided_accumulate_kernel>(sycl::nd_range<1>(nblocks*TTHREADS, TTHREADS), [=](sycl::nd_item<1> item) {
      int index = item.get_local_id(0);
      int stride = item.get_group(0);

      int i;
      int idx[MAXDIM];
      int elements_per_block = arg.elements_per_block;
      int bytes_per_thread = arg.elem_size*elements_per_block;
      int stride_levels = arg.stride_levels;
      int src_block_offset; /* Offset based on chunk_index */
      int dst_block_offset; /* Offset based on chunk_index */
      void *src, *dst;
      int op;

      /* Determine location of chunk_index in array based
         on the thread id and the block id */
      index = index + stride * item.get_local_range(0);
      /* If the thread index is bigger than the total transfer
         entities then this thread does not participate in the
         copy */
      if(index >= arg.totalCopyElems) {
        return;
      }
      /* Find the indices that mark the location of this element within
         the block of data that will be moved */
      // index *= elements_per_block;
      // Calculate the index starting points
      for (i=0; i<=stride_levels; i++) {
        idx[i] = index%arg.dims[i];
        index = (index-idx[i])/arg.dims[i];
      }
      /* Calculate the block offset for this thread */
      src_block_offset = bytes_per_thread*idx[0];
      dst_block_offset = bytes_per_thread*idx[0];
      for (i=0; i<stride_levels; i++) {
        src_block_offset += arg.src_strides[i]*idx[i+1];
        dst_block_offset += arg.dst_strides[i]*idx[i+1];
      }

      /* Start copying element by element
         TODO: Make it sure that it is continuous and replace the loop
         with a single memcpy */
      src = (void*)((char*)arg.src + src_block_offset);
      dst = (void*)((char*)arg.dst + dst_block_offset);
      op = arg.op;
      if (op == COMEX_ACC_INT) {
        int a = *((int*)src);
        int scale = *((int*)arg.scale);
        *((int*)dst) += a*scale;
      } else if (op == COMEX_ACC_LNG) {
        long a = *((long*)src);
        long scale = *((long*)arg.scale);
        *((long*)dst) += a*scale;
      } else if (op == COMEX_ACC_FLT) {
        float a = *((float*)src);
        float scale = *((float*)arg.scale);
        *((float*)dst) += a*scale;
      } else if (op == COMEX_ACC_DBL) {
        double a = *((double*)src);
        double scale = *((double*)arg.scale);
        *((double*)dst) += a*scale;
      } else if (op == COMEX_ACC_CPL) {
        float ar = *((float*)src);
        float ai = *(((float*)src)+1);
        float scaler = *((float*)arg.scale);
        float scalei = *(((float*)arg.scale)+1);
        *((float*)dst) += ar*scaler-ai*scalei;
        *(((float*)dst)+1) += ar*scalei+ai*scaler;
      } else if (op == COMEX_ACC_DCP) {
        double ar = *((double*)src);
        double ai = *(((double*)src)+1);
        double scaler = *((double*)arg.scale);
        double scalei = *(((double*)arg.scale)+1);
        *((double*)dst) += ar*scaler-ai*scalei;
        *(((double*)dst)+1) += ar*scalei+ai*scaler;
      }
      /* Synchronize the threads before returning  */
      sycl::group_barrier(item.get_group());
    });



  /*
  cudaDeviceSynchoronize();
  */
  if (src_on_host) {
    cudaHostUnregister(src);
  }
  if (dst_on_host) {
    cudaHostUnregister(dst);
  }
}
//};
