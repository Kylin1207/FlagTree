#include <math_constants.h>
__device__ auto SoftmaxKernel(__attribute__((address_space(3))) float* output_allocated, 
                            __attribute__((address_space(3))) float* output_aligned, 
                            const int64_t output_offsets,
                            const int64_t output_sizes1,
                            const int64_t output_stride1,
                            __attribute__((address_space(3))) float* input_allocated,
                            __attribute__((address_space(3))) float* input_aligned, 
                            const int64_t input_offsets,
                            const int64_t input_size1,
                            const int64_t input_stride1
                            ) {
    const int idx = threadIdx.x;
    const int bdimx=blockDim.x;
    const int gridx=gridDim.x;
    const int bldx=blockIdx.x;
    for (int i = idx; i < output_sizes1; i +=  bdimx) {
        output_aligned[i]=input_aligned[i];
    }

    __syncthreads();
    for (int arg = (output_sizes1)>>1; arg > 0; arg=(arg)>>1) {
        for (int i = idx; i+arg< output_sizes1; i += bdimx) {
            if(output_aligned[i]<output_aligned[i+arg])
                output_aligned[i]=output_aligned[i+arg];
        }
        __syncthreads();
    }
    float max_val=output_aligned[0];
    for (int i = idx; i < output_sizes1; i += bdimx) {
        float v=CUDART_L2E*(input_aligned[i]-max_val);
        float y;
        asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(v));
        output_aligned[i]=y;
    }
    __syncthreads();
    for (int arg = (output_sizes1)>>1; arg > 0; arg=(arg)>>1) {
        for (int i = idx; i+arg< output_sizes1; i +=  bdimx) {
            output_aligned[i]=output_aligned[i]+output_aligned[i+arg];
        }
        __syncthreads();
    }
    float inv = 1.0f / output_aligned[0];
    for (int i = idx; i < output_sizes1; i += bdimx) {
        float v=CUDART_L2E*(input_aligned[i]-max_val);
        float y;
        asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(v));
        output_aligned[i]=y * inv;
    }
    __syncthreads();
    struct {
        __attribute__((address_space(3))) float* allocated;
        __attribute__((address_space(3))) float* aligned;
        int64_t offsets;
        int64_t sizes1[1];
        int64_t stride1[1];
       
    }r{
        output_allocated,
        output_aligned, 
        output_offsets,
        output_sizes1,
        output_stride1,
    };
    return r;
}
