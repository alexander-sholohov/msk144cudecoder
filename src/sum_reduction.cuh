#pragma once

//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//


// This function gathers values from all threads and calculates summ of them. 
// It supports any number of threads but not more than 1024 (32 * 32 = 1024) and no less than 32.
// Supported types: float, int.
// shared_mem: array of 32 elements.
template<typename T>
__device__ T sum_reduction_two_cycles(T value, T* shared_mem)
{
    shared_mem[threadIdx.x % WarpSize] = 0; // initialize shared memory (this why we need 32+ threads)
    __syncthreads();

    const unsigned mask = 0xffffffff;
    // WARP-based sum reduction
    value += __shfl_down_sync(mask, value, 1); // after this threads  0, 2, 4, 6, 8, 10, 12, 14  will have necessary values
    value += __shfl_down_sync(mask, value, 2); // 0,4,8 ...
    value += __shfl_down_sync(mask, value, 4); // 0,8,16,24 ...
    value += __shfl_down_sync(mask, value, 8); // 0,16 ...
    value += __shfl_down_sync(mask, value, 16); // 0

    T v = __shfl_sync(mask, value, 0); // take value from thread 0

    shared_mem[threadIdx.x / WarpSize] = v; // each warp stores value to its cell
    __syncthreads();

    value = shared_mem[threadIdx.x % WarpSize]; // each thread within warp loads value

    // make reduction again
    value += __shfl_down_sync(mask, value, 1); // after this threads  0, 2, 4, 6, 8, 10, 12, 14  will have necessary values
    value += __shfl_down_sync(mask, value, 2); // 0,4,8 ...
    value += __shfl_down_sync(mask, value, 4); // 0,8,16,24 ...
    value += __shfl_down_sync(mask, value, 8); // 0,16 ...
    value += __shfl_down_sync(mask, value, 16); // 0

    v = __shfl_sync(mask, value, 0);  // take value from thread 0

    return v;
}
