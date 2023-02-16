//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#pragma once

struct GpuTimer
{
    using U64 = unsigned long long;
#ifdef USE_SIMPLE_GPU_TIMER
    U64 t_start;
    U64 t_stop;
    U64 c_start;
    U64 c_stop;
#endif

    __device__ void Start()
    {
#ifdef USE_SIMPLE_GPU_TIMER
        U64 tmp;

        asm("mov.u64 %0, %%globaltimer;" : "=l"(tmp));
        t_start = tmp;
        asm("mov.u64 %0, %%clock64;" : "=l"(tmp));
        c_start = tmp;
#endif
    }

    __device__ void Stop()
    {
#ifdef USE_SIMPLE_GPU_TIMER
        U64 tmp;

        asm("mov.u64 %0, %%globaltimer;" : "=l"(tmp));
        t_stop = tmp;
        asm("mov.u64 %0, %%clock64;" : "=l"(tmp));
        c_stop = tmp;
#endif
    }

    __device__ void DisplayResult(const char* note)
    {
#ifdef USE_SIMPLE_GPU_TIMER
        U64 clock_diff = c_stop - c_start;

        U64 elapsed = t_stop - t_start;
        float ms = static_cast<float>(elapsed) / 576000.0f;

        printf("%s - %llu - %fms\n", note, clock_diff, ms);
#endif
    }
};
