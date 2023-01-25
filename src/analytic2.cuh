//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#include "common.h"

constexpr float sc45 = 0.707106781f; // sin(45) = sqrt(2)/2 = 0.7071...

template<unsigned int NumSlices, unsigned NumAnalyticThreads>
__device__ void _frequency_shift_fs8_left(Complex* cdat)
{
    Complex wttt;
    switch(threadIdx.x & 7)
    {
    case 0:
        wttt = Complex(sc45, -sc45); // 315
        break;
    case 1:
        wttt = Complex(0.0f, -1.0f); // 270
        break;
    case 2:
        wttt = Complex(-sc45, -sc45); // 225
        break;
    case 3:
        wttt = Complex(-1.0f, 0.0f); // 180
        break;
    case 4:
        wttt = Complex(-sc45, sc45); // 135
        break;
    case 5:
        wttt = Complex(0.0f, 1.0f); // 90
        break;
    case 6:
        wttt = Complex(sc45, sc45); // 45
        break;
    case 7:
        wttt = Complex(1.0f, 0.0f); // 0
        break;
    }

    // Apply frequency shifting by Fs/8 (1500 Hz). <<<----
    for(unsigned slice_no = 0; slice_no < NumSlices; slice_no++)
    {
        const unsigned long_idx = slice_no * NumAnalyticThreads + threadIdx.x;
        cdat[long_idx] = cdat[long_idx] * wttt;
    }
    __syncthreads();
}

template<unsigned int NumSlices, unsigned NumAnalyticThreads>
__device__ void _frequency_shift_fs8_right(Complex* cdat)
{
    Complex wttt;
    switch(threadIdx.x & 7)
    {
    case 0:
        wttt = Complex(1.0f, 0.0f); // 0
        break;
    case 1:
        wttt = Complex(sc45, sc45); // 45
        break;
    case 2:
        wttt = Complex(0.0f, 1.0f); // 90
        break;
    case 3:
        wttt = Complex(-sc45, sc45); // 135
        break;
    case 4:
        wttt = Complex(-1.0f, 0.0f); // 180
        break;
    case 5:
        wttt = Complex(-sc45, -sc45); // 225
        break;
    case 6:
        wttt = Complex(0.0f, -1.0f); // 270
        break;
    case 7:
        wttt = Complex(sc45, -sc45); // 315
        break;
    }

    // Apply frequency shifting by Fs/8. ---->>>
    for(unsigned slice_no = 0; slice_no < NumSlices; slice_no++)
    {
        const unsigned long_idx = slice_no * NumAnalyticThreads + threadIdx.x;
        cdat[long_idx] = cdat[long_idx] * wttt;
    }
    __syncthreads();
}

template<unsigned TotalElements, unsigned NumSlices, unsigned NumAnalyticThreads>
__device__ void _copy_from_global_to_shared(const Complex* __restrict__ input, Complex* cdat)
{
    const Complex zero_c = Complex(0.0f, 0.0f);

    // fill head and tail
    if(threadIdx.x < 32)
    {
        cdat[threadIdx.x] = zero_c;
    }
    if(threadIdx.x < 32)
    {
        cdat[TotalElements - threadIdx.x - 1] = zero_c;
    }

    // copy from global memory to cdat
    for(unsigned slice_no = 0; slice_no < NumSlices - 2; slice_no++)
    {
        const unsigned long_idx = slice_no * NumAnalyticThreads + threadIdx.x;
        cdat[32 + long_idx] = input[long_idx];
    }
    __syncthreads();
}

template<unsigned TotalElements, unsigned NumSlices, unsigned NumAnalyticThreads>
__device__ void _lpf_convolution(Complex* cdat)
{
    //
    // This is how LPF coefficients was calculated:
    //
    // import scipy.signal as signal
    // numtaps = 15
    // signal.firwin(numtaps, 0.2, pass_zero=True, window='boxcar')
    // taps[abs(taps) <= 1e-4] = 0.
    // for x in taps: print(x)
    //
    // -0.04225694
    // -0.03046893
    // 0.
    // 0.04570339
    // 0.09859952
    // 0.14789927
    // 0.18281356
    // 0.19542026
    // 0.18281356
    // 0.14789927
    // 0.09859952
    // 0.04570339
    // 0.
    // - 0.03046893
    // - 0.04225694

    const float h1 = -0.04225694f;
    const float h2 = -0.03046893f;

    const float h4 = 0.04570339f;
    const float h5 = 0.09859952f;
    const float h6 = 0.14789927f;
    const float h7 = 0.18281356f;
    const float h8 = 0.19542026f;
    const float h9 = 0.18281356f;
    const float h10 = 0.14789927f;
    const float h11 = 0.09859952f;
    const float h12 = 0.04570339f;

    const float h14 = -0.03046893f;
    const float h15 = -0.04225694f;

#if 1
    // Apply Half-Band Low Pass Filter.
    for(int slice_no = 0; slice_no < NumSlices - 1; slice_no++)
    {
        const int long_idx = slice_no * NumAnalyticThreads + threadIdx.x;

        auto s = Complex(0.0f, 0.0f);

        s += h1 * cdat[long_idx + (16 - 1)];
        s += h2 * cdat[long_idx + (16 - 2)];

        s += h4 * cdat[long_idx + (16 - 4)];
        s += h5 * cdat[long_idx + (16 - 5)];
        s += h6 * cdat[long_idx + (16 - 6)];
        s += h7 * cdat[long_idx + (16 - 7)];
        s += h8 * cdat[long_idx + (16 - 8)];
        s += h9 * cdat[long_idx + (16 - 9)];
        s += h10 * cdat[long_idx + (16 - 10)];
        s += h11 * cdat[long_idx + (16 - 11)];
        s += h12 * cdat[long_idx + (16 - 12)];

        s += h14 * cdat[long_idx + (16 - 14)];
        s += h15 * cdat[long_idx + (16 - 15)];

        __syncthreads(); // Not a mistake. We need sync threads before write.
        cdat[long_idx] = s;
    }
    __syncthreads();
#endif

#if 1
    // Apply this filter again for reverse ordered samples.
    // This will give us zero-phase shifting result for all frequency range.
    // I'm not sure if we really need zero-phase shifting, but applying the same filter twice is also good.
    for(int slice_no = 0; slice_no < NumSlices - 1; slice_no++)
    {
        const int long_idx = TotalElements - slice_no * NumAnalyticThreads - threadIdx.x - 1;

        auto s = Complex(0.0f, 0.0f);

        s += h1 * cdat[long_idx - (16 - 1)];
        s += h2 * cdat[long_idx - (16 - 2)];

        s += h4 * cdat[long_idx - (16 - 4)];
        s += h5 * cdat[long_idx - (16 - 5)];
        s += h6 * cdat[long_idx - (16 - 6)];
        s += h7 * cdat[long_idx - (16 - 7)];
        s += h8 * cdat[long_idx - (16 - 8)];
        s += h9 * cdat[long_idx - (16 - 9)];
        s += h10 * cdat[long_idx - (16 - 10)];
        s += h11 * cdat[long_idx - (16 - 11)];
        s += h12 * cdat[long_idx - (16 - 12)];

        s += h14 * cdat[long_idx - (16 - 14)];
        s += h15 * cdat[long_idx - (16 - 15)];

        __syncthreads();
        cdat[long_idx] = s;
    }
    __syncthreads();
#endif
}

template<unsigned int NumSlices, unsigned NumAnalyticThreads>
__device__ void _backcopy_to_global_memory(const Complex* cdat, Complex* result)
{
    // Copy result from shared memory to global memory
    for(unsigned slice_no = 0; slice_no < NumSlices - 2; slice_no++)
    {
        const unsigned long_idx = slice_no * NumAnalyticThreads + threadIdx.x;
        result[long_idx] = cdat[32 + long_idx];
    }
}

template<unsigned int NumSamples, unsigned NumAnalyticThreads>
__global__ void apply_shift_filter_shift(const Complex* __restrict__ input, Complex* result)
{
    //
    // cdat plan: [32] + [NumSamples] + [32] = 32*2 + 5184 = 5216
    //
    constexpr int TotalElements = 32 * 2 + NumSamples; // 5248
    static_assert(NumSamples % NumAnalyticThreads == 0, "NumSamples % NumAnalyticThreads == 0");
    static_assert(TotalElements % NumAnalyticThreads == 0, "TotalElements % NumAnalyticThreads == 0");
    constexpr int NumSlices = TotalElements / NumAnalyticThreads;

    __shared__ Complex cdat[TotalElements];
    static_assert(sizeof(cdat) <= 48 * 1024, "Not enough of shared memory");

    // Do nothing and return if the kernel was called with wrong argument.
    if(NumAnalyticThreads != blockDim.x)
        return;

    _copy_from_global_to_shared<TotalElements, NumSlices, NumAnalyticThreads>(input, cdat);
    _frequency_shift_fs8_left<NumSlices, NumAnalyticThreads>(cdat);
    _lpf_convolution<TotalElements, NumSlices, NumAnalyticThreads>(cdat);
    _frequency_shift_fs8_right<NumSlices, NumAnalyticThreads>(cdat);
    _backcopy_to_global_memory<NumSlices, NumAnalyticThreads>(cdat, result);
}

template<unsigned int NumSamples, unsigned NumAnalyticThreads>
__global__ void apply_filter(const Complex* __restrict__ input, Complex* result)
{
    //
    // cdat plan: [32] + [NumSamples] + [32] = 32*2 + 5184 = 5216
    //
    constexpr int TotalElements = 32 * 2 + NumSamples; // 5248
    static_assert(NumSamples % NumAnalyticThreads == 0, "NumSamples % NumAnalyticThreads == 0");
    static_assert(TotalElements % NumAnalyticThreads == 0, "TotalElements % NumAnalyticThreads == 0");
    constexpr int NumSlices = TotalElements / NumAnalyticThreads;

    __shared__ Complex cdat[TotalElements];
    static_assert(sizeof(cdat) <= 48 * 1024, "Not enough of shared memory");

    // Do nothing and return if the kernel was called with wrong argument.
    if(NumAnalyticThreads != blockDim.x)
        return;

    _copy_from_global_to_shared<TotalElements, NumSlices, NumAnalyticThreads>(input, cdat);
    _lpf_convolution<TotalElements, NumSlices, NumAnalyticThreads>(cdat);
    _backcopy_to_global_memory<NumSlices, NumAnalyticThreads>(cdat, result);
}
