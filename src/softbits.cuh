//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//


__global__ void softbits_kernel(MSK144SearchContext ctx, const Complex* __restrict__  cdat)
{
    static_assert(NumSoftbitsThreads >= 144, "Must be 144+ threads");

    __shared__ Complex cdat2[Num6x864];
    __shared__ Complex cdat3[Num864];
    __shared__ Complex summ_reduction[Num42];

    // When cdat3 calculated cdat2 is no more necessary and we can reuse it for softbits
    float* softbits = reinterpret_cast<float*>(cdat2);

    // reconstruct pattern_idx and candidate_num from block number dimension - y
    unsigned pattern_idx = blockIdx.y / NumCandidatesPerPattern;
    unsigned candidate_num = blockIdx.y % NumCandidatesPerPattern;

    // Get frequncy to shift on.
    const float f0 = -1 * ctx.getFrequencyForThread(blockIdx.x);
    Complex cb42_thr = _cb42[threadIdx.x % Num42];

    // Decimation in parallel using N threads.
    {
        const float twopi = 2.0f * CUDART_PI_F;
        constexpr unsigned num_slices = Num6x864 / NumSoftbitsThreads; // 5184 / 256 = 20,
        for(unsigned slice_no = 0; slice_no < num_slices; slice_no++)
        {
            const unsigned long_idx = slice_no * NumSoftbitsThreads + threadIdx.x;
            const float phi = static_cast<float>(long_idx) * twopi * f0 / SampleRate;
            Complex w = smath::make_complex_from_phi(phi);
            cdat2[long_idx] = w * cdat[long_idx];
        }

        // fill rest 5184 - 20 * 256 = 64
        constexpr unsigned start_of_rest = NumSoftbitsThreads * num_slices;
        constexpr unsigned rest = Num6x864 - start_of_rest;
        if(threadIdx.x < rest)
        {
            const unsigned long_idx = start_of_rest + threadIdx.x;
            const float phi = static_cast<float>(long_idx) * twopi * f0 / SampleRate;
            Complex w = smath::make_complex_from_phi(phi);
            cdat2[long_idx] = w * cdat[long_idx];
        }
        __syncthreads();

    } // end of scope

    const unsigned pos = ctx.resultKeeper().get_pos_for_candidate(blockIdx.x, pattern_idx, candidate_num);

    // Copy to cdat3 using AVG and cdat2 processing rules.
    for(unsigned slice_no = 0; slice_no < Num864 / NumSoftbitsThreads + 1; slice_no++)
    {
        const unsigned long_idx = slice_no * NumSoftbitsThreads + threadIdx.x; // !

        // AVG. average few blocks
        Complex s = Complex(0.0f, 0.0f);
        for(unsigned mask_idx = 0; mask_idx < NumPatternBitsToScan; mask_idx++)
        {
            if(_gpattern[pattern_idx].mask[mask_idx])
            {
                unsigned long_cdat2_idx = (pos + long_idx + Num864 * mask_idx);
                if(long_cdat2_idx >= Num6x864)
                {
                    long_cdat2_idx -= Num6x864;
                }
                s += cdat2[long_cdat2_idx];
            }
        }

        if(long_idx < Num864)
        {
            cdat3[long_idx] = s;
        }
    }
    __syncthreads();

    // Estimate carrier phase.
    // Calculate phase error as product of all sync bits.
    // Only 84 threads are using (42 first sync block + 42 second sync block).
    {
        const unsigned tidx = threadIdx.x;
        const unsigned base = (tidx < Num42) ? FirstSyncBase : SecondSyncBase;
        const unsigned k = base + (tidx % Num42);

        Complex x = cdat3[k] * conj(cb42_thr);

        // summ reduction
        // reuse summ_reduction shared array again

        if(tidx < Num42)
        {
            summ_reduction[tidx] = x;
        }
        __syncthreads();

        if(tidx >= Num42 && tidx < Num42 * 2)
        {
            summ_reduction[tidx - Num42] += x;
        }
        __syncthreads();
        // at now we have 42 elements in shared array

        if(tidx < 10)
        {
            summ_reduction[tidx] += summ_reduction[32 + tidx];
        }
        __syncthreads();
        // at now we have 32 elements in shared array

        // summ reduction 32 -> 1
        for(int size = 16; size > 0; size /= 2)
        {
            if(tidx < size)
                summ_reduction[tidx] += summ_reduction[tidx + size];
            __syncthreads();
        }

        // we have result at element [0]

    } // end of scope

    // at now we have result in summ_reduction[0]

    // t2_2.Stop();

    // Compensate Phase Error Block
    const Complex s = summ_reduction[0];
    // calculate phase error
    const float phase0 = atan2(s.imag(), s.real());

    const Complex w = smath::make_complex_from_phi(phase0);
    const Complex cfac = conj(w);

    // printf("thr=%d cfac_re=%f cfac_im=%f\n", threadIdx.x, cfac.real(), cfac.imag());

    // Remove phase error - want constellation rotated so that sample points lie on I / Q axes
    // Process in loop using all threads
    for(unsigned slice_no = 0; slice_no < (Num864 / NumSoftbitsThreads) + 1; slice_no++)
    {
        unsigned long_idx = slice_no * NumSoftbitsThreads + threadIdx.x;
        if(long_idx < Num864)
        {
            cdat3[long_idx] = cdat3[long_idx] * cfac;
        }
    }
    __syncthreads();

    // only 144 threads do the job. Each thread calculates its own softbit.
    {
        const int pos_iq = threadIdx.x % 72;       // This is I and Q index. [0..71].
        const int iq_selection = threadIdx.x / 72; // [0-72) = 0(Q), [72-143) = 1(I)

        const int base1 = (iq_selection) ? 0 : (Num864 - 6);
        const int d = 12 * pos_iq;

        float sb = 0.0f; // initial value of softbit for each thread.
        for(int idx = 0; idx < 12; idx++)
        {
            const int k = (base1 + d + idx) % Num864; // navigate element with boundary control
            const Complex r = cdat3[k];
            const float v = (iq_selection) ? r.real() : r.imag(); // 1-I-real; 0-Q-imag.

            sb = sb + v * _gpp12[idx];
        }

        // store result to shared memory, but no more than necessary.
        if(threadIdx.x < NumberOfSoftBits)
        {
            softbits[pos_iq * 2 + iq_selection] = sb; // QIQI...
        }
        __syncthreads();
    } // end of scope

    // at now softbits buffer has 144 elements. QIQIQIQIQ.....

    if(threadIdx.x < 16)
    {
        // Calculate BadSync factor in parallel using 16 threads

        // use bits 2,1,0 as pointer to sync item
        const int bit_number = threadIdx.x & 0x7; // [0..7]
        // use bit 3 as sync group selection 0/1
        const int base = (threadIdx.x & 0x8) ? FirstHardbitsSyncBase : SecondHardbitsSyncBase; // two sync groups
        const unsigned mask = 0xffffffff;

        const float sb = softbits[base + bit_number];
        const int hardbit = (sb < 0.0f) ? -1 : 1;

        // multiply hardbits by expecting pattern
        int v = hardbit * _gs8[bit_number];

        // 32 threads in warp, so 32 values total, but only first 16 are interested.

        // at now threads [0..15] have necessary values
        v += __shfl_down_sync(mask, v, 1); // after this threads  0, 2, 4, 6, 8, 10, 12, 14  will have necessary values
        v += __shfl_down_sync(mask, v, 2); // after this threads  0,    4,    8,     12      will have necessary values
        v += __shfl_down_sync(mask, v, 4); // after this threads  0,          8              will have necessary values

        int mm = (8 - v) / 2;                // mm=0 - pattern match; 1 - maybe; 2+ - definitely no sync.
        int smm1 = __shfl_sync(mask, mm, 0); // take value from thread 0 (first sync block)
        int smm2 = __shfl_sync(mask, mm, 8); // take value from thread 8 (second sync block)

        int nbadsync = smm1 + smm2; // 0-excellent, 1-maybe, 2+ - definitely no sync.

        // only the first thread will do the job of storing result.
        if(threadIdx.x == 0)
        {
            ctx.resultKeeper().put_softbits(blockIdx.x, pattern_idx, candidate_num, nbadsync, softbits);
        }
    }

    __syncthreads();
}
