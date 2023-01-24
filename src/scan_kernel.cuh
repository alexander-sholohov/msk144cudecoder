//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#include "gpu_timer.h"

struct BestCandidateInGroup
{
    unsigned pos;
    float xb;

    __device__ void reset()
    {
        pos = 0;
        xb = 0.0f;
    }
};

struct FindMaxInterWarpItem
{
    float xb;
    unsigned pos;
};

__global__ void scan_kernel(MSK144SearchContext ctx, const Complex* __restrict__  cdat)
{
    GpuTimer t1;
    GpuTimer t2;
    GpuTimer t_total;

    __shared__ Complex cdat2[Num6x864];
    __shared__ BestCandidateInGroup best_candidates[NumCandidatesPerPattern]; // 8 or 16 are ok
    
    constexpr unsigned NumWarpsInSlice = NumScanThreads / WarpSize;
    __shared__ FindMaxInterWarpItem best_xb_for_reduction[NumWarpsInSlice];

    static_assert(sizeof(cdat2) + sizeof(best_xb_for_reduction) + sizeof(best_candidates) <= 48 * 1024, "Not enough of shared memory");

    t_total.Start();
    t1.Start();

    // Get frequency to shift on.
    const float f0 = -1 * ctx.getFrequencyForThread(blockIdx.x);

    // Decimation in parallel using N threads.
    {
        const float twopi = 2.0f * CUDART_PI_F;
        constexpr unsigned num_slices = Num6x864 / NumScanThreads; // 5184 / 256 = 20,
        for(unsigned slice_no = 0; slice_no < num_slices; slice_no++)
        {
            const unsigned long_idx = slice_no * NumScanThreads + threadIdx.x;
            const float phi = static_cast<float>(long_idx) * twopi * f0 / SampleRate;
            Complex w = smath::make_complex_from_phi(phi);
            cdat2[long_idx] = w * cdat[long_idx];
        }

        // fill rest 5184 - 20 * 256 = 64
        constexpr unsigned start_of_rest = NumScanThreads * num_slices;
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

    t1.Stop();
    t2.Start();

    for(unsigned pattern_idx = 0; pattern_idx < ctx.scanDepth(); pattern_idx++)
    {
        if(threadIdx.x < NumCandidatesPerPattern)
        {
            best_candidates[threadIdx.x].reset();
        }
        __syncthreads();

        // Do it in parallel using N threads.
        constexpr unsigned num_slices = (Num6x864 % NumScanThreads == 0) ? (Num6x864 / NumScanThreads) : (Num6x864 / NumScanThreads + 1); // 5184 / 256 + 1 = 21,

        for(unsigned slice_no = 0; slice_no < num_slices; slice_no++)
        {
            const unsigned base = slice_no * NumScanThreads + threadIdx.x; // !

            Complex s = Complex(0.0f, 0.0f);

            // #pragma unroll
            for(unsigned idx = 0; idx < Num42; idx++)
            {
                const unsigned long_idx = base + idx;

                Complex y = Complex(0.0f, 0.0f);
                // AVG. average few blocks
                // #pragma unroll 1
                for(unsigned mask_idx = 0; mask_idx < NumPatternBitsToScan; mask_idx++)
                {
                    const int b = _gpattern[pattern_idx].mask[mask_idx];
                    if(b)
                    {
                        unsigned idx_a = (long_idx + Num864 * mask_idx);
                        if(idx_a >= Num6x864)
                        {
                            idx_a -= Num6x864;
                        }
                        y += cdat2[idx_a];

                        unsigned idx_b = (long_idx + Num864 * mask_idx + SecondSyncBase);
                        if(idx_b >= Num6x864)
                        {
                            idx_b -= Num6x864;
                        }
                        y += cdat2[idx_b];
                    }
                }
                s += conj(y) * _cb42[idx];
            }

            float xb_best = abs(s); //
            unsigned pos_best = base;
            // printf("threadIdx=%d base=%d xb_best=%f\n", threadIdx.x, base, xb_best);

#if 0
            if(threadIdx.x < 32)
            {
                printf("thr=%d xb=%f pos_best=%d\n", threadIdx.x, xb_best, pos_best);
            }
            __syncthreads();
#endif

            const unsigned mask = 0xffffffff;

            static_assert(NumScanThreads >= 32, "<32 threads not supported because we need full warp.");

            // Find max xb and correlated pos within EACH WARP using shfl functions reduction.

            {
                float xb_other = __shfl_down_sync(mask, xb_best, 1); // [0,2,4,6,8..30]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 1);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }

            {
                float xb_other = __shfl_down_sync(mask, xb_best, 2); // [0,4,8, ... 28]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 2);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }

            {
                float xb_other = __shfl_down_sync(mask, xb_best, 4); // [0,8,16,24]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 4);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }

            {
                float xb_other = __shfl_down_sync(mask, xb_best, 8); // [0,16]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 8);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }

            {
                float xb_other = __shfl_down_sync(mask, xb_best, 16); // [0]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 16);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }

            __syncthreads(); //

            // only first thread in each warp writes to shared memory
            if(threadIdx.x % WarpSize == 0)
            {
                const unsigned warp_no = threadIdx.x / WarpSize;
                FindMaxInterWarpItem& interwarp_item = best_xb_for_reduction[warp_no];
                interwarp_item.xb = xb_best;
                interwarp_item.pos = pos_best;
            }
            __syncthreads();

            const unsigned warp_lane = threadIdx.x % NumWarpsInSlice;
            const FindMaxInterWarpItem& loaded_item = best_xb_for_reduction[warp_lane]; // take value from shared memory to do reduction again within Warp.

            xb_best = loaded_item.xb;
            pos_best = loaded_item.pos;

#if NUM_SCAN_THREADS >= 64
            {
                float xb_other = __shfl_down_sync(mask, xb_best, 1); // [0,2,4,6,8..30]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 1);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }
#endif

#if NUM_SCAN_THREADS >= 128
            {
                float xb_other = __shfl_down_sync(mask, xb_best, 2); // [0,4,8, ... 28]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 2);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }
#endif

#if NUM_SCAN_THREADS >= 256
            {
                float xb_other = __shfl_down_sync(mask, xb_best, 4); // [0,8,16,24]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 4);
                if(xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }
#endif

#if NUM_SCAN_THREADS >= 512
            {
                float xb_other = __shfl_down_sync(mask, xb_best, 8); // [0,16]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 8);
                if (xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }
#endif

#if NUM_SCAN_THREADS >= 1024
            {
                float xb_other = __shfl_down_sync(mask, xb_best, 16); // [0]
                unsigned pos_other = __shfl_down_sync(mask, pos_best, 16);
                if (xb_other > xb_best)
                {
                    xb_best = xb_other;
                    pos_best = pos_other;
                }
            }
#endif
            static_assert(NumScanThreads <= 1024, "support of 1024+ threads is not implemented yet.");
            // TODO: implement extra reduction level for 1024+ threads. Also best candidate selection should be reviewed.

            // at now xb_best, pos_best - the values we needed
            // Find worse item in best_candidates to replace by just found best item
            static_assert(NumCandidatesPerPattern <= 32, "NumCandidates must be 32 max");

            unsigned candidate_idx = threadIdx.x % NumCandidatesPerPattern;
            float stored_xb = best_candidates[candidate_idx].xb;
            __syncthreads();

            {
                float other_xb = __shfl_down_sync(mask, stored_xb, 1); // [0 2 4 6 8 10 12 14 16 18]
                unsigned other_candidate_idx = __shfl_down_sync(mask, candidate_idx, 1);
                if(other_xb < stored_xb)
                {
                    stored_xb = other_xb;
                    candidate_idx = other_candidate_idx;
                }
            }

            {
                float other_xb = __shfl_down_sync(mask, stored_xb, 2); // [0  4  8  12  16    ]
                unsigned other_candidate_idx = __shfl_down_sync(mask, candidate_idx, 2);
                if(other_xb < stored_xb)
                {
                    stored_xb = other_xb;
                    candidate_idx = other_candidate_idx;
                }
            }

#if NUM_CANDIDATES_PER_PATTERN >= 8
            {
                float other_xb = __shfl_down_sync(mask, stored_xb, 4); // [0  8  16      ]
                unsigned other_candidate_idx = __shfl_down_sync(mask, candidate_idx, 4);
                if(other_xb < stored_xb)
                {
                    stored_xb = other_xb;
                    candidate_idx = other_candidate_idx;
                }
            }
#endif

#if NUM_CANDIDATES_PER_PATTERN >= 16
            {
                float other_xb = __shfl_down_sync(mask, stored_xb, 8); // [0  16     ]
                unsigned other_candidate_idx = __shfl_down_sync(mask, candidate_idx, 8);
                if(other_xb < stored_xb)
                {
                    stored_xb = other_xb;
                    candidate_idx = other_candidate_idx;
                }
            }
#endif

#if NUM_CANDIDATES_PER_PATTERN == 32
            {
                float other_xb = __shfl_down_sync(mask, stored_xb, 16); // [0       ]
                unsigned other_candidate_idx = __shfl_down_sync(mask, candidate_idx, 16);
                if(other_xb < stored_xb)
                {
                    stored_xb = other_xb;
                    candidate_idx = other_candidate_idx;
                }
            }
#endif

            // only first thread handles best candidate for groups
            if(threadIdx.x == 0)
            {
                // for(int i=0;i<NumCandidates;i++) { printf("xb[%d]=%f\n", i, best_candidates[i].xb_w_weight); }
                // printf("worse idx=%d, xb=%f\n", candidate_idx, stored_xb);
                // printf("xb_best=%f xb_w_weight=%f\n", xb_best, xb_w_weight);
                auto& worse_candidate_from_best = best_candidates[candidate_idx];

                // replace only if just found candidate is better that stored in array
                if(xb_best > worse_candidate_from_best.xb)
                {
                    worse_candidate_from_best.xb = xb_best;
                    worse_candidate_from_best.pos = pos_best;
                    // worse_candidate_from_best.pattern_idx = pattern_idx;
                    // worse_candidate_from_best.weight = weight;
                }
            }
            __syncthreads();

        } // slice_no loop

        // save best candidates to context
        if(threadIdx.x < NumCandidatesPerPattern)
        {
            unsigned candidate_idx = threadIdx.x;
            auto& candidate = best_candidates[candidate_idx];
            ctx.resultKeeper().put_candidate(blockIdx.x, pattern_idx, candidate_idx, candidate.pos, candidate.xb, f0, _gpattern[pattern_idx].num_avg);
            // printf("put candidate thr=%d pos=%d xb=%f\n", threadIdx.x, candidate.pos, candidate.xb);
        }

    } // pattern_idx loop

    // now we have filled best_item_so_far
    //

    t2.Stop();
    t_total.Stop();

#if 0
    if(threadIdx.x == 0)
    {
        t1.DisplayResult("timing t1");
        
        t2.DisplayResult("timing t2");
        //t2_1.DisplayResult("  timing t2_1");
        //t2_2.DisplayResult("  timing t2_2");
        //t2_3.DisplayResult("timing t2_3");
        //t2_4.DisplayResult("timing t2_4");
        //t2_5.DisplayResult("timing t2_5");
        
        t3.DisplayResult("timing t3");
        //t4.DisplayResult("timing t4");
        t_total.DisplayResult("timing t_total");
    }
#endif

    // All done. Host should call Result Keeper for results gathered from all blocks.
}
