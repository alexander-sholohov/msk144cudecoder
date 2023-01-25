//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

#include "pattern_item.h"
#include "result_keeper.cuh"
#include "ldpc_context.cuh"

#include <math_constants.h>

__constant__ Complex _cb42[42];
__constant__ int _gs8[8];
__constant__ float _gpp12[12];
__constant__ PatternItem _gpattern[ScanDepthMax];

class MSK144SearchContext
{
public:
    MSK144SearchContext(float center_freq, float search_width, float search_step, int scan_depth, int nbadsync_threshold)
        : _shared_count(new int(1))
        , _scan_depth(scan_depth)
        , _nbadsync_threshold(nbadsync_threshold)
    {
        // validate scan_depth
        // clang-format off
        if (_scan_depth < 1) { _scan_depth = 1; }
        if (_scan_depth > ScanDepthMax) { _scan_depth = ScanDepthMax; }
        if (NumPatternBitsToScan < FixedNumBitsInPattern && _scan_depth > NumPatternBitsToScan) { _scan_depth = NumPatternBitsToScan; }
        // clang-format on

        init(center_freq, search_width, search_step);
        _result_keeper.init(_number_of_blocks, _scan_depth, _nbadsync_threshold);
        _ldpc_context.init();
    }

    MSK144SearchContext(const MSK144SearchContext& other)
        : _shared_count(other._shared_count)
        , _if1(other._if1)
        , _step(other._step)
        , _number_of_blocks(other._number_of_blocks)
        , _number_of_threads(other._number_of_threads)
        , _center_freq(other._center_freq)
        , _scan_depth(other._scan_depth)
        , _nbadsync_threshold(other._nbadsync_threshold)
        , _result_keeper(other._result_keeper)
        , _ldpc_context(other._ldpc_context)
    {
        ++*_shared_count;
    }

    MSK144SearchContext& operator=(const MSK144SearchContext& other)
    {
        ++*other._shared_count;
        --*_shared_count;
        if(*_shared_count == 0)
        {
            delete _shared_count;
            _shared_count = nullptr;

            deinit();
            // std::cout << "in operator = in delete" << std::endl;
        }

        _if1 = other._if1;
        _step = other._step;
        _number_of_blocks = other._number_of_blocks;
        _number_of_threads = other._number_of_threads;
        _center_freq = other._center_freq;
        _scan_depth = other._scan_depth;
        _nbadsync_threshold = other._nbadsync_threshold;
        _result_keeper = other._result_keeper;
        _ldpc_context = other._ldpc_context;

        return *this;
    }

    ~MSK144SearchContext()
    {
        --*_shared_count;
        if(*_shared_count == 0)
        {
            delete _shared_count;
            _shared_count = nullptr;
            // std::cout << "in destructor real delete " << std::endl;

            deinit();
        }
    }

private:
    __host__ void init(float center_freq, float search_width, float search_step)
    {
        assert(search_step > 0);

        _center_freq = center_freq;
        _step = search_step;

        float half_len_in_hz = search_width / 2;           // 100/2 = 50.0
        float half_len_cnt = half_len_in_hz / search_step; // 50.0 / 2 = 25
        int half_len = static_cast<int>(half_len_cnt);

        _number_of_blocks = half_len * 2 + 1; // 25 * 2 + 1 = 51
        _if1 = -1 * half_len * search_step;   // -25 * 2 = -50
        _number_of_threads = NumScanThreads;  // constant = 16

        make_msk_sync42();

        make_pattern();
    }

    void deinit()
    {
        // std::cout << "in MSK144SearchContext::deinit\n";
        _result_keeper.deinit();
        _ldpc_context.deinit();
    }

public:
    __host__ dim3 getBlocks() const { return dim3(_number_of_blocks); }

    __host__ dim3 getThreads() const { return dim3(_number_of_threads); }

    __host__ dim3 getSoftBitsBlocks() const { return _result_keeper.getSoftBitsBlocks(); }

    __host__ dim3 getSoftBitsThreads() const { return _result_keeper.getSoftBitsThreads(); }

    __host__ float leftBound() const { return _center_freq + _if1; }

    __host__ float rightBound() const { return _center_freq + _if1 + (_number_of_blocks - 1) * _step; }

    __device__ float getFrequencyForThread(unsigned blkIdx) const { return _center_freq + _if1 + static_cast<int>(blkIdx) * _step; }

    __host__ static void fill_pp_float(float* pp)
    {
        const float pi = CUDART_PI_F;
        for(size_t i = 0; i < 12; i++)
        {
            float angle = i * pi / 12.0f;
            pp[i] = sinf(angle);
        }
    }

    __host__ static void fill_s8(int* s8)
    {
        int s8_org[] = {0, 1, 1, 1, 0, 0, 1, 0}; // msk144 sync sequence
        for(size_t i = 0; i < 8; i++)
        {
            s8[i] = 2 * s8_org[i] - 1; // transform to +-1
        }
    }

    __device__ __host__ ResultKeeper& resultKeeper() { return _result_keeper; }

    __device__ __host__ LDPCContext const& ldpcContext() const { return _ldpc_context; }

    __device__ __host__ int scanDepth() const { return _scan_depth; }

    __host__ int getNBadSyncThreshold() const { return _nbadsync_threshold; }

private:
    int* _shared_count;
    float _if1;
    float _step;
    int _number_of_blocks;
    int _number_of_threads;
    float _center_freq;
    int _scan_depth;
    int _nbadsync_threshold;
    ResultKeeper _result_keeper;
    LDPCContext _ldpc_context;

    __host__ void make_msk_sync42()
    {
        int s8[8]; // = { -1, 1, 1, 1, -1, -1, 1, -1 };
        fill_s8(s8);

        float pp[12];
        fill_pp_float(pp);

        float cbi[42];
        float cbq[42];

        // clang-format off
        for (size_t i = 0; i < 6;  i++) { cbq[ 0 + i] = pp[6 + i] * s8[0]; }
        for (size_t i = 0; i < 12; i++) { cbq[ 6 + i] = pp[i] * s8[2]; }
        for (size_t i = 0; i < 12; i++) { cbq[18 + i] = pp[i] * s8[4]; }
        for (size_t i = 0; i < 12; i++) { cbq[30 + i] = pp[i] * s8[6]; }

        for (size_t i = 0; i < 12; i++) { cbi[ 0 + i] = pp[i] * s8[1]; }
        for (size_t i = 0; i < 12; i++) { cbi[12 + i] = pp[i] * s8[3]; }
        for (size_t i = 0; i < 12; i++) { cbi[24 + i] = pp[i] * s8[5]; }
        for (size_t i = 0; i < 6;  i++) { cbi[36 + i] = pp[i] * s8[7]; }
        // clang-format on

        {
            thrust::host_vector<Complex> tmp(42);
            for(size_t i = 0; i < 42; i++)
            {
                tmp[i] = Complex(cbi[i], cbq[i]);
            }
            auto rc = cudaMemcpyToSymbol(_cb42, &tmp[0], sizeof(Complex) * 42);
            if(rc != cudaSuccess)
            {
                throw std::runtime_error("MemcpyToSymbol error.");
            }
        }

        {
            auto rc = cudaMemcpyToSymbol(_gs8, &s8[0], sizeof(int) * 8);
            if(rc != cudaSuccess)
            {
                throw std::runtime_error("MemcpyToSymbol error.");
            }
        }

        {
            auto rc = cudaMemcpyToSymbol(_gpp12, &pp[0], sizeof(float) * 12);
            if(rc != cudaSuccess)
            {
                throw std::runtime_error("MemcpyToSymbol error.");
            }
        }
    }

    __host__ void make_pattern()
    {
        PatternItem arr[] = {
          PatternItem(1, 0, 0, 0, 0, 0), // 1
          PatternItem(1, 1, 0, 0, 0, 0), // 2
          PatternItem(1, 1, 1, 0, 0, 0), // 3
          PatternItem(1, 1, 1, 1, 0, 0), // 4
          PatternItem(1, 1, 1, 1, 1, 0), // 5
          PatternItem(1, 1, 1, 1, 1, 1), // 6
          PatternItem(1, 0, 0, 1, 0, 0), // 7
          PatternItem(1, 0, 0, 1, 1, 0), // 8
        };

        static_assert(sizeof(arr) / sizeof(arr[0]) == ScanDepthMax);

        auto rc = cudaMemcpyToSymbol(_gpattern, &arr[0], sizeof(arr));
        if(rc != cudaSuccess)
        {
            throw std::runtime_error("MemcpyToSymbol error1.");
        }

        // for(int idx=0; idx < ScanDepthMax; idx++)
        //{
        //     auto rc = cudaMemcpyToSymbol(_gpattern, &arr[idx], sizeof(PatternItem), idx * sizeof(PatternItem));
        //     if (rc != cudaSuccess) { throw std::runtime_error("MemcpyToSymbol error1."); }
        // }
    }
};
