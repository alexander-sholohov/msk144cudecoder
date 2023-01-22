//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

class ResultKeeper
{
public:
    struct ResultItem
    {
        unsigned block_idx;
        unsigned pattern_idx;
        unsigned pos;
        float f0;
        int nbadsync;
        float xb;
        int num_avg;
        float softbits[NumberOfSoftBits];
        float softbits_wo_sync[128];
        //
        bool is_message_present;
        int ldpc_num_iterations;
        int ldpc_num_hard_errors;
        char message[NumberOfMessageBits];
    };

    ResultKeeper() = default;
    ResultKeeper(const ResultKeeper&) = default;

    __host__ void init(unsigned num_blocks, int scan_depth)
    {
        _scan_depth = scan_depth;
        _num_blocks = num_blocks;

        const unsigned total_items = num_blocks * _scan_depth * NumCandidatesPerPattern;
        _total_items = total_items;

        _result_items = thrust::device_malloc<ResultItem>(total_items);
    }

    __host__ void deinit()
    {
        thrust::device_free(_result_items);
    }

    __host__ void clear_result()
    {
        if(cudaSuccess != cudaMemset(thrust::raw_pointer_cast(&_result_items[0]), 0, sizeof(ResultItem) * _total_items))
        {
            throw std::runtime_error("cudaMemset error.");
        }
    }

    __device__ ResultItem& get_result_item_by_block_coordinates(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num) const
    {
        unsigned k = blk_id * _scan_depth * NumCandidatesPerPattern + pattern_idx * NumCandidatesPerPattern + candidate_num;

        ResultItem* items_buf = thrust::raw_pointer_cast(_result_items);
        return items_buf[k];
    }

    __device__ void put_candidate(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num, unsigned pos, float xb, float f0, int num_avg)
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);

        item.block_idx = blk_id;
        item.pattern_idx = pattern_idx;
        item.pos = pos;
        item.xb = xb;
        item.f0 = f0;
        item.num_avg = num_avg;
    }

    __device__ void put_softbits(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num, int nbadsync, const float* softbits)
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);

        item.nbadsync = nbadsync;

        for(unsigned idx = 0; idx < NumberOfSoftBits; idx++)
        {
            item.softbits[idx] = softbits[idx];
        }
    }

    __device__ void put_softbits_wo_sync(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num, const float* softbits_wo_sync)
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);

        for(unsigned idx = 0; idx < NumberOfSoftBitsWithoutSync; idx++)
        {
            item.softbits_wo_sync[idx] = softbits_wo_sync[idx];
        }
    }

    __device__ void put_ldpc_decode_result(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num, const char* message, int num_iterations, int num_hard_errors)
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);
        for(int idx = 0; idx < NumberOfMessageBits; idx++)
        {
            item.message[idx] = message[idx];
        }
        item.is_message_present = true;
        item.ldpc_num_iterations = num_iterations;
        item.ldpc_num_hard_errors = num_hard_errors;
    }

    __device__ const float* get_softbits(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num) const
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);
        return item.softbits_wo_sync;

        // pattern for 4 iteration
        //static float arr[128] = {-6.293818 ,-7.185833 ,-5.296424 ,-8.056597 ,8.702086 ,-5.622574 ,7.000746 ,9.924503 ,-8.784536 ,-9.247722 ,8.158647 ,9.445013 ,-6.845063 ,7.558348 ,-6.130745 ,5.499822 ,9.426908 ,5.107175 ,7.541466 ,8.242576 ,-7.707906 ,8.512366 ,-7.320877 ,-8.702724 ,5.652862 ,-8.128260 ,4.081129 ,-4.782642 ,-6.768012 ,7.145503 ,6.682837 ,-7.346786 ,-9.940599 ,-6.823922 ,5.428613 ,-8.840699 ,3.706920 ,-4.461401 ,-6.913119 ,5.683021 ,10.698535 ,6.585670 ,-4.084242 ,8.685247 ,-6.118265 ,-8.811988 ,7.527551 ,7.333613 ,5.867682 ,7.396698 ,-6.587626 ,-6.820182 ,2.445028 ,-7.701050 ,5.274485 ,7.124141 ,-6.125362 ,-2.238184 ,-3.302717 ,4.309085 ,8.565947 ,0.641817 ,5.696017 ,1.268913 ,3.093736 ,-3.454431 ,-4.750222 ,4.114901 ,7.892020 ,3.472466 ,-0.947921 ,6.371738 ,0.156821 ,6.191186 ,-3.974459 ,-4.338089 ,4.297905 ,3.826895 ,-1.091831 ,2.052665 ,2.664372 ,-5.931645 ,-0.026093 ,-1.691547 ,-6.067714 ,0.926820 ,-1.595734 ,2.328002 ,5.633004 ,-1.405352 ,5.331262 ,-0.156908 ,0.726322 ,-6.339333 ,1.903530 ,2.878520 ,0.588436 ,4.451695 ,-2.755684 ,-2.372380 ,3.260587 ,2.593289 ,0.171194 ,1.092363 ,5.088827 ,1.631720 ,-2.385000 ,-2.163793 ,2.601409 ,2.720500 ,-2.364061 ,0.526213 ,-0.448253 ,4.579746 ,-1.468629 ,-2.442158 ,1.760642 ,1.276744 ,-0.134276 ,0.644546 ,0.650366 ,-0.594585 ,-3.474879 ,0.678906 ,-1.460457 ,2.735036 ,1.191745 ,2.774232 };

        // ideal pattern
        //static float arr[] = {-6.917636 ,-6.190248 ,-5.899355 ,-7.713686 ,7.738573 ,-5.987260 ,6.745874 ,9.518480 ,-8.424323 ,-8.786919 ,7.776994 ,8.973592 ,-6.050537 ,7.382261 ,-5.489295 ,5.137037 ,9.092748 ,4.583800 ,7.252029 ,7.744722 ,-7.149550 ,8.069152 ,-6.879371 ,-8.194984 ,5.382341 ,-7.611909 ,3.981714 ,-4.550130 ,-6.449366 ,6.823149 ,6.423380 ,-7.037288 ,-9.203778 ,-6.510252 ,5.690207 ,-8.134647 ,4.058394 ,-4.403567 ,-6.695155 ,5.676703 ,9.862463 ,6.490009 ,-4.755414 ,7.891111 ,-6.049268 ,-8.565968 ,7.512718 ,7.258935 ,5.809515 ,7.427251 ,-6.785971 ,-6.966565 ,3.874657 ,-6.928414 ,5.462711 ,7.218122 ,-6.587000 ,-3.601966 ,-3.729444 ,4.983152 ,8.010149 ,2.046047 ,4.915369 ,2.914101 ,3.471951 ,-4.096270 ,-5.494949 ,4.810356 ,7.346817 ,4.323187 ,-2.982732 ,5.928195 ,-1.943269 ,5.508858 ,-4.677507 ,-5.053520 ,5.033289 ,4.465194 ,-2.675816 ,2.729526 ,3.626355 ,-5.802946 ,1.575959 ,-2.429933 ,-6.131845 ,-1.171217 ,-2.370228 ,3.444505 ,5.689749 ,0.448310 ,5.017456 ,2.023158 ,1.708724 ,-6.117103 ,3.145067 ,3.849833 ,-1.420945 ,4.158947 ,-3.675761 ,-3.689566 ,4.058065 ,4.010004 ,-2.099950 ,2.140676 ,5.236421 ,2.721317 ,-3.622581 ,-3.458414 ,3.624581 ,3.578872 ,-3.594823 ,-1.391497 ,-1.611751 ,4.554566 ,-2.445459 ,-3.234372 ,3.188134 ,2.315187 ,-1.951785 ,1.277625 ,1.940769 ,-1.892364 ,-4.122101 ,-1.468534 ,-2.347014 ,3.083031 ,-0.416811 ,2.598789 };

        //return arr;
    }

    __device__ int get_nbadsync(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num) const
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);
        return item.nbadsync;
    }


    __device__ unsigned get_pos_for_candidate(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num) const
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);
        return item.pos;
    }

    __host__ ResultItem get_result_item_by_index(const unsigned idx) const
    {
        assert(idx < _total_items);

        ResultItem res;
        thrust::copy(_result_items + idx, _result_items + (idx + 1), &res);

        return res;
    }

    __host__ thrust::host_vector<ResultItem> get_all_results() const
    {
        thrust::host_vector<ResultItem> res(_total_items);
        thrust::copy(_result_items, _result_items + _total_items, &res[0]);

        return res;
    }


    __host__ dim3 getSoftBitsBlocks() const { return dim3(_num_blocks, _scan_depth * NumCandidatesPerPattern); }

    __host__ dim3 getSoftBitsThreads() const { return dim3(NumSoftbitsThreads); }


private:
    unsigned _num_blocks;
    unsigned _total_items;
    int _scan_depth;

    thrust::device_ptr<ResultItem> _result_items;
};
