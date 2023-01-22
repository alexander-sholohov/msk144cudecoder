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
        float softbits_wo_sync[NumberOfSoftBitsWithoutSync];
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

    __device__ void put_softbits(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num, int nbadsync, const float* softbits_wo_sync)
    {
        ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);

        item.nbadsync = nbadsync;

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
