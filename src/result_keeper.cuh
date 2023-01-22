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

    __host__ void init(unsigned num_blocks, int scan_depth, int nbadsync_threshold)
    {
        _scan_depth = scan_depth;
        _num_blocks = num_blocks;

        const unsigned total_items = num_blocks * _scan_depth * NumCandidatesPerPattern;
        _total_items = total_items;

        _nbadsync_threshold = nbadsync_threshold;

        _num_filtered_candidates = 0;

        _result_items = thrust::device_malloc<ResultItem>(total_items);
        _filtered_candidate_index = thrust::device_malloc<unsigned>(total_items);
    }

    __host__ void deinit()
    {
        thrust::device_free(_result_items);
    }

    __host__ void clear_result()
    {
        _num_filtered_candidates = 0;

        if(cudaSuccess != cudaMemset(thrust::raw_pointer_cast(&_result_items[0]), 0, sizeof(ResultItem) * _total_items))
        {
            throw std::runtime_error("cudaMemset error.");
        }
        if(cudaSuccess != cudaMemset(thrust::raw_pointer_cast(&_filtered_candidate_index[0]), 0, sizeof(unsigned) * _total_items))
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

    //__device__ const float* get_softbits(unsigned blk_id, unsigned pattern_idx, unsigned candidate_num) const
    //{
    //    ResultItem& item = get_result_item_by_block_coordinates(blk_id, pattern_idx, candidate_num);
    //    return item.softbits_wo_sync;
    //}

    __device__ const float* get_filtered_softbits(unsigned blk_id) const
    {
        const unsigned* index_buf = thrust::raw_pointer_cast(_filtered_candidate_index);
        const unsigned real_index = index_buf[blk_id];

        const ResultItem* items_buf = thrust::raw_pointer_cast(_result_items);
        return items_buf[real_index].softbits_wo_sync;
    }

    __device__ void put_ldpc_decode_result(unsigned blk_id, const char* message, int num_iterations, int num_hard_errors)
    {
        const unsigned* index_buf = thrust::raw_pointer_cast(_filtered_candidate_index);
        const unsigned real_index = index_buf[blk_id];

        ResultItem* items_buf = thrust::raw_pointer_cast(_result_items);

        ResultItem& item = items_buf[real_index];
        item.is_message_present = true;
        item.ldpc_num_iterations = num_iterations;
        item.ldpc_num_hard_errors = num_hard_errors;
        for(unsigned idx = 0; idx < NumberOfMessageBits; idx++)
        {
            item.message[idx] = message[idx];
        }
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

    __host__ void filter_candidates()
    {
        thrust::host_vector<ResultItem> res(_total_items);
        thrust::copy(_result_items, _result_items + _total_items, &res[0]);


        thrust::host_vector<unsigned> indexes;

        for(unsigned idx=0; idx < _total_items; idx++)
        {
            if(res[idx].nbadsync <= _nbadsync_threshold)
            {
                indexes.push_back(idx);
            }
        }
        _num_filtered_candidates = indexes.size();
        thrust::copy(indexes.begin(), indexes.end(), _filtered_candidate_index);
    }

    __host__ dim3 getSoftBitsBlocks() const { return dim3(_num_blocks, _scan_depth * NumCandidatesPerPattern); }

    __host__ dim3 getSoftBitsThreads() const { return dim3(NumSoftbitsThreads); }

    __host__ dim3 getFilteredCandidatesBlocks() const { return dim3(_num_filtered_candidates); }


private:
    unsigned _num_blocks;
    unsigned _total_items;
    int _scan_depth;
    int _nbadsync_threshold;
    int _num_filtered_candidates;

    thrust::device_ptr<ResultItem> _result_items;
    thrust::device_ptr<unsigned> _filtered_candidate_index;
};
