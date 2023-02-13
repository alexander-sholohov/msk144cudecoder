//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

__global__ void index_kernel(MSK144SearchContext ctx)
{
    const int nbadsync_threshold = ctx.resultKeeper().nbadsync_threshold();
    const int num_total_items = ctx.resultKeeper().num_total_items();

    int* indexes = ctx.resultKeeper().get_indexes_begin();
    const ResultKeeper::ResultItem* src = ctx.resultKeeper().get_result_items_begin();
    int total_num_filtered_results = 0;
    __shared__ int shared_flags[NumIndexThreads];

    const int num_iterations = num_total_items / NumIndexThreads;

    for(int iter = 0; iter < num_iterations; iter++)
    {
        const int base = iter * NumIndexThreads;
        if(src[base + threadIdx.x].nbadsync <= nbadsync_threshold)
        {
            shared_flags[threadIdx.x] = 1;
        }
        else
        {
            shared_flags[threadIdx.x] = 0;
        }

        __syncthreads();

        // only thread 0 constructs indexes in global memory
        if(threadIdx.x == 0)
        {
            for(int k = 0; k < NumIndexThreads; k++)
            {
                if(shared_flags[k])
                {
                    indexes[total_num_filtered_results] = base + k;
                    total_num_filtered_results++;
                }
            }
        }
        __syncthreads();
    }

    shared_flags[threadIdx.x] = 0;
    const int rest_items = num_total_items - num_iterations * NumIndexThreads;
    const int base = num_iterations * NumIndexThreads;

    if(threadIdx.x < rest_items)
    {
        if(src[base + threadIdx.x].nbadsync <= nbadsync_threshold)
        {
            shared_flags[threadIdx.x] = 1;
        }
    }
    __syncthreads();

    // only thread 0 constructs indexes in global memory
    if(threadIdx.x == 0)
    {
        for(int k = 0; k < rest_items; k++)
        {
            if(shared_flags[k])
            {
                indexes[total_num_filtered_results] = base + k;
                total_num_filtered_results++;
            }
        }

        // and store number of indexes
        *ctx.resultKeeper().p_number_of_sorted_candidates() = total_num_filtered_results;
    }
}
