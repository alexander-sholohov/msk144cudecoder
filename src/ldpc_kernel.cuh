//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#include "sum_reduction.cuh"

__device__ void convert_cw_to_bytes(const char* cw, unsigned char* byte_buf)
{
    // Convert 128 chars of single bits -> 16 * 8bits bytes.
    // We need exact 128 threads to do the job.

    const unsigned mask = 0xffffffff;

    int v = cw[threadIdx.x]; // 01 01 01 01 01 ...
    int other_v;
    other_v = __shfl_down_sync(mask, v, 1); // take value from neighbor thread. 
    v = (v << 1) | other_v;  // 03 xx 03 xx ...
    other_v = __shfl_down_sync(mask, v, 2);
    v = (v << 2) | other_v;  // 0f xx xx xx 0f xx xx xx 0f xx xx xx ...
    other_v = __shfl_down_sync(mask, v, 4);
    v = (v << 4) | other_v;  // ff xx xx xx xx xx xx xx ff xx xx .....

    // ok. 8 bits were gathered. Each warp contains 4 result bytes in threads: [0,8,16,24]. Number of warps = 4 (128/32).
    
    v = __shfl_sync(mask, v, threadIdx.x & 0x18); // Spread(duplicate) values from threads [0,8,16,24] across 8 neighbor right threads.
    byte_buf[threadIdx.x / 8] = v;
    __syncthreads();
}

__device__ uint16_t calc_crc13(const unsigned char* buf, const unsigned length, const uint16_t* crc_table)
{
    uint16_t remainder = 0;
    for(int i=0; i<length; i++)
    {
        const int index = (remainder >> (13 - 8)) & 0xff;
        remainder <<= 8;
        remainder |= buf[i];
        remainder ^= crc_table[index];
    }
    return remainder & 0x1fff;
}

__device__ bool check_crc(unsigned char* byte_buf, const uint16_t* crc_table)
{
    // We have 77 bits for message. This takes 9 full bytes ([0..8]) + 5 bits in next byte.
    // CRC starts from bit 78 and takes 13 bits.

    // Take crc.
    const uint32_t unaligned_crc_from_message = ((static_cast<uint32_t>(byte_buf[9]) & 0x7) << 16) | (static_cast<uint32_t>(byte_buf[10]) << 8) | (byte_buf[11] & 0xc0); // 3 + 8 + 2 = 13
    const uint16_t crc_from_message = unaligned_crc_from_message >> 6;

    // Eliminate crc from initial buffer.
    byte_buf[9] &= 0xf8;
    byte_buf[10] = 0;
    byte_buf[11] = 0;

    const uint16_t calculated_crc = calc_crc13(byte_buf, 12, crc_table);

    return crc_from_message == calculated_crc;
}

__device__ float platanh(float x)
{
    float isign = 1.0f;
    float z = x;
    if(x < 0.0)
    {
        isign = -1.0f;
        z = abs(x);
    }

    if(z <= 0.664f)
    {
        return x / 0.83f;
    }
    else if( z <= 0.9217f)
    {
        return isign*(z-0.4064f)/0.322f;
    }
    else if(z <= 0.9951f)
    {
        return isign*(z-0.8378f)/0.0524f;
    }
    else if(z <= 0.9998f)
    {
        return isign*(z-0.9914f)/0.0012f;
    }

    return isign*7.0f;
}

//
// The algorighm is taken from WSJT project. File name is bpdecode128_90.f90 . 
// A log-domain belief propagation decoder for the (128,90) LDPC code.
// It is adapted to be used by 128 parallel CUDA threads.
//
__global__ void ldpc_kernel(MSK144SearchContext ctx)
{
    __shared__ char mp[128][3][2];
    __shared__ bool is_full_row[38];
    __shared__ float toc[11][38];
    __shared__ float tov[3][128];
    __shared__ float zn[128];
    __shared__ char cw[128];
    __shared__ char chk_cw_toc[11][38];
    __shared__ int arr_reduction_helper[32];
    __shared__ bool message_found;
    __shared__ uint16_t crc_table[256];
    __shared__ unsigned char crc_buf[16];
    
    const float* softbits = ctx.resultKeeper().get_softbits_by_filtered_index(blockIdx.x);

    {
        // Copy CRC table from device memory to shared memory as it is faster.
        const uint16_t* mem_crc_table = ctx.ldpcContext().get_crc_table();
        crc_table[threadIdx.x * 2] = mem_crc_table[threadIdx.x * 2];
        crc_table[threadIdx.x * 2 + 1] = mem_crc_table[threadIdx.x * 2 + 1];
    }

    {
        // Fill reverse map in shared array. We use shared array as fast memory.
        const char* rev_map = ctx.ldpcContext().get_reverse_map();
        char* rev_map_dst = &mp[0][0][0];
        for(int i=0; i<6; i++)
        {
            const int long_idx = 6 * threadIdx.x + i; // 128 rows with 6 bytes in each row
            rev_map_dst[long_idx] = rev_map[long_idx];
        }
    }

    {
        // Init 'is_full_row' array.
        const bool* full_row_table = ctx.ldpcContext().get_is_full_row();
        if(threadIdx.x < 38)
        {
            is_full_row[threadIdx.x] = full_row_table[threadIdx.x];
        }
    }
    __syncthreads(); // 

    const unsigned thr_idx = threadIdx.x;

    // Initialize 'tov' array.
    for(size_t i=0; i < 3; i++) 
    { 
        tov[i][thr_idx] = 0.0f; 
    }

    if(thr_idx < 38)
    {
        // Init only last row in these two arrays. All other rows will be set in loops.
        toc[10][thr_idx] = 0.0f;
        chk_cw_toc[10][thr_idx] = 0;
    }

    __syncthreads();


    for(unsigned iter=0; iter < NumberOfLDPCIterations; iter++)
    {
        // Update bit log likelihood ratios (tov=0 in iteration 0).
        float sum = 0.0f;
        for(size_t k=0;k<3;k++) { sum += tov[k][thr_idx]; }
        zn[thr_idx] = softbits[thr_idx] + sum;
        cw[thr_idx] = (zn[thr_idx] > 0.0f)? 1 : 0;
        __syncthreads();

        for(int k=0; k< 3; k++)
        {
            chk_cw_toc[mp[thr_idx][k][0]][mp[thr_idx][k][1]] = cw[thr_idx];
        }
        __syncthreads();

        // Check to see if we have a codeword (check before we do any iteration).
        int local_ncheck = 0;
        if(thr_idx < 38)
        {
            // Only first 38 threads calculates sum for correspondings columns.
            int sum = 0;
            for(size_t i=0; i < 11; i++)
            {
                sum += chk_cw_toc[i][thr_idx];
            }

            local_ncheck = sum % 2;
        }
        // calc summ of local_ncheck from all 38 threads
        int ncheck = sum_reduction_two_cycles(local_ncheck, arr_reduction_helper);
        convert_cw_to_bytes(cw, crc_buf);

        bool is_crc_valid = false;
        // only first thread calculates and checks CRC and only if no errors in check table.
        if(ncheck == 0 && threadIdx.x == 0)
        {
            is_crc_valid = check_crc(crc_buf, crc_table);
        }

        int is_bit_bad = (cw[thr_idx] == 1 && softbits[thr_idx] > 0.0f || cw[thr_idx] == 0 && softbits[thr_idx] <= 0.0f)? 0: 1;
        int num_hard_errors = sum_reduction_two_cycles(is_bit_bad, arr_reduction_helper);
        if(threadIdx.x == 0)
        {
            // Only thread 0 has is_crc_valid in valid state.
            // Take it from this thread and save to common shared memory.
            message_found = is_crc_valid && num_hard_errors < 18;
        }
        __syncthreads();

        if(message_found)
        {
            // Only thread 0 saves the result.
            if(threadIdx.x == 0)
            {
                ctx.resultKeeper().put_ldpc_decode_result(blockIdx.x, cw, iter, num_hard_errors);
            }

            return;
        }


        // ---- Send messages from bits to check nodes 
        for(int k=0; k < 3; k++)
        {
            toc[mp[thr_idx][k][0]][mp[thr_idx][k][1]] = zn[thr_idx] - tov[k][thr_idx]; // subtract off what the bit had received from the check
        }
        __syncthreads();

        // ----- send messages from check nodes to variable nodes
        for(int k=0; k < 3; k++)
        {
            const int column = mp[thr_idx][k][1];
            const int row_to_exclude = mp[thr_idx][k][0];
            float product = 1.0f;
            for(int j=0; j < 11; j++)
            {
                if((j < 10 || is_full_row[column]) && j != row_to_exclude) 
                {
                    product *= tanh(-0.5f * toc[j][column]);
                }
            }

            tov[k][thr_idx] = 2.0f * platanh(-product);
        }
        __syncthreads();

    }
}
