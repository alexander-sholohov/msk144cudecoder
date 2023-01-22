

#define POLY 0x15D7

//struct LDPCMapItem {
//  unsigned char map[3][2];
//};


const char ldpc_reverse_map[128][3][2] = { 
    {{0, 20}, {0, 33}, {0, 35}},
    {{0, 0}, {0, 7}, {0, 27}},
    {{0, 1}, {0, 8}, {0, 36}},
    {{0, 2}, {0, 6}, {0, 18}},
    {{0, 3}, {0, 15}, {0, 31}},
    {{1, 1}, {0, 4}, {0, 21}},
    {{0, 5}, {0, 12}, {0, 24}},
    {{0, 9}, {0, 30}, {0, 32}},
    {{0, 10}, {0, 23}, {0, 26}},
    {{0, 11}, {0, 14}, {0, 22}},
    {{0, 13}, {0, 17}, {0, 25}},
    {{0, 16}, {0, 19}, {0, 28}},
    {{1, 16}, {0, 29}, {1, 33}},
    {{1, 5}, {2, 33}, {0, 34}},
    {{1, 0}, {1, 9}, {1, 29}},
    {{1, 2}, {1, 17}, {1, 22}},
    {{1, 3}, {1, 11}, {1, 24}},
    {{1, 4}, {1, 27}, {1, 35}},
    {{1, 6}, {1, 13}, {1, 20}},
    {{1, 7}, {1, 14}, {1, 30}},
    {{1, 8}, {1, 26}, {1, 31}},
    {{1, 10}, {1, 18}, {1, 34}},
    {{1, 12}, {1, 15}, {1, 36}},
    {{1, 19}, {1, 23}, {0, 37}},
    {{2, 20}, {1, 21}, {1, 25}},
    {{2, 11}, {1, 28}, {1, 32}},
    {{2, 0}, {2, 16}, {2, 34}},
    {{2, 1}, {2, 27}, {2, 29}},
    {{2, 2}, {2, 9}, {2, 31}},
    {{2, 3}, {2, 7}, {2, 35}},
    {{2, 4}, {2, 18}, {2, 28}},
    {{2, 5}, {2, 19}, {2, 26}},
    {{2, 6}, {2, 21}, {2, 36}},
    {{2, 8}, {2, 10}, {2, 32}},
    {{2, 12}, {2, 23}, {2, 25}},
    {{2, 13}, {2, 30}, {3, 33}},
    {{2, 14}, {2, 15}, {2, 24}},
    {{3, 12}, {2, 17}, {1, 37}},
    {{3, 7}, {3, 19}, {2, 22}},
    {{3, 0}, {3, 31}, {3, 32}},
    {{3, 1}, {3, 16}, {3, 18}},
    {{3, 2}, {3, 23}, {4, 33}},
    {{3, 3}, {3, 6}, {2, 37}},
    {{3, 4}, {3, 10}, {3, 30}},
    {{3, 5}, {3, 17}, {3, 20}},
    {{3, 8}, {3, 14}, {3, 35}},
    {{3, 9}, {3, 15}, {3, 27}},
    {{3, 11}, {3, 25}, {3, 29}},
    {{3, 13}, {3, 26}, {3, 28}},
    {{3, 21}, {3, 24}, {3, 34}},
    {{3, 22}, {4, 29}, {4, 31}},
    {{4, 3}, {4, 10}, {3, 36}},
    {{4, 0}, {4, 13}, {4, 22}},
    {{4, 1}, {4, 7}, {4, 24}},
    {{4, 2}, {4, 12}, {4, 26}},
    {{4, 4}, {4, 9}, {4, 36}},
    {{4, 5}, {4, 15}, {4, 30}},
    {{4, 6}, {4, 14}, {4, 17}},
    {{4, 8}, {4, 21}, {4, 23}},
    {{4, 11}, {4, 18}, {4, 35}},
    {{4, 16}, {4, 25}, {3, 37}},
    {{4, 19}, {4, 20}, {4, 32}},
    {{5, 19}, {4, 27}, {4, 34}},
    {{5, 3}, {4, 28}, {5, 33}},
    {{5, 0}, {5, 25}, {5, 35}},
    {{5, 1}, {5, 22}, {6, 33}},
    {{5, 2}, {5, 8}, {4, 37}},
    {{5, 4}, {5, 5}, {5, 16}},
    {{5, 6}, {5, 26}, {5, 34}},
    {{5, 7}, {5, 13}, {5, 31}},
    {{5, 9}, {5, 14}, {5, 21}},
    {{5, 10}, {5, 17}, {5, 28}},
    {{5, 11}, {5, 12}, {5, 27}},
    {{5, 15}, {5, 18}, {5, 32}},
    {{5, 20}, {5, 24}, {5, 30}},
    {{5, 23}, {5, 29}, {5, 36}},
    {{6, 0}, {6, 2}, {6, 20}},
    {{6, 1}, {6, 17}, {6, 30}},
    {{6, 3}, {6, 5}, {6, 8}},
    {{6, 4}, {6, 7}, {6, 32}},
    {{6, 6}, {6, 28}, {6, 31}},
    {{6, 9}, {6, 12}, {6, 18}},
    {{6, 10}, {6, 21}, {6, 22}},
    {{6, 11}, {6, 26}, {7, 33}},
    {{6, 13}, {6, 14}, {6, 29}},
    {{6, 15}, {7, 26}, {5, 37}},
    {{6, 16}, {6, 27}, {6, 36}},
    {{6, 19}, {6, 24}, {6, 25}},
    {{7, 4}, {6, 23}, {6, 34}},
    {{7, 2}, {7, 5}, {6, 35}},
    {{7, 0}, {7, 11}, {7, 30}},
    {{7, 1}, {7, 3}, {7, 32}},
    {{8, 2}, {7, 15}, {7, 29}},
    {{8, 0}, {8, 1}, {7, 23}},
    {{8, 4}, {7, 22}, {8, 26}},
    {{8, 5}, {7, 27}, {7, 31}},
    {{7, 6}, {7, 16}, {7, 35}},
    {{7, 7}, {7, 21}, {6, 37}},
    {{7, 8}, {7, 17}, {7, 19}},
    {{7, 9}, {7, 20}, {7, 28}},
    {{7, 10}, {7, 12}, {8, 33}},
    {{8, 3}, {7, 13}, {8, 19}},
    {{8, 10}, {8, 29}, {7, 37}},
    {{8, 13}, {7, 34}, {7, 36}},
    {{7, 14}, {7, 18}, {7, 25}},
    {{9, 2}, {8, 27}, {8, 28}},
    {{8, 6}, {8, 7}, {8, 8}},
    {{9, 4}, {8, 17}, {9, 33}},
    {{8, 12}, {8, 14}, {8, 16}},
    {{8, 11}, {8, 15}, {8, 34}},
    {{8, 9}, {8, 22}, {7, 24}},
    {{8, 18}, {8, 20}, {8, 36}},
    {{9, 16}, {9, 26}, {8, 30}},
    {{8, 23}, {8, 24}, {8, 35}},
    {{9, 0}, {9, 17}, {9, 18}},
    {{9, 5}, {8, 25}, {8, 32}},
    {{8, 21}, {9, 30}, {8, 31}},
    {{10, 2}, {9, 19}, {9, 21}},
    {{9, 3}, {9, 20}, {10, 26}},
    {{9, 1}, {9, 12}, {9, 28}},
    {{10, 5}, {9, 6}, {9, 11}},
    {{9, 14}, {9, 23}, {9, 31}},
    {{9, 8}, {9, 24}, {9, 29}},
    {{9, 22}, {9, 36}, {8, 37}},
    {{10, 4}, {9, 15}, {9, 25}},
    {{9, 10}, {9, 13}, {9, 27}},
    {{9, 32}, {9, 35}, {9, 37}},
    {{9, 7}, {9, 9}, {9, 34}} 
};


class LDPCContext
{
public:
    LDPCContext() = default;
    LDPCContext(const LDPCContext&) = default;

    __host__ void init() 
    {
        // CRC13
        thrust::host_vector<uint16_t> table(256);
        gen_crc13_table(&table[0]);

        _crc_table = thrust::device_malloc<uint16_t>(256);
        thrust::copy(table.begin(), table.end(), _crc_table);


        // num columns in row.  full=11, not full=10
        _is_full_row = thrust::device_malloc<bool>(38);
        thrust::host_vector<bool> is_full_row(38);
        is_full_row[2] = true;
        is_full_row[4] = true;
        is_full_row[5] = true;
        is_full_row[26] = true;
        thrust::copy(is_full_row.begin(), is_full_row.end(), _is_full_row);

        //
        _ldpc_reverse_map = thrust::device_malloc<char>(sizeof(ldpc_reverse_map));
        thrust::copy((char*)ldpc_reverse_map, (char*)ldpc_reverse_map + sizeof(ldpc_reverse_map), _ldpc_reverse_map);
    }

    __host__ void deinit() 
    {
        thrust::device_free(_crc_table);
        thrust::device_free(_is_full_row);
        thrust::device_free(_ldpc_reverse_map);
    }

    __device__ const uint16_t* get_crc_table() const 
    { 
        return thrust::raw_pointer_cast(_crc_table); 
    }

    __device__ const bool* get_is_full_row() const 
    { 
        return thrust::raw_pointer_cast(_is_full_row); 
    }

    __device__ const char* get_reverse_map() const 
    { 
        return thrust::raw_pointer_cast(_ldpc_reverse_map); 
    }

private:
    void gen_crc13_table(uint16_t* table)
    {
        const int LengthCRC = 13;
        const uint16_t polynomial = POLY;
        const uint16_t high_bit_mask = (1 << (LengthCRC - 1));
        const int N = 256;

        for(int i=0; i<N; i++)
        {
            uint16_t dividend = i;
            uint16_t remainder = 0;
            for (int bit = 0; bit < 8; bit++)
            {
                if(dividend & 0x80)
                {
                    remainder ^= high_bit_mask;
                }
                bool const  quotient = remainder & high_bit_mask;
                remainder <<= 1;
                if(quotient)
                {
                    remainder ^= polynomial;
                }

                dividend <<= 1;
            }
            table[i] = remainder;
        }
    }

private:

    thrust::device_ptr<uint16_t> _crc_table;
    thrust::device_ptr<bool> _is_full_row;
    thrust::device_ptr<char> _ldpc_reverse_map;
};