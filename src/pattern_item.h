//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

#include <stdexcept>

struct PatternItem
{
    PatternItem() = default;

    PatternItem(int m0, int m1, int m2, int m3, int m4, int m5)
    {
        static_assert(FixedNumBitsInPattern == 6, "Implemented only for FixedNumBitsInPattern=6");

        mask[0] = m0;
        mask[1] = m1;
        mask[2] = m2;
        mask[3] = m3;
        mask[4] = m4;
        mask[5] = m5;

        for(int i = 0; i < FixedNumBitsInPattern; i++)
        {
            if(mask[i] != 0 && mask[i] != 1)
            {
                throw std::runtime_error("mask value should be in [0,1]");
            }
        }

        int w = 0;
        // int len = 0;
        for(int i = 0; i < FixedNumBitsInPattern; i++)
        {
            if(mask[i])
            {
                w++;
            }
            // if (mask[i]) { len = i; }
        }

        if(w == 0)
        {
            throw std::runtime_error("Zero weight in PatternItem.");
        }

        num_avg = w;
    }

    int num_avg;
    // float weight;
    // int mask_length; // is not using
    int mask[FixedNumBitsInPattern];
};
