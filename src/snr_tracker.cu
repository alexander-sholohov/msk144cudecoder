//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#include "snr_tracker.h"
#include <cassert>
#include <cmath>
#include <vector>

SNRTracker::SNRTracker()
    : _noise_power(0.0f)
    , _snr(0.0f)
{
}

void SNRTracker::process_data(const Complex const* data, const unsigned length)
{
    //assert(data.size() >= length);

    const int num_elements = 8;
    std::vector<float> arr(num_elements, 0.0f);
    const int block_size = Num6x864 / num_elements;
    for(int idx = 0; idx < Num6x864; idx++)
    {
        auto y = conj(data[idx]) * data[idx];

        int pos = idx / block_size;
        if(pos < num_elements)
        {
            arr[pos] += y.real();
        }
    }

    float summ = 0.0f;
    for(auto& x : arr)
    {
        summ += x;
    }
    float avg = summ / num_elements;

    float peak = 0.0f;
    for(auto& x : arr)
    {
        if(peak < x)
        {
            peak = x;
        }
    }

    if(_noise_power <= 0.0f)
    {
        _noise_power = avg; // initial
    }
    else if(_noise_power < avg)
    {
        _noise_power = 0.9f * _noise_power + 0.1f * avg; // noise level is slow to rise
    }
    else
    {
        _noise_power = avg; // and quick to fall
    }

    if(_noise_power > 0.0f)
    {
        _snr = 10.0f * std::log10(peak / _noise_power - 1.0f);
    }

    if(_snr > 24.0f)
    {
        _snr = 24.0f;
    }
    if(_snr < -8.0f)
    {
        _snr = -8.0f;
    }
}
