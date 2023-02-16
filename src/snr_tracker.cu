//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#include "snr_tracker.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>


SNRTracker::SNRTracker()
    : _noise_power(0.0f)
    , _snr(0.0f)
{
}

void SNRTracker::process_data(const Complex* data, const unsigned length)
{
    const int num_elements = 8;
    std::vector<float> arr(num_elements, 0.0f);
    const int block_size = length / num_elements;
    for(int idx = 0; idx < block_size * num_elements; idx++)
    {
        auto y = conj(data[idx]) * data[idx];

        int pos = idx / block_size;
        arr[pos] += y.real();
    }

    const float summ = std::accumulate(arr.begin(), arr.end(), 0.0f);
    const float avg = summ / num_elements;

    const float peak = *std::max_element(arr.begin(), arr.end());

    if(_noise_power <= 0.0f)
    {
        _noise_power = avg; // initial
    }
    else if(avg > _noise_power)
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
    else
    {
        _snr = 0.0f;
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
