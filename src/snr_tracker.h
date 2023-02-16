//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#pragma once

#include "common.h"
#include <thrust/host_vector.h>

class SNRTracker
{
public:
    SNRTracker();
    void process_data(const Complex* data, const unsigned length);
    float getSNRF() const { return _snr; }
    int getSNRI() const { return static_cast<int>(_snr); }

private:
    float _noise_power;
    float _snr;
};
