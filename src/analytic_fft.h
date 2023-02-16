//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#pragma once

#include "common.h"
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Analytic
{
public:
    Analytic(const int nfft);
    ~Analytic();
    void execute(thrust::host_vector<Complex> const& in, size_t npts);
    thrust::device_ptr<Complex> getResultDevice();
    thrust::host_vector<Complex> const& getResultHost() const;
    thrust::device_vector<Complex> const& getResultDeviceVector() const;

private:
    int _nfft;
    cufftHandle _fftHandle;
    thrust::host_vector<Complex> _corr;
    thrust::host_vector<float> _h;
    thrust::device_vector<Complex> _analytic_result_dev;
    thrust::host_vector<Complex> _analytic_result_host;

    // void init();
};
