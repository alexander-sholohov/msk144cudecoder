//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#include "analytic_fft.h"

#include <cuComplex.h>
#include <math_constants.h>
#include <thrust/reduce.h>

inline cuFloatComplex operator*(const float m, const cuFloatComplex& a)
{
    return make_cuFloatComplex(a.x * m, a.y * m);
}

Analytic::Analytic(const int nfft)
    : _nfft(nfft)
{
    //-----------------------------------------------------------
    cufftResult fftrc;
    fftrc = cufftPlan1d(&_fftHandle, nfft, CUFFT_C2C, 1);
    if(fftrc != CUFFT_SUCCESS)
    {
        throw std::runtime_error("fft plan error");
    }

    _analytic_result_dev.resize(nfft);

    //-----------------------------------------------------------
    float df = 12000.0f / nfft;
    int nh = nfft / 2;

    _h.resize(nfft);

    const float pi = CUDART_PI_F;

    // 2000Hz wide filter across 1500Hz center.

    {
        float t = 1.0f / 2000.0f;
        float beta = 0.1f;
        for(int i = 0; i < nh; i++)
        {
            float ff = i * df;
            float f = ff - 1500.0f;
            _h[i] = 1.0f;
            if(abs(f) > (1 - beta) / (2 * t) && abs(f) <= (1 + beta) / (2 * t))
            {
                _h[i] = _h[i] * 0.5f * (1.0f + cos((pi * t / beta) * (abs(f) - (1 - beta) / (2 * t))));
            }
            else if(abs(f) > (1 + beta) / (2 * t))
            {
                _h[i] = 0;
            }
        }
    }

    float ac[5] = {1.0, 0.0, 0.0, 0.0, 0.0};
    float pc[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    _corr.resize(nfft);

    {
        for(int i = 0; i < nh; i++)
        {
            float ff = i * df;
            float f = ff - 1500.0f;
            float fp = f / 1000.0f;

            float d_corr = ac[0] + fp * (ac[1] + fp * (ac[2] + fp * (ac[3] + fp * ac[4])));
            Complex x_corr = Complex(d_corr, 0.0f);
            float pd = fp * fp * (pc[2] + fp * (pc[3] + fp * pc[4])); //  !ignore 1st two terms
            _corr[i] = x_corr * Complex(cos(pd), sin(pd));
        }
    }
}

Analytic::~Analytic()
{
    cufftDestroy(_fftHandle);
}

void Analytic::execute(thrust::host_vector<Complex> const& in, size_t npts)
{
    assert(npts <= _nfft);

    float fac = 2.0f / _nfft;

    thrust::host_vector<cuFloatComplex> in2(_nfft);

    // copy partial with scaling, leaving rest as zero
    thrust::transform(in.begin(), in.begin() + npts, in2.begin(),
                      [fac](Complex const& x) -> cuFloatComplex
                      {
                          Complex r = fac * x;
                          return make_cuFloatComplex(r.real(), r.imag());
                      });

    thrust::device_vector<cuFloatComplex> in_device = in2;
    thrust::device_vector<cuFloatComplex> out_device(_nfft);

    cufftResult fftrc;

    fftrc = cufftExecC2C(_fftHandle, thrust::raw_pointer_cast(&in_device[0]), thrust::raw_pointer_cast(&out_device[0]),
                         CUFFT_FORWARD);

    if(fftrc != CUFFT_SUCCESS)
    {
        throw std::runtime_error("cufftExecC2C forward error");
    }

    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        throw std::runtime_error("Cuda error: Failed to synchronize");
    }

    thrust::host_vector<cuFloatComplex> specter = out_device;

    // apply filter
    for(int i = 0; i < _nfft / 2; i++)
    {
        specter[i] = _h[i] * specter[i];
    }

    // half DC
    specter[0] = 0.5f * specter[0];

    // fill negative frequency to zero
    thrust::fill(specter.begin() + _nfft / 2, specter.end(), cuFloatComplex());

    // copy to device
    thrust::copy(specter.begin(), specter.end(), out_device.begin());

    fftrc = cufftExecC2C(_fftHandle, thrust::raw_pointer_cast(&out_device[0]), thrust::raw_pointer_cast(&in_device[0]),
                         CUFFT_INVERSE);
    if(fftrc != CUFFT_SUCCESS)
    {
        throw std::runtime_error("cufftExecC2C forward error");
    }

    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        throw std::runtime_error("Cuda error: Failed to synchronize");
    }

    // Copy cuFloatComplex -> Complex.
    // Probaly these types are the same at binary level, but I'm not sure this is valid for all platforms.

    thrust::host_vector<cuFloatComplex> res_arr_in = in_device; // copy from device to host

    _analytic_result_host.resize(res_arr_in.size());
    for(size_t i = 0; i < res_arr_in.size(); i++)
    {
        cuFloatComplex a = res_arr_in[i];
        _analytic_result_host[i] = Complex(cuCrealf(a), cuCimagf(a));
    }

    // copy to class' member
    thrust::copy(_analytic_result_host.begin(), _analytic_result_host.end(), _analytic_result_dev.begin());
}

thrust::device_ptr<Complex> Analytic::getResultDevice()
{
    return thrust::device_ptr<Complex>(&_analytic_result_dev[0]);
}

thrust::host_vector<Complex> const& Analytic::getResultHost() const
{
    return _analytic_result_host;
}

thrust::device_vector<Complex> const& Analytic::getResultDeviceVector() const
{
    return _analytic_result_dev;
}
