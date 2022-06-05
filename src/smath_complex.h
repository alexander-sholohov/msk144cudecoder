//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

namespace smath
{

struct Complex;

__host__ __device__ inline Complex operator+(const Complex& x, const Complex& y);

struct Complex
{
    Complex() = default;
    // Complex(const Complex&) = default;

    __host__ __device__ Complex(float re, float im)
        : _re(re)
        , _im(im)
    {
    }

    __host__ Complex(double re, double im)
        : _re(static_cast<float>(re))
        , _im(static_cast<float>(im))
    {
    }

    __host__ __device__ float abs() const { return hypotf(_re, _im); }

    __host__ __device__ float norm() const { return _re * _re + _im * _im; }

    __host__ __device__ float real() const { return _re; }

    __host__ __device__ float imag() const { return _im; }

    __host__ __device__ Complex& operator+=(const Complex& z)
    {
        *this = *this + z;
        return *this;
    }

    float _re;
    float _im;
};

__host__ __device__ inline Complex operator+(const Complex& x, const Complex& y)
{
    return Complex(x.real() + y.real(), x.imag() + y.imag());
}

__host__ __device__ inline Complex operator*(const Complex& x, const Complex& y)
{
    return Complex(x.real() * y.real() - x.imag() * y.imag(), x.real() * y.imag() + x.imag() * y.real());
}

__host__ __device__ inline Complex operator*(float m, const Complex& z)
{
    return Complex(m * z.real(), m * z.imag());
}

__host__ __device__ inline Complex conj(const Complex& z)
{
    return Complex(z.real(), -z.imag());
}

__host__ __device__ inline float abs(const Complex& z)
{
    return z.abs();
}

__host__ __device__ inline float norm(const Complex& z)
{
    return z.norm();
}

__device__ Complex inline make_complex_from_phi(float phi)
{
    float x;
    float y;
    sincosf(phi, &y, &x);
    return Complex(x, y);
}

} // namespace smath
