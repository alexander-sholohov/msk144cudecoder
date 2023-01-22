//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#include "decode_softbits.h"
#include "f_interop.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

DecodeResult::DecodeResult(std::string msg, int iter)
    : m_found(true)
    , m_iter(iter)
    , m_message(std::move(msg))
{
}

DecodeResult decode_softbits(const std::vector<float> const& softbits)
{
    const size_t softbits_size = 144;
    if(softbits.size() != softbits_size)
    {
        throw std::runtime_error("softbits[] must have size of 144 elements.");
    }

    // The following code is a part of msk144decodeframe.f90

    float sum_sav = std::accumulate(softbits.begin(), softbits.end(), 0.0f, [](float acc, float x) { return acc + x; });
    float sum_s2av = std::accumulate(softbits.begin(), softbits.end(), 0.0f, [](float acc, float x) { return acc + x * x; });

    float sav = sum_sav / softbits_size;
    float s2av = sum_s2av / softbits_size;
    float ssig = std::sqrt(s2av - sav * sav);

    std::vector<float> normalized_softbits(softbits_size);
    std::transform(softbits.begin(), softbits.end(), normalized_softbits.begin(), [ssig](float x) { return x / ssig; });

    std::vector<float> llr(128);
    // copy without sync. schema: 8+48+8+80=144, where 8 - sync.
    std::copy(normalized_softbits.begin() + 8, normalized_softbits.begin() + 8 + 48, llr.begin());
    std::copy(normalized_softbits.begin() + 8 + 48 + 8, normalized_softbits.end(), llr.begin() + 48);
    const float sigma = 0.60f;
    std::transform(llr.begin(), llr.end(), llr.begin(), [sigma](float x) { return 2.0f * x / (sigma * sigma); });

    std::vector<char> apmask(128, 0);
    int maxiterations = 10; // 10 was
    std::vector<char> message77(77);
    char cw[128];
    int nharderror = 0;
    int iter = 0;

    fortran_bpdecode128_90(&llr[0], &apmask[0], &maxiterations, &message77[0], cw, &nharderror, &iter);

    // first check for error
    if(nharderror < 0 || nharderror >= 18)
        return DecodeResult();

    return decode_message(message77);
}

DecodeResult decode_message(const std::vector<char> const& message77)
{
    auto bits2int = [](char b2, char b1, char b0) -> int { return (b2 << 2) | (b1 << 1) | (b0); };
    int n3 = bits2int(message77[71], message77[72], message77[73]);
    int i3 = bits2int(message77[74], message77[75], message77[76]);

    if((i3 == 0 && (n3 == 1 || n3 == 3 || n3 == 4 || n3 > 5)) || i3 == 3 || i3 > 5)
        return DecodeResult();

    // rought checks passed. Next step - pass message77 to decoder

    std::vector<char> c77(77);
    std::transform(message77.begin(), message77.end(), c77.begin(), [](char x) { return '0' + x; });

    const int nrx = 1; // nrx=1 when unpacking a received message
    std::vector<char> msg(37, ' ');
    int unpk77_success = 0;

    fortran_unpack77(&c77[0], // 77
        &nrx,
        &msg[0], // 37
        &unpk77_success);

    if(unpk77_success == 0)
        return DecodeResult();

    // trim right
    auto it = std::find_if(msg.rbegin(), msg.rend(), [](char x) { return x != ' '; });
    size_t num_chars_to_cut = it - msg.rbegin();
    std::string res(msg.begin(), msg.begin() + msg.size() - num_chars_to_cut);

    return DecodeResult(res, 0);

}
