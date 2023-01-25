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

DecodedResult::DecodedResult(std::string msg)
    : m_found(true)
    , m_message(std::move(msg))
{
}

DecodedResult decode_softbits(const std::vector<float>& softbits)
{
    const size_t softbits_size = 128;
    if(softbits.size() != softbits_size)
    {
        throw std::runtime_error("softbits[] must have size of 128 elements.");
    }

    // The following code is a part of msk144decodeframe.f90

    std::vector<char> apmask(128, 0);
    int maxiterations = 10;
    std::vector<char> message77(77);
    char cw[128];
    int nharderror = 0;
    int iter = 0;

    std::vector<float> llr = softbits;

    fortran_bpdecode128_90(&llr[0], &apmask[0], &maxiterations, &message77[0], cw, &nharderror, &iter);

    // first check for error
    if(nharderror < 0 || nharderror >= 18)
        return DecodedResult();

    return decode_message(message77);
}

DecodedResult decode_message(const std::vector<char>& message77)
{
    auto bits2int = [](char b2, char b1, char b0) -> int { return (b2 << 2) | (b1 << 1) | (b0); };
    int n3 = bits2int(message77[71], message77[72], message77[73]);
    int i3 = bits2int(message77[74], message77[75], message77[76]);

    if((i3 == 0 && (n3 == 1 || n3 == 3 || n3 == 4 || n3 > 5)) || i3 == 3 || i3 > 5)
        return DecodedResult();

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
        return DecodedResult();

    // trim right
    auto it = std::find_if(msg.rbegin(), msg.rend(), [](char x) { return x != ' '; });
    size_t num_chars_to_cut = it - msg.rbegin();
    std::string res(msg.begin(), msg.begin() + msg.size() - num_chars_to_cut);

    return DecodedResult(res);
}
