//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

#include <string>
#include <utility>
#include <vector>

class DecodeResult
{
public:
    DecodeResult() = default;
    DecodeResult(std::string msg);
    bool found() const { return m_found; }
    std::string const& message() const { return m_message; }

private:
    bool m_found{};
    std::string m_message{};
};

DecodeResult decode_softbits(const std::vector<float> const& softbits);
DecodeResult decode_message(const std::vector<char> const& message);

void init_pbdecode_if_need();
