//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

#include <string>
#include <utility>
#include <vector>

class DecodedResult
{
public:
    DecodedResult() = default;
    DecodedResult(std::string msg);
    bool found() const { return m_found; }
    std::string const& message() const { return m_message; }

private:
    bool m_found{};
    std::string m_message{};
};

DecodedResult decode_softbits(std::vector<float> const& softbits);
DecodedResult decode_message(std::vector<char> const& message);
