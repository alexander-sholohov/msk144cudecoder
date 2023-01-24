//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

#include <chrono>
#include <string>
#include <vector>

class ResultFilter
{
public:
    struct ResultItem
    {
        std::chrono::system_clock::time_point enrance_stamp;
        std::chrono::system_clock::time_point update_stamp;
        int snr;
        float f0;
        int num_avg;
        int nbadsync;
        int pattern_idx;
        std::string message;

        std::string updateStampAsString() const;
    };

    void blockBegin();
    void blockEnd();
    void putMessage(int snr, float f0, int num_avg, int nbadsync, int pattern_idx, std::string const& message);

    std::vector<ResultItem> const& getBlockResult() const { return m_block_result; }

private:
    std::vector<ResultItem> m_block_scope;
    std::vector<ResultItem> m_block_result;
};
