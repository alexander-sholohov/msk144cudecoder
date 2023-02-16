//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#include "result_filter.h"

#include <algorithm>
#include <iterator>
#include <set>

std::string ResultFilter::ResultItem::updateStampAsString() const
{
    // std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(update_stamp);
    std::tm now_tm = *std::localtime(&now_c);
    char buff[70];
    strftime(buff, sizeof buff, "%Y%m%d%H%M%S", &now_tm);
    return std::string(buff);
}

void ResultFilter::blockBegin()
{
    m_block_scope.clear();
}

void ResultFilter::putMessage(int snr, float f0, int num_avg, int nbadsync, int pattern_idx, std::string const& message)
{
    ResultItem item;
    item.enrance_stamp = std::chrono::system_clock::now();
    item.update_stamp = item.enrance_stamp;
    item.snr = snr;
    item.f0 = f0;
    item.num_avg = num_avg;
    item.nbadsync = nbadsync;
    item.pattern_idx = pattern_idx;
    item.message = message;

    m_block_scope.push_back(std::move(item));
}

void ResultFilter::blockEnd()
{
    // clear previous block
    m_block_result.clear();

    // construct new block result by leaving only unique messages from m_block_scope

    std::set<std::string> only_messages;
    for(auto const& elm : m_block_scope)
    {
        only_messages.insert(elm.message);
    }

    for(auto const& msg : only_messages)
    {
        std::vector<ResultItem> filtered;
        std::copy_if(m_block_scope.begin(), m_block_scope.end(), std::back_inserter(filtered), [msg](ResultItem const& x) { return x.message == msg; });
        // Sort by num_avg. Item with lowest num_avg is more interested. Within same num_avg we sort by nbadsync.
        std::sort(filtered.begin(), filtered.end(),
                  [](ResultItem const& a, ResultItem const& b)
                  {
                      if(a.num_avg == b.num_avg)
                      {
                          return a.nbadsync < b.nbadsync;
                      }

                      return a.num_avg < b.num_avg;
                  });

        m_block_result.push_back(*filtered.begin());
    }
}
