//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#include "metrics.h"

#include <iostream>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

SimpleMetrics::SimpleMetrics(std::string name, int level)
    : m_level(level)
    , m_name(std::move(name))
{
    start();
}

SimpleMetrics::~SimpleMetrics()
{
    if(m_is_working)
        stop();
}

void SimpleMetrics::start()
{
#ifdef USE_SIMPLE_METRICS
    m_stamp = high_resolution_clock::now();
    m_is_working = true;
#endif
}

void SimpleMetrics::stop()
{
#ifdef USE_SIMPLE_METRICS
    auto now = high_resolution_clock::now();
    duration<double, std::milli> ms_diff = now - m_stamp;

    auto pad1 = std::string(m_level * 2, ' ');
    auto pad2 = std::string(std::max(1, static_cast<int>(25 - m_name.size() - pad1.size())), ' ');
    std::cout << "Measured time: " << pad1 << m_name << pad2 << ms_diff.count() << "ms" << std::endl;
    m_is_working = false;
#endif
}

//------------------------------------------

SimpleTimer::SimpleTimer()
{
    reset();
}

void SimpleTimer::reset()
{
    m_stamp = high_resolution_clock::now();
}

int SimpleTimer::millisecondsElapsed() const
{
    auto now = high_resolution_clock::now();
    duration<double, std::milli> ms_diff = now - m_stamp;
    return static_cast<int>(ms_diff.count());
}

int SimpleTimer::microsecondsElapsed() const
{
    auto now = high_resolution_clock::now();
    duration<double, std::micro> ms_diff = now - m_stamp;
    return static_cast<int>(ms_diff.count());
}
