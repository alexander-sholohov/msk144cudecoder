//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: GPLv3
//

#pragma once

#include <chrono>
#include <string>

//============================================
class SimpleMetrics
{
public:
    SimpleMetrics(std::string name, int level = 0);
    ~SimpleMetrics();
    void start();
    void stop();

private:
    int m_level;
    std::string m_name;
    bool m_is_working = false;
    std::chrono::high_resolution_clock::time_point m_stamp;
};

//============================================
class SimpleTimer
{
public:
    SimpleTimer();
    void reset();
    int millisecondsElapsed() const;
    int microsecondsElapsed() const;

private:
    std::chrono::high_resolution_clock::time_point m_stamp;
};
