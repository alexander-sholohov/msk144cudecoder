//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#pragma once

#include "smath_complex.h"

// using Complex = thrust::complex<float>;
using Complex = smath::Complex;

constexpr unsigned Num864 = 864;
constexpr unsigned Num6x864 = 6 * 864; // 5184

constexpr unsigned Num42 = 42;

constexpr unsigned FirstSyncBase = 0;
constexpr unsigned SecondSyncBase = (8 + 48) * 6;

constexpr unsigned FirstHardbitsSyncBase = 0;
constexpr unsigned SecondHardbitsSyncBase = (8 + 48);

constexpr unsigned NumberOfSoftBits = 144;
constexpr unsigned NumberOfSoftBitsWithoutSync = 128;
constexpr unsigned NumberOfMessageBits = 77;

constexpr unsigned NumberOfLDPCIterations = 10;

constexpr unsigned NumScanThreads = 256; // must be power of two. 64, 128, 256, 512, 1024 are ok.
#define NUM_SCAN_THREADS (256)           // this is for conditional compilation due to lack of "if constexpr" support in Nvidia Jetson

constexpr unsigned NumCandidatesPerPattern = 8; // 8 or 16 are ok, but no more 32 because of reduction.
#define NUM_CANDIDATES_PER_PATTERN (16)         // emulation "if constexpr"

constexpr unsigned WarpSize = 32; // CUDA constant.

constexpr unsigned ScanDepthMax = 8;
constexpr unsigned FixedNumBitsInPattern = 6;
constexpr unsigned NumPatternBitsToScan = 6; // from 1 to FixedNumBitsInPattern. scan_depth will be limited by this value.

constexpr unsigned NumSoftbitsThreads = 160;

constexpr unsigned NumIndexThreads = 64;

constexpr float SampleRate = 12000.0f;
