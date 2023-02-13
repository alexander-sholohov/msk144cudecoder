//
// Author: Alexander Sholokhov <ra9yer@yahoo.com>
//
// License: MIT
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#include "msk_context.cuh"
#include "scan_kernel.cuh"

#include "analytic_fft.h"
#include "analytic2.cuh"
#include "softbits_kernel.cuh"
#include "ldpc_kernel.cuh"
#include "index_kernel.cuh"

#include "decode_softbits.h"
#include "f_interop.h"
#include "metrics.h"
#include "result_filter.h"
#include "snr_tracker.h"

#include <thrust/reduce.h>

#include <cuComplex.h>
#include <cufft.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <map>

#ifdef _MSC_VER
#define __GNU_LIBRARY__ // win32 getopt specific
#endif

#include <getopt.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define SET_BINARY_MODE(handle) setmode(handle, O_BINARY)
#else
#define SET_BINARY_MODE(handle) ((void)0)
#endif

void do_decode(MSK144SearchContext& ctx, const Complex* device_input_data, ResultFilter& result_filter, SNRTracker& snr_tracker);

void showHelp(const char* prog)
{
    // clang-format off
    std::cout << "Calling conversion: " << prog << " {[--help] | <options> }" << std::endl;
    std::cout << " Where options are: " << std::endl;
    std::cout << "                   --help                      Show this help and exit." << std::endl;
    std::cout << "                   --center-frequency=1500.0   Center frequency in Hz." << std::endl;
    std::cout << "                   --search-step=2.0           Search step in Hz. " << std::endl;
    std::cout << "                   --search-width=100.0        Window in Hz around center frequency to find msk144 signal in. The more Search Width the more GPU resources are needed." << std::endl;
    std::cout << "                   --scan-depth=[1..8]         The more depth the more averagable patterns will be tried. Default=3" << std::endl;
    std::cout << "                   --read-mode=[1|2]           1=Audio,16 bit, mono, 12000sps, 1500Hz-recommended center; 2=IQ,8 bit, 12000sps, 0Hz-center. Default mode = 1." << std::endl;
    std::cout << "                   --analytic-method=[1|2]     How to convert real signal to ananlytyc quadrature signal. 1 = FFT; 2 = Shift-left + LPF + Shift-right. Default=2." << std::endl;
    std::cout << "                   --nbadsync-threshold=[1..4] Specifies how many errors in sync pattern are acceptable to be passed to LDPC decoder. Default=2." << std::endl;
    // clang-format on

    return;
}

enum class INPUT_MODE
{
    Real16bit = 1,
    IQ8bits = 2,
};

static std::string input_mode_to_string(INPUT_MODE input_mode)
{
    std::string res;
    if(input_mode == INPUT_MODE::Real16bit)
    {
        res = "Audio. 16 bits signed.";
    }
    else if(input_mode == INPUT_MODE::IQ8bits)
    {
        res = "IQ. 8+8 bits.";
    }
    else
    {
        res = "unknown";
    }

    return res;
}

int main(int argc, char* const argv[])
{
    try
    {
        // Windows specific - load bpdecode.dll. The dll has to be built under MinGW/MSYS2 environment.
        init_f_interop();
    }
    catch(std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 2;
    }

    SET_BINARY_MODE(0); // Windws specific - set stdin as binary.

    // This should decrease CPU usage a little on waiting GPU result.
#if 1
    if(cudaSuccess != cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync))
    {
        std::cerr << "cudaSetDeviceFlags error." << std::endl;
        return 2;
    }
#endif

    INPUT_MODE input_mode = INPUT_MODE::Real16bit;

    const float default_center_frequency_audio = 1500.0f;
    const float default_center_frequency_iq = 0.0f;
    float center_frequency_in_hz = 0.0f; // Will be overwritten.
    bool center_frequency_was_set = false;

    float search_step_in_hz = 2.0f;    // 2Hz is optimal.
    float search_width_in_hz = 100.0f; // 100Hz width is +-50Hz of center frequency.
    int scan_depth = 3;                // the more level the more blocks will be averaged
    int analytic_method = 2;
    int nbadsync_threshold = 1; // max reasonable value = 4

    // clang-format off
    static struct option long_options[] = {{"help", no_argument, 0, 0},                     // 0
                                           {"center-frequency", required_argument, 0, 0},   // 1
                                           {"search-step", required_argument, 0, 0},        // 2
                                           {"search-width", required_argument, 0, 0},       // 3
                                           {"scan-depth", required_argument, 0, 0},         // 4
                                           {"read-mode", required_argument, 0, 0},          // 5
                                           {"analytic-method", required_argument, 0, 0},    // 6
                                           {"nbadsync-threshold", required_argument, 0, 0}, // 7
                                           {0, 0, 0, 0}};
    // clang-format on

    while(true)
    {
        int option_index = 0;

        int c = getopt_long(argc, argv, "", long_options, &option_index);

        if(c == -1)
            break;

        if(c == 0)
        {
            switch(option_index)
            {
            case 0:
                showHelp(argv[0]);
                exit(0);
            case 1:
                center_frequency_in_hz = atof(optarg);
                center_frequency_was_set = true;
                break;
            case 2:
                search_step_in_hz = atof(optarg);
                break;
            case 3:
                search_width_in_hz = atof(optarg);
                break;
            case 4:
                scan_depth = atoi(optarg);
                break;
            case 5:
                input_mode = static_cast<INPUT_MODE>(atoi(optarg));
                break;
            case 6:
                analytic_method = atoi(optarg);
                break;
            case 7:
                nbadsync_threshold = atoi(optarg);
                break;
            default:
                showHelp(argv[0]);
                exit(0);
            }
        }
    }

    // force set center frequency to default if it not set by user
    if(!center_frequency_was_set)
    {
        if(input_mode == INPUT_MODE::Real16bit)
        {
            center_frequency_in_hz = default_center_frequency_audio;
        }
        else if(input_mode == INPUT_MODE::IQ8bits)
        {
            center_frequency_in_hz = default_center_frequency_iq;
        }
        else
        {
            std::cerr << "Wrong read mode " << static_cast<int>(input_mode) << std::endl;
            return 2;
        }
    }

    SimpleMetrics sm1("startup");
    thrust::host_vector<short> real16_buf(Num6x864); //

    thrust::host_vector<Complex> a_in_host(Num6x864);
    thrust::device_vector<Complex> a_in_device(Num6x864);
    thrust::host_vector<Complex> a_out_host(Num6x864);
    thrust::device_vector<Complex> a_out_device(Num6x864);

    thrust::host_vector<int8_t> iq2x8_buf(Num6x864 * 2); //
    thrust::host_vector<Complex> iq2x8_complex(Num6x864);

    const int NFFT = 8192;
    thrust::host_vector<Complex> host_complex_8192(NFFT);

    Analytic analytic(NFFT);

    MSK144SearchContext ctx(center_frequency_in_hz, search_width_in_hz, search_step_in_hz, scan_depth, nbadsync_threshold);
    const auto blocks = ctx.getBlocks();
    const auto threads = ctx.getThreads();

    const auto sbBlocks = ctx.getSoftBitsBlocks();
    const auto sbThreads = ctx.getSoftBitsThreads();

    std::cerr << "Actual parameters:" << std::endl
              << "Center Frequency: " << center_frequency_in_hz << "Hz" << std::endl
              << "Search Step: " << search_step_in_hz << "Hz" << std::endl
              << "Search Width: " << search_width_in_hz << "Hz" << std::endl
              << "Scan Depth: " << ctx.scanDepth() << std::endl
              << "Left Boundary: " << ctx.leftBound() << "Hz" << std::endl
              << "Right Boundary: " << ctx.rightBound() << "Hz" << std::endl
              << "Read Mode: (" << input_mode_to_string(input_mode) << ")" << std::endl;

    if(input_mode == INPUT_MODE::Real16bit)
    {
        std::cerr << "Analytic Method: " << analytic_method << std::endl;
    }

    std::cerr << "Badsync Threshold: " << ctx.getNBadSyncThreshold() << std::endl
              << "Scan-kernel CUDA blocks: " << blocks.x << std::endl
              << "Scan-kernel CUDA threads: " << threads.x << std::endl
              << "Softbit-kernel CUDA blocks: " << sbBlocks.x << "*" << sbBlocks.y << "=" << (sbBlocks.x * sbBlocks.y) << std::endl
              << "Softbit-kernel CUDA threads: " << sbThreads.x << std::endl
              << std::endl;

    sm1.stop();

    bool first_read = true;

    SNRTracker snr_tracker;
    ResultFilter result_filter;

    while(true)
    {
        SimpleMetrics sm5("working-loop-total");
        auto working_loop_calculation_timer = SimpleTimer();

        const Complex* host_data_buffer = 0;
        const Complex* device_data_buffer = 0;

        if(input_mode == INPUT_MODE::Real16bit)
        {
            if(first_read)
            {
                size_t rc = fread(&real16_buf[0], sizeof(short), Num6x864, stdin);
                if(rc != Num6x864)
                {
                    std::cerr << "Incomplete read error. rc=" << rc << std::endl;
                    break;
                }

                first_read = false;
            }
            else
            {
                size_t half_len = Num6x864 / 2;
                // copy second half to begin
                memcpy(&real16_buf[0], &real16_buf[half_len], half_len * sizeof(short));
                // read to second part
                size_t rc = fread(&real16_buf[half_len], sizeof(short), half_len, stdin);
                if(rc != half_len)
                {
                    std::cerr << "Incomplete read error. rc=" << rc << std::endl;
                    break;
                }
            }

            working_loop_calculation_timer.reset(); // start calculating after read

            // convert to IQ using Direct FFT + Inverse FFT

            SimpleMetrics sm2("reduce");
            float sum_rms2 = thrust::reduce(real16_buf.begin(), real16_buf.end(), 0.0f, [](float acc, short const& x) -> double {
                float b = static_cast<float>(x);
                return acc + b * b;
            });

            float rms = sqrt(sum_rms2 / Num6x864);
            float fac = 1.0f / rms;
            sm2.stop();

            if(analytic_method == 1)
            {
                SimpleMetrics sm3("fft");
                thrust::transform(real16_buf.begin(), real16_buf.end(), host_complex_8192.begin(), [fac](short v) -> Complex { return Complex(fac * v, 0.0f); });
                analytic.execute(host_complex_8192, Num6x864);
                sm3.stop();

                host_data_buffer = analytic.getResultHost().data();
                device_data_buffer = thrust::raw_pointer_cast(analytic.getResultDevice());
            }
            else if(analytic_method == 2)
            {
                SimpleMetrics sm3("analytic2");
                thrust::transform(real16_buf.begin(), real16_buf.end(), a_in_host.begin(), [fac](short v) -> Complex { return Complex(fac * v, 0.0f); });
                // copy to device
                thrust::copy(a_in_host.begin(), a_in_host.end(), a_in_device.begin());
                apply_shift_filter_shift<Num6x864, 32><<<1, 32>>>(thrust::raw_pointer_cast(a_in_device.data()), thrust::raw_pointer_cast(a_out_device.data()));
                thrust::copy(a_out_device.begin(), a_out_device.end(), a_out_host.begin());
                sm3.stop();

                host_data_buffer = a_out_host.data();
                device_data_buffer = thrust::raw_pointer_cast(a_out_device.data());
            }
        }
        else if(input_mode == INPUT_MODE::IQ8bits)
        {
            SimpleMetrics sm21("fread");
            if(first_read)
            {
                size_t rc = fread(&iq2x8_buf[0], sizeof(int8_t), Num6x864 * 2, stdin);
                if(rc != Num6x864 * 2)
                {
                    std::cerr << "Incomplete read error. rc=" << rc << std::endl;
                    break;
                }
                first_read = false;
            }
            else
            {
                const size_t half_len = Num6x864;
                // copy second half to begin
                memcpy(&iq2x8_buf[0], &iq2x8_buf[half_len], half_len);
                // read to second part
                size_t rc = fread(&iq2x8_buf[half_len], sizeof(int8_t), half_len, stdin);
                if(rc != half_len)
                {
                    std::cerr << "Incomplete read error. rc=" << rc << std::endl;
                    break;
                }
            }
            sm21.stop();

            working_loop_calculation_timer.reset(); // start calculating after read

            // convert to vector of Complex.
            for(int idx = 0; idx < Num6x864; idx++)
            {
                const float i = static_cast<float>(iq2x8_buf[idx * 2 + 0]);
                const float q = static_cast<float>(iq2x8_buf[idx * 2 + 1]);
                const float divider = 128.0f;
                iq2x8_complex[idx] = Complex(i / divider, q / divider);
            }

            thrust::copy(iq2x8_complex.begin(), iq2x8_complex.end(), a_in_device.begin());
            apply_filter<Num6x864, 32><<<1, 32>>>(thrust::raw_pointer_cast(a_in_device.data()), thrust::raw_pointer_cast(a_out_device.data()));

            // we copy to host (a_out_host) for SNR calculation
            thrust::copy(a_out_device.begin(), a_out_device.end(), a_out_host.begin());

            host_data_buffer = a_out_host.data();
            device_data_buffer = thrust::raw_pointer_cast(a_out_device.data());
        }
        else
        {
            std::cerr << "Unsupported mode. Exit." << std::endl;
            break;
        }

        snr_tracker.process_data(host_data_buffer, Num6x864); // calculate peak and AVG power

        result_filter.blockBegin();

        do_decode(ctx, device_data_buffer, result_filter, snr_tracker);

        // aggregate result
        result_filter.blockEnd();

        // Print warning if consume a lot of cpu time.
        const int working_loop_timeout_soft_limit_in_ms = 210;
        if(working_loop_calculation_timer.millisecondsElapsed() > working_loop_timeout_soft_limit_in_ms)
        {
            std::cerr << "Warning: Working loop takes too much time: " << working_loop_calculation_timer.millisecondsElapsed() << " ms"
                      << " of " << working_loop_timeout_soft_limit_in_ms << " ms max." << std::endl;
        }

        // print agregated result
        for(auto const& elm : result_filter.getBlockResult())
        {
            // clang-format off
            std::cout << "***  "
                      << "snr=" << std::setw(2) << elm.snr << "; "
                      << "f0=" << std::setw(6) << elm.f0 << "; "
                      << "num_avg=" << elm.num_avg << "; "
                      << "nbadsync=" << elm.nbadsync << "; "
                      << "pattern_idx=" << elm.pattern_idx << "; "
                      << "date=" << elm.updateStampAsString() << "; "
                      << "msg='" << elm.message << "'" << "; "
                      << std::endl;
            // clang-format on
        }

        sm5.stop();
    }

    std::cout << "Done" << std::endl;
    return 0;
}

void do_decode(MSK144SearchContext& ctx, const Complex* device_input_data, ResultFilter& result_filter, SNRTracker& snr_tracker)
{
    struct MessageItem
    {
        MessageItem(const std::vector<char>& message)
            : m(message)
        {
        }
        std::vector<char> m;
        bool operator<(const MessageItem& other) const
        {
            for(size_t idx = 0; idx > m.size(); idx++)
            {
                if(m[idx] < other.m[idx])
                    return true;
            }
            return false;
        }
    };

    std::map<MessageItem, DecodedResult> decode_cache;

    SimpleMetrics sm1("do_decode");

    int cnt_decoded = 0;
    int cnt_probes = 0;

    const dim3 blocks = ctx.getBlocks();
    const dim3 threads = ctx.getThreads();

    const auto sbBlocks = ctx.getSoftBitsBlocks();
    const auto sbThreads = ctx.getSoftBitsThreads();

    ctx.resultKeeper().clear_result();

    scan_kernel<<<blocks, threads>>>(ctx, device_input_data);
    softbits_kernel<<<sbBlocks, sbThreads>>>(ctx, device_input_data);
    index_kernel<<<1, NumIndexThreads>>>(ctx);
    const auto ldpcBlocks = ctx.resultKeeper().getIndexedCandidatesBlocks();
    ldpc_kernel<<<ldpcBlocks, 128>>>(ctx);
    if(cudaDeviceSynchronize() != cudaSuccess)
    {
        throw std::runtime_error("Cuda error: Failed to synchronize kernel..");
    }
    sm1.stop();

    SimpleMetrics sm6("JTdecode");

    SimpleMetrics sm_ga("get_all_results", 1);
    thrust::host_vector<ResultKeeper::ResultItem> all_results = ctx.resultKeeper().get_all_results();
    sm_ga.stop();
    SimpleMetrics sm_loop("loop", 1);
    for(unsigned idx = 0; idx < all_results.size(); idx++)
    {
        ResultKeeper::ResultItem const& item = all_results[idx];

        if(item.is_message_present)
        {
            cnt_probes++;

#if 0
            // Sync present. Try to decode entire message.
            std::vector<float> sb(item.softbits_wo_sync, item.softbits_wo_sync + NumberOfSoftBitsWithoutSync);
            auto res = decode_softbits(sb);
#endif

            std::vector<char> message77(item.message, item.message + NumberOfMessageBits);
            MessageItem msg_item(message77);

            if(decode_cache.find(msg_item) == decode_cache.end())
            {
                // Cache miss. Decode message.
                auto res = decode_message(message77);
                decode_cache[msg_item] = res; // put to cache.
            }

            const auto& res = decode_cache[msg_item];

            if(res.found())
            {
                cnt_decoded++;
#if 0
                std::cout << "D " << cnt_decoded << ": "
                    << " snr=" << std::setw(4) << snr_tracker.getSNRI()
                    << " idx=" << idx
                    << " f0=" << std::setw(6) << item.f0
                    << " num_avg=" << item.num_avg
                    << " ptrn_idx=" << item.pattern_idx
                    << " pos=" << std::setw(6) << item.pos
                    << " xb=" << std::setw(8) << item.xb
                    << " badsync=" << std::setw(4) << item.nbadsync
                    << " iter=" << item.ldpc_num_iterations
                    << " msg='" << res.message() << "'" << std::endl;
#endif
                result_filter.putMessage(snr_tracker.getSNRI(), item.f0, item.num_avg, item.nbadsync, item.pattern_idx, res.message());
            }
        }
    }
    sm_loop.stop();
    sm6.stop();

#if 0
    std::cout << "Total/Probes/Success = " << all_results.size() << "/" << cnt_probes << "/" << cnt_decoded << std::endl;
#endif
}