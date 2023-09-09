/*
 * Timer definition
 */

// Headers
#include <algorithm>
#include "timer.hpp"

// Functions definition
template<typename DType>
Timer<DType>::Timer()
{

}

template<typename DType>
Timer<DType>::~Timer()
{

}

template<typename DType>
void Timer<DType>::initialize()
{
    // Measure synchronization time
    const int num_tests = 24;
    DType sync_time[num_tests];

    for (int i = 0; i < num_tests; i++)
    {
        std::chrono::_V2::high_resolution_clock::time_point sync_start = std::chrono::_V2::high_resolution_clock::now();
        device.finish();
        std::chrono::_V2::high_resolution_clock::time_point sync_stop = std::chrono::_V2::high_resolution_clock::now();
        std::chrono::duration<DType> t_elapsed = std::chrono::duration_cast<std::chrono::duration<DType>>(sync_stop - sync_start);

        sync_time[i] = t_elapsed.count();
    }

    std::sort(sync_time, sync_time + num_tests);

    if (num_tests % 2 == 0)
        t_sync = 0.5 * (sync_time[(num_tests - 1) / 2] + sync_time[num_tests / 2]);
    else
        t_sync = sync_time[num_tests / 2];
}

template<typename DType>
void Timer<DType>::start(const char *name, bool global_synchronize)
{
    device.finish();
    if (global_synchronize) MPI_Barrier(MPI_COMM_WORLD);
    t_start[name] = std::chrono::_V2::high_resolution_clock::now();
}

template<typename DType>
void Timer<DType>::stop(const char *name, bool global_synchronize)
{
    device.finish();
    t_stop[name] = std::chrono::_V2::high_resolution_clock::now();

    std::chrono::duration<DType> t_elapsed = std::chrono::duration_cast<std::chrono::duration<DType>>(t_stop[name] - t_start[name]);

    if (t_total.find(name) == t_total.end()) t_total[name].resize(num_procs);

    t_total[name][proc_id] += t_elapsed.count() - t_sync;

    if (global_synchronize) MPI_Allreduce(MPI_IN_PLACE, t_total[name].data(), num_procs, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

template<typename DType>
void Timer<DType>::reset(const char *name)
{
    for (int p = 0; p < num_procs; p++)
        t_total[name][p] = 0.0;
}

template<typename DType>
DType Timer<DType>::total(const char *name)
{
    return t_total[name][proc_id];
}

template<typename DType>
DType Timer<DType>::total(const char *name, const char *type)
{
    if (!strcmp(type, "mean"))
    {
        DType t_sum = 0.0;

        for (int p = 0; p < num_procs; p++)
            t_sum += t_total[name][p];

        return t_sum / (DType)(num_procs);
    }
    else if (!strcmp(type, "max"))
    {
        DType t_max = 0.0;

        for (int p = 0; p < num_procs; p++)
            t_max = std::max(t_max, t_total[name][p]);

        return t_max;
    }
    else
    {
        return - 1.0;
    }
}

template<typename DType>
void Timer<DType>::total(const char *name, std::string &output)
{
    char word[80];
    output.clear();

    for (int p = 0; p < num_procs; p++)
    {
        if (p < num_procs - 1)
            sprintf(word, "%12.08f ", t_total[name][p]);
        else
            sprintf(word, "%12.08f", t_total[name][p]);

        output += word;
    }
}
