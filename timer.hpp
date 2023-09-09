/*
 * Timer class declaration
 */

// Headers
#include <chrono>
#include <occa.hpp>
#include <cstring>
#include <unordered_map>

// Class definition
#ifndef TIMER_HPP
#define TIMER_HPP

template<typename DType = double>
class Timer
{
    private:
        // Member variables
        std::unordered_map<const char*, std::chrono::_V2::high_resolution_clock::time_point> t_start;
        std::unordered_map<const char*, std::chrono::_V2::high_resolution_clock::time_point> t_stop;
        std::unordered_map<const char*, std::vector<DType>> t_total;
        DType t_sync;

    public:
        // Constructor
        Timer();
        ~Timer();

        // Utility functions
        void initialize();
        void start(const char*, bool = true);
        void stop(const char*, bool = true);
        void reset(const char*);
        DType total(const char*);
        DType total(const char*, const char*);
        void total(const char*, std::string&);
};

#include "timer.tpp"

#endif
