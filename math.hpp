/*
 * Math header file
 */

// Headers
#include "config.hpp"

// Functions declaration
#ifndef MATH_HPP
#define MATH_HPP

template<typename DType>
class Math
{
    private:
        // Kernels
        occa::properties properties;
        occa::kernel set_to_value_kernel;
        occa::kernel invert_vector_elements_kernel;
        occa::kernel vector_vector_addition_kernel;
        occa::kernel vector_scaling_kernel;

    public:
        // Constructor and destructor
        Math();
        ~Math();

        // Member functions
        void set_to_value(occa::memory&, DType, int, int = 0);
        void invert_vector_elements(occa::memory&, int);
        void matrix_matrix_multiply(DType*, const DType*, const DType*, int, int, int, bool = false, bool = false);
        void vector_vector_addition(occa::memory&, const DType, const occa::memory&, const DType, const occa::memory&, const int);
        void vector_scaling(occa::memory&, const DType, const occa::memory&, const int);
};

#include "math.tpp"

#endif
