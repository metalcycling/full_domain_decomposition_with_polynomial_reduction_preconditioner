/*
 * CSR matrix header
 */

// Headers
#include <tuple>
#include <vector>
#include <occa.hpp>
#include "config.hpp"

// Class declaration
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

template<typename DType>
class CSR_Matrix
{
    private:
        // Variables
        int is_initialized = false;
        DType sparse_tolerance;
        std::vector<std::tuple<int, int, DType>> entries;

        // Kernels
        occa::kernel multiply_kernel;
        occa::kernel multiply_range_kernel;
        occa::kernel multiply_weight_kernel;

        // Utility functions
        void initialization_check();

    public:
        // Variables
        int num_rows = 0;
        int num_cols = 0;
        int num_nnz = 0;
        occa::memory ptr;
        occa::memory col;
        occa::memory val;

        // Constructor and destructor
        CSR_Matrix();
        CSR_Matrix(int, int);
        ~CSR_Matrix();

        // Functions
        void initialize(int, int);
        void add_entry(int, int, DType);
        void assemble();
        void print(FILE* = NULL, int = 0);
        void multiply(occa::memory&, occa::memory&);
        void multiply_range(occa::memory&, occa::memory&, int, int);
        void multiply_weight(occa::memory&, occa::memory&, occa::memory&);
        void transpose(CSR_Matrix&);
        void diagonal(occa::memory);
};

#include "csr_matrix.tpp"

#endif
