/*
 * CSR_Matrix header
 */

// Headers
#include <cstring>
#include <cusparse.h>
#include "AMG/config.hpp"
#include "AMG/vector.hpp"

// Class declaration
#ifndef AMG_CSR_MATRIX_HPP
#define AMG_CSR_MATRIX_HPP

namespace amg
{

class CSR_Matrix
{
    public:
        // Member variables
        const char* mem_loc;

        int num_rows;
        int num_cols;
        int num_nnz;

        int *ptr;
        int *col;
        Float *val;

        cusparseSpMatDescr_t desc;
        cusparseHandle_t cusparse_handle;
        cudaStream_t stream;
        cudaDataType dtype;

        size_t buffer_size;
        void *buffer_data;

        // Constructors
        CSR_Matrix();

        // Destructor
        ~CSR_Matrix();

        // Functions
        void initialize(const char*, int, int, int, int* = NULL, int* = NULL, Float* = NULL, cudaStream_t = NULL);
        void matvec(Vector&, const Vector&, const Float = 1.0, const Float = 0.0);
        void matvec(Vector&, const Vector&, const Vector&, const Float = 1.0, const Float = 0.0);
        void print(const char*);
};

}

#endif
