/*
 * Vector header
 */

// Headers
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cusparse.h>
#include <cublas_v2.h>
#include "AMG/config.hpp"

// Class declaration
#ifndef AMG_VECTOR_HPP
#define AMG_VECTOR_HPP

namespace amg
{

class Vector
{
    public:
        // Member variables
        const char* mem_loc;

        int size;
        Float *data;
        cusparseDnVecDescr_t desc;
        cublasHandle_t cublas_handle;
        cudaStream_t stream;
        cudaDataType dtype;

        // Constructors
        Vector();

        // Destructor
        ~Vector();

        // Functions
        void initialize(const char*, int, Float* = NULL, cudaStream_t = NULL);
        void set_to_value(Float);
        void print(const char*);
        void copy_to(Vector&);
        void copy_from(const Vector&);
        Float norm();
        Float dot_product(const Vector&);
};

}

#endif
