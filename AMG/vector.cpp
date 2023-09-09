/*
 * Vector source file
 */

// Class headers
#include "vector.hpp"

// Headers
#include <vector>
#include <cuda_runtime.h>

// Namespaces
using namespace amg;

// Constructors
Vector::Vector()
{
    size = -1;
    data = NULL;

    dtype = (typeid(Float) == typeid(double)) ? CUDA_R_64F : CUDA_R_32F;
}

// Destructor
Vector::~Vector()
{
    if (data != NULL)
    {
        if (strcmp(mem_loc, "host") == 0) delete data;
        else cudaFree(data);
    }

    data = NULL;
}

// Functions
void Vector::initialize(const char *mem_loc_, int size_, Float *data_, cudaStream_t stream_)
{
    if (not ((strcmp(mem_loc_, "host") == 0) or (strcmp(mem_loc_, "device") == 0)))
    {
        printf("Memory location '%s' is not supported\n", mem_loc_);
        exit(EXIT_FAILURE);
    }

    mem_loc = mem_loc_;
    size = size_;

    if (size > 0)
    {
        if (strcmp(mem_loc, "host") == 0)
        {
            data = new Float[size];
            if (data_ != NULL) memcpy(data, data_, size * sizeof(Float));
        }
        else
        {
            stream = stream_;

            cudaMalloc((void**)(&data), size * sizeof(Float));
            if (data_ != NULL) cudaMemcpyAsync(data, data_, size * sizeof(Float), cudaMemcpyHostToDevice, stream);

            cusparseCreateDnVec(&desc, size, data, dtype);
            cublasCreate(&cublas_handle);
            cublasSetStream(cublas_handle, stream);

            cudaStreamSynchronize(stream);
        }
    }
}

extern "C" Float vector_set_to_value(Float*, const Float, const int, cudaStream_t);

void Vector::set_to_value(Float value)
{
    if (strcmp(mem_loc, "host") == 0)
    {
        for (int idx = 0; idx < size; idx++)
            data[idx] = value;
    }
    else
    {
        vector_set_to_value(data, value, size, stream);
    }
}

Float Vector::norm()
{
    Float sum = 0.0;

    if (strcmp(mem_loc, "host") == 0)
    {
        for (int idx = 0; idx < size; idx++)
          sum += data[idx] * data[idx];  
    }
    else
    {
        if (dtype == CUDA_R_64F)
        {
            double sum_;
            cublasDdot(cublas_handle, size, (const double*)(data), 1, (const double*)(data), 1, &sum_);
            sum = sum_;
        }
        else
        {
            float sum_;
            cublasSdot(cublas_handle, size, (const float*)(data), 1, (const float*)(data), 1, &sum_);
            sum = sum_;
        }

    }

    return std::sqrt(sum);
}

Float Vector::dot_product(const Vector &u)
{
    Float sum = 0.0;

    if (strcmp(mem_loc, "host") == 0)
    {
        for (int idx = 0; idx < size; idx++)
          sum += data[idx] * u.data[idx];  
    }
    else
    {
        if (dtype == CUDA_R_64F)
        {
            double sum_;
            cublasDdot(cublas_handle, size, (const double*)(data), 1, (const double*)(u.data), 1, &sum_);
            sum = sum_;
        }
        else
        {
            float sum_;
            cublasSdot(cublas_handle, size, (const float*)(data), 1, (const float*)(u.data), 1, &sum_);
            sum = sum_;
        }
    }

    return sum;
}

void Vector::print(const char *file_name)
{
    if (strcmp(mem_loc, "host") == 0)
    {
        FILE *file_ptr = fopen(file_name, "w");

        for (int idx = 0; idx < size; idx++)
            fprintf(file_ptr, "%.16g\n", data[idx]);

        fclose(file_ptr);
    }
    else
    {
        std::vector<Float> data_(size);
        cudaMemcpyAsync(data_.data(), data, size * sizeof(Float), cudaMemcpyDeviceToHost, stream);

        FILE *file_ptr = fopen(file_name, "w");

        for (int idx = 0; idx < size; idx++)
            fprintf(file_ptr, "%.16g\n", data_[idx]);

        fclose(file_ptr);
    }
}

void Vector::copy_to(Vector &u)
{
    if ((strcmp(mem_loc, "host") == 0) and (strcmp(u.mem_loc, "host") == 0))
        memcpy(u.data, data, size * sizeof(Float));

    else if ((strcmp(mem_loc, "host") == 0) and (strcmp(u.mem_loc, "device") == 0))
        cudaMemcpyAsync(u.data, data, size * sizeof(Float), cudaMemcpyHostToDevice, u.stream);

    else if ((strcmp(mem_loc, "device") == 0) and (strcmp(u.mem_loc, "host") == 0))
        cudaMemcpyAsync(u.data, data, size * sizeof(Float), cudaMemcpyDeviceToHost, stream);

    else
        cudaMemcpyAsync(u.data, data, size * sizeof(Float), cudaMemcpyDeviceToDevice, stream);
}

void Vector::copy_from(const Vector &u)
{
    if ((strcmp(mem_loc, "host") == 0) and (strcmp(u.mem_loc, "host") == 0))
        memcpy(data, u.data, size * sizeof(Float));

    else if ((strcmp(mem_loc, "host") == 0) and (strcmp(u.mem_loc, "device") == 0))
        cudaMemcpyAsync(data, u.data, size * sizeof(Float), cudaMemcpyDeviceToHost, u.stream);

    else if ((strcmp(mem_loc, "device") == 0) and (strcmp(u.mem_loc, "host") == 0))
        cudaMemcpyAsync(data, u.data, size * sizeof(Float), cudaMemcpyHostToDevice, stream);

    else
        cudaMemcpyAsync(data, u.data, size * sizeof(Float), cudaMemcpyDeviceToDevice, stream);
}
