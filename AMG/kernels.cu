/*
 * CSR_Matrix cuda file
 */

// Headers
#include <stdlib.h>
#include <stdio.h>
#include "AMG/config.hpp"

// Functions definition
__global__ void vector_set_to_value_kernel(Float *data, const Float value, const int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) data[idx] = value;
}

extern "C" void vector_set_to_value(Float *data, const Float value, const int size, cudaStream_t stream)
{
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vector_set_to_value_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(data, value, size);
}

__global__ void main_scaled_residual_kernel(Float *Sr, Float *w, const Float *f_m_Au, const Float *S, const Float alpha, const int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
    {
        Sr[idx] = S[idx] * f_m_Au[idx];
        w[idx] = alpha * Sr[idx];
    }
}

extern "C" void main_scaled_residual(Float *Sr, Float *w, const Float *f_m_Au, const Float *S, const Float alpha, const int size, cudaStream_t stream)
{
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    main_scaled_residual_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(Sr, w, f_m_Au, S, alpha, size);
}

__global__ void main_polynomial_evaluation_kernel(Float *w, Float *v, const Float *r, const Float *D_val, const Float alpha, const int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
    {
        v[idx] *= D_val[idx];
        w[idx] = alpha * r[idx] + v[idx];
    }
}

extern "C" void main_polynomial_evaluation(Float *w, Float *v, const Float *r, const Float *D_val, const Float alpha, const int size, cudaStream_t stream)
{
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    main_polynomial_evaluation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(w, v, r, D_val, alpha, size);
}

__global__ void main_update_field_kernel(Float *u, const Float *w, const Float *D_val, const int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
    {
        u[idx] += D_val[idx] * w[idx];
    }
}

extern "C" void main_update_field(Float *u, const Float *w, const Float *D_val, const int size, cudaStream_t stream)
{
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    main_update_field_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(u, w, D_val, size);
}

// Math functions
__global__ void vector_multiplication_kernel(Float *uv, const Float *u, const Float *v, const int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
    {
        uv[idx] = u[idx] * v[idx];
    }
}

extern "C" void vector_multiplication(Float *uv, const Float *u, const Float *v, const int size, cudaStream_t stream)
{
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vector_multiplication_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(uv, u, v, size);
}
