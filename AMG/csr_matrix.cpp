/*
 * CSR_Matrix source file
 */

// Class headers
#include "csr_matrix.hpp"

// Headers
#include <vector>
#include <cuda_runtime.h>

// Namespaces
using namespace amg;

// Constructors
CSR_Matrix::CSR_Matrix()
{
    num_rows = -1;
    num_cols = -1;
    num_nnz = -1;

    ptr = NULL;
    col = NULL;
    val = NULL;

    dtype = (typeid(Float) == typeid(double)) ? CUDA_R_64F : CUDA_R_32F;
}

// Destructor
CSR_Matrix::~CSR_Matrix()
{
    if (ptr != NULL)
    {
        if (strcmp(mem_loc, "host") == 0) delete ptr;
        else cudaFree(ptr);
    }

    if (col != NULL)
    {
        if (strcmp(mem_loc, "host") == 0) delete col;
        else cudaFree(col);
    }

    if (val != NULL)
    {
        if (strcmp(mem_loc, "host") == 0) delete val;
        else cudaFree(val);
    }

    ptr = NULL;
    col = NULL;
    val = NULL;
}

// Functions
void CSR_Matrix::initialize(const char *mem_loc_, int num_rows_, int num_cols_, int num_nnz_, int *ptr_, int *col_, Float *val_, cudaStream_t stream_)
{
    if (not ((strcmp(mem_loc_, "host") == 0) or (strcmp(mem_loc_, "device") == 0)))
    {
        printf("Memory location '%s' is not supported\n", mem_loc_);
        exit(EXIT_FAILURE);
    }

    mem_loc = mem_loc_;
    num_rows = num_rows_;
    num_cols = num_cols_;
    num_nnz  = num_nnz_;

    if ((num_rows > 0) and (num_cols > 0) and (num_nnz > 0))
    {
        if (strcmp(mem_loc, "host") == 0)
        {
            ptr = new int[num_rows + 1];
            col = new int[num_nnz];
            val = new Float[num_nnz];

            if ((ptr_ != NULL) and (col_ != NULL) and (val_ != NULL))
            {
                memcpy(ptr, ptr_, (num_rows + 1) * sizeof(int));
                memcpy(col, col_, num_nnz * sizeof(int));
                memcpy(val, val_, num_nnz * sizeof(Float));
            }
        }
        else
        {
            stream = stream_;

            cudaMalloc((void**)(&ptr), (num_rows + 1) * sizeof(int));
            cudaMalloc((void**)(&col), num_nnz * sizeof(int));
            cudaMalloc((void**)(&val), num_nnz * sizeof(Float));

            if ((ptr_ != NULL) and (col_ != NULL) and (val_ != NULL))
            {
                cudaMemcpyAsync(ptr, ptr_, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(col, col_, num_nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(val, val_, num_nnz * sizeof(Float), cudaMemcpyHostToDevice, stream);
            }

            cusparseCreate(&cusparse_handle);
            cusparseSetStream(cusparse_handle, stream);
            cusparseCreateCsr(&desc, num_rows, num_cols, num_nnz, ptr, col, val,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dtype);

            cudaStreamSynchronize(stream);
        }
    }
}

/*
 * Matvec of the form: y = alpha A x + beta y
 */
void CSR_Matrix::matvec(Vector &y, const Vector &x, const Float alpha, const Float beta)
{
    if (strcmp(mem_loc, "host") == 0)
    {
        for (int row = 0; row < num_rows; row++)
        {
            Float Ax = 0.0;

            for (int idx = ptr[row]; idx < ptr[row + 1]; idx++)
                Ax += val[idx] * x.data[col[idx]];

            y.data[row] = alpha * Ax + beta * y.data[row];
        }
    }
    else
    {
#if HOSTNAME == 0
        cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc, x.desc, &beta, y.desc, dtype, CUSPARSE_SPMV_CSR_ALG1, buffer_data);
#else
        cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, desc, x.desc, &beta, y.desc, dtype, CUSPARSE_CSRMV_ALG1, buffer_data);
#endif
    }
}

/*
 * Matvec of the form: z = alpha A x + beta y
 */
void CSR_Matrix::matvec(Vector &z, const Vector &x, const Vector &y, const Float alpha, const Float beta)
{
    if (strcmp(mem_loc, "host") == 0)
    {
        for (int row = 0; row < num_rows; row++)
        {
            Float Ax = 0.0;

            for (int idx = ptr[row]; idx < ptr[row + 1]; idx++)
                Ax += val[idx] * x.data[col[idx]];

            z.data[row] = alpha * Ax + beta * y.data[row];
        }
    }
    else
    {
        printf("Not implemented\n");
        exit(EXIT_SUCCESS);
    }
}

void CSR_Matrix::print(const char *file_name)
{
    if (strcmp(mem_loc, "host") == 0)
    {
        FILE *file_ptr = fopen(file_name, "w");

        for (int row = 0; row < num_rows; row++)
            for (int idx = ptr[row]; idx < ptr[row + 1]; idx++)
                fprintf(file_ptr, "(%9d, %9d): %.16g\n", row, col[idx], val[idx]);

        fclose(file_ptr);
    }
    else
    {
        std::vector<int> ptr_(num_rows + 1);
        std::vector<int> col_(num_nnz);
        std::vector<Float> val_(num_nnz);

        cudaMemcpyAsync(ptr_.data(), ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(col_.data(), col, num_nnz * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(val_.data(), val, num_nnz * sizeof(Float), cudaMemcpyDeviceToHost, stream);

        FILE *file_ptr = fopen(file_name, "w");

        for (int row = 0; row < num_rows; row++)
            for (int idx = ptr_[row]; idx < ptr_[row + 1]; idx++)
                fprintf(file_ptr, "(%9d, %9d): %.16g\n", row, col_[idx], val_[idx]);

        fclose(file_ptr);
    }
}
