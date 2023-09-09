/*
 * Math template file
 */

// Headers

// Constructor and destructor
template<typename DType>
Math<DType>::Math()
{
    if (typeid(DType) == typeid(double))
        properties["defines/DType"] = "double";
    else
        properties["defines/DType"] = "float";

    properties["defines/BLOCK_SIZE"] = BLOCK_SIZE;

    if (proc_id == 0)
    {
        set_to_value_kernel = device.buildKernel("math.okl", "set_to_value", properties);
        invert_vector_elements_kernel = device.buildKernel("math.okl", "invert_vector_elements", properties);
        vector_vector_addition_kernel = device.buildKernel("math.okl", "vector_vector_addition", properties);
        vector_scaling_kernel = device.buildKernel("math.okl", "vector_scaling", properties);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (proc_id > 0)
    {
        set_to_value_kernel = device.buildKernel("math.okl", "set_to_value", properties);
        invert_vector_elements_kernel = device.buildKernel("math.okl", "invert_vector_elements", properties);
        vector_vector_addition_kernel = device.buildKernel("math.okl", "vector_vector_addition", properties);
        vector_scaling_kernel = device.buildKernel("math.okl", "vector_scaling", properties);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename DType>
Math<DType>::~Math()
{

}

template<typename DType>
void Math<DType>::set_to_value(occa::memory &u, DType alpha, int n, int offset)
{
    set_to_value_kernel(u, alpha, n, offset);
}

template<typename DType>
void Math<DType>::invert_vector_elements(occa::memory &u, int n)
{
    invert_vector_elements_kernel(u, n);
}

template<typename DType>
void Math<DType>::vector_vector_addition(occa::memory &uv, const DType alpha, const occa::memory &u, const DType beta, const occa::memory &v, const int n)
{
    vector_vector_addition_kernel(uv, alpha, u, beta, v, n);
}

template<typename DType>
void Math<DType>::vector_scaling(occa::memory &au, const DType alpha, const occa::memory &u, const int n)
{
    vector_scaling_kernel(au, alpha, u, n);
}

template<typename DType>
void Math<DType>::matrix_matrix_multiply(DType *C, const DType *A, const DType *B, int n, int p, int m, bool A_t, bool B_t)
{
    if ((not A_t) and (not B_t))
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                DType C_ij = 0.0;

                for (int k = 0; k < p; k++)
                    C_ij += A[i * p + k] * B[k * p + j];

                C[i * m + j] = C_ij;
            }
        }
    }
    else
    {
        pstdout("Not implemented, yet");
        quit();
    }
}
