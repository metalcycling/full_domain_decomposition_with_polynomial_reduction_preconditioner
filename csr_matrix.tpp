/*
 * CSR matrix source
 */

// Headers
#include <algorithm>

// Constructor and destructor
template<typename DType>
CSR_Matrix<DType>::CSR_Matrix()
{

}

template<typename DType>
CSR_Matrix<DType>::CSR_Matrix(int num_rows_, int num_cols_)
{
    initialize(num_rows_, num_cols_);
}

template<typename DType>
CSR_Matrix<DType>::~CSR_Matrix()
{

}

template<typename DType>
void CSR_Matrix<DType>::initialize(int num_rows_, int num_cols_)
{
    num_rows = num_rows_;
    num_cols = num_cols_;
    num_nnz = 0;

    int proc_id; MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    occa::properties properties;

    if (typeid(DType) == typeid(double))
        properties["defines/DType"] = "double";
    else
        properties["defines/DType"] = "float";

    if (proc_id == 0)
    {
        multiply_kernel = device.buildKernel("csr_matrix.okl", "multiply", properties);
        multiply_range_kernel = device.buildKernel("csr_matrix.okl", "multiply_range", properties);
        multiply_weight_kernel = device.buildKernel("csr_matrix.okl", "multiply_weight", properties);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (proc_id > 0)
    {
        multiply_kernel = device.buildKernel("csr_matrix.okl", "multiply", properties);
        multiply_range_kernel = device.buildKernel("csr_matrix.okl", "multiply_range", properties);
        multiply_weight_kernel = device.buildKernel("csr_matrix.okl", "multiply_weight", properties);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (typeid(DType) == typeid(double))
        sparse_tolerance = 1.0e-12;
    else
        sparse_tolerance = 1.0e-6;

    is_initialized = true;
}

template<typename DType>
void CSR_Matrix<DType>::add_entry(int row, int col, DType val)
{
    if ((row < 0) or (row >= num_rows) or (col < 0) or (col >= num_cols))
    {
        printf("ERROR: Entry at (%d, %d) is outside the matrix of size (%d, %d)\n", row, col, num_rows, num_cols);
        exit(EXIT_FAILURE);
    }

    // Add entry
    if (std::abs(val) > sparse_tolerance)
        entries.push_back(std::tuple<int, int, DType>(row, col, val));
}

template<typename DType>
void CSR_Matrix<DType>::initialization_check()
{
    if (not is_initialized)
    {
        printf("ERROR: CSR matrix has not been initialize\n");
        exit(EXIT_FAILURE);
    }
}

template<typename DType>
void CSR_Matrix<DType>::assemble()
{
    if ((num_rows == 0) or (num_cols == 0) or (entries.size() == 0)) return;

    // Check if initialized
    initialization_check();

    // Sort entries by row first and column second
    std::sort(entries.begin(), entries.end(), [](const std::tuple<int, int, DType> &a, const std::tuple<int, int, DType> &b)
    {
        if (std::get<0>(a) < std::get<0>(b))
            return true;
        else
            if (std::get<0>(a) > std::get<0>(b))
                return false;
            else
                if (std::get<1>(a) < std::get<1>(b))
                    return true;
                else
                    return false;
    });

    // Count how many non-zeros there are
    std::tuple<int, int, DType> current = entries[0];
    num_nnz = 1;

    for (unsigned int i = 1; i < entries.size(); i++)
    {
        std::tuple<int, int, DType> &entry = entries[i];

        if ((std::get<0>(entry) != std::get<0>(current)) or (std::get<1>(entry) != std::get<1>(current)))
        {
            current = entry;
            num_nnz++;
        }
    }

    // Allocate memory
    ptr = device.malloc<int>(num_rows + 1);
    col = device.malloc<int>(num_nnz);
    val = device.malloc<DType>(num_nnz);

    // Assemble
    int count = 0;
    int *ptr_hst = new int[num_rows + 1]();
    int *col_hst = new int[num_nnz];
    DType *val_hst = new DType[num_nnz];

    current = entries[0];

    ptr_hst[std::get<0>(current) + 1]++;
    col_hst[count] = std::get<1>(current);
    val_hst[count] = std::get<2>(current);

    for (unsigned int i = 1; i < entries.size(); i++)
    {
        std::tuple<int, int, DType> &entry = entries[i];

        if ((std::get<0>(entry) != std::get<0>(current)) or (std::get<1>(entry) != std::get<1>(current)))
        {
            current = entry;
            count++;

            ptr_hst[std::get<0>(entry) + 1]++;
            col_hst[count] = std::get<1>(entry);
            val_hst[count] = std::get<2>(entry);
        }
        else
        {
            val_hst[count] += std::get<2>(entry);
        }
    }

    for (int i = 1; i <= num_rows; i++)
        ptr_hst[i] += ptr_hst[i - 1];

    ptr.copyFrom(ptr_hst, (num_rows + 1) * sizeof(int));
    col.copyFrom(col_hst, num_nnz * sizeof(int));
    val.copyFrom(val_hst, num_nnz * sizeof(DType));

    // Free memory
    entries.clear();

    delete[] ptr_hst;
    delete[] col_hst;
    delete[] val_hst;
}

template<typename DType>
void CSR_Matrix<DType>::print(FILE *file_ptr, int offset)
{
    // Print matrix
    if (file_ptr == NULL)
    {
        pstdout("num_rows = %d, num_cols = %d, num_nnz = %d\n", num_rows, num_cols, num_nnz);
    }
    else
    {
        fprintf(file_ptr, "num_rows = %d, num_cols = %d, num_nnz = %d\n", num_rows, num_cols, num_nnz);
    }

    if ((num_rows == 0) or (num_cols == 0) or (num_nnz == 0)) return;

    // Get device data
    int *ptr_hst = new int[num_rows + 1]();
    int *col_hst = new int[num_nnz];
    DType *val_hst = new DType[num_nnz];

    ptr.copyTo(ptr_hst, (num_rows + 1) * sizeof(int));
    col.copyTo(col_hst, num_nnz * sizeof(int));
    val.copyTo(val_hst, num_nnz * sizeof(DType));

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = ptr_hst[i]; j < ptr_hst[i + 1]; j++)
        {
            if (file_ptr == NULL)
            {
                pstdout("(%d, %d): %.16g\n", i + offset, col_hst[j] + offset, val_hst[j]);
            }
            else
            {
                fprintf(file_ptr, "(%d, %d): %.16g\n", i + offset, col_hst[j] + offset, val_hst[j]);
            }
        }
    }

    // Free memory
    delete[] ptr_hst;
    delete[] col_hst;
    delete[] val_hst;
}

template<typename DType>
void CSR_Matrix<DType>::transpose(CSR_Matrix &At)
{
    At.initialize(num_cols, num_rows);

    if ((num_rows == 0) or (num_cols == 0)) return;

    // Get device data
    int *ptr_hst = new int[num_rows + 1]();
    int *col_hst = new int[num_nnz];
    DType *val_hst = new DType[num_nnz];

    ptr.copyTo(ptr_hst, (num_rows + 1) * sizeof(int));
    col.copyTo(col_hst, num_nnz * sizeof(int));
    val.copyTo(val_hst, num_nnz * sizeof(DType));

    // Transpose matrix
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = ptr_hst[i]; j < ptr_hst[i + 1]; j++)
        {
            At.add_entry(col_hst[j], i, val_hst[j]);
        }
    }

    At.assemble();

    // Free memory
    delete[] ptr_hst;
    delete[] col_hst;
    delete[] val_hst;
}

template<typename DType>
void CSR_Matrix<DType>::diagonal(occa::memory D)
{
    if ((num_rows == 0) or (num_cols == 0)) return;

    // Check if initialized
    initialization_check();

    // Get device data
    int *ptr_hst = new int[num_rows + 1]();
    int *col_hst = new int[num_nnz];
    DType *val_hst = new DType[num_nnz];

    ptr.copyTo(ptr_hst, (num_rows + 1) * sizeof(int));
    col.copyTo(col_hst, num_nnz * sizeof(int));
    val.copyTo(val_hst, num_nnz * sizeof(DType));

    DType *work = new DType[num_rows]();

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = ptr_hst[i]; j < ptr_hst[i + 1]; j++)
        {
            if (i == col_hst[j])
            {
                work[i] = val_hst[j];

                break;
            }
        }
    }

    D.copyFrom(work, num_rows * sizeof(DType));

    // Free memory
    delete[] ptr_hst;
    delete[] col_hst;
    delete[] val_hst;
    delete[] work;
}

template<typename DType>
void CSR_Matrix<DType>::multiply(occa::memory &Au, occa::memory &u)
{
    if ((num_rows == 0) or (num_cols == 0)) return;

    // Check if initialized
    initialization_check();

    // Multiply
    multiply_kernel(Au, ptr, col, val, u, num_rows);
}

template<typename DType>
void CSR_Matrix<DType>::multiply_range(occa::memory &Au, occa::memory &u, int row_start, int row_end)
{
    if ((num_rows == 0) or (num_cols == 0)) return;

    // Check if initialized
    initialization_check();

    // Multiply
    if (row_end < row_start)
    {
        printf("Row end (i_e = %d) has to be greater or equal to row start (i_s = %d)\n", row_end, row_start);
        exit(EXIT_FAILURE);
    }

    multiply_range_kernel(Au, ptr, col, val, u, row_start, row_end);
}

template<typename DType>
void CSR_Matrix<DType>::multiply_weight(occa::memory &Au, occa::memory &u, occa::memory &weight)
{
    if ((num_rows == 0) or (num_cols == 0)) return;

    // Check if initialized
    initialization_check();

    // Multiply
    multiply_weight_kernel(Au, ptr, col, val, u, weight, num_rows);
}
