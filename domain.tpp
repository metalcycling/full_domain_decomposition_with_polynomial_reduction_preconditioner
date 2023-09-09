/*
 * Domain template file
 */

// Headers
#include <cstdarg>
#include <cstring>
#include <silo.h>
#include <unordered_map>

// Constructor and destructor
template<typename DType>
Domain<DType>::Domain()
{

}

template<typename DType>
Domain<DType>::Domain(char *directory_, int poly_degree_)
{
    initialize(directory_, poly_degree_);
}

template<typename DType>
Domain<DType>::~Domain()
{

}

template<typename DType>
void Domain<DType>::initialize(char *directory_, int poly_degree_)
{
    // Arguments
    directory = directory_;
    poly_degree = poly_degree_;

    // Variables
    int error;
    FILE *file_ptr;
    char file_name[4096];
    const char *format = (typeid(DType) == typeid(double)) ? "%lf" : "%f";

    // Size data
    int n_x, n_y, n_z;
    sprintf(file_name, "%s/lx1_%d/size_%d.%d.dat", directory, poly_degree + 1, proc_id, poly_degree);
    file_ptr = fopen(file_name, "r");
    error = fscanf(file_ptr, "%d %d %d %d %d", &dim, &n_x, &n_y, &n_z, &num_local_elements);

    num_total_elements = num_local_elements;
    MPI_Allreduce(MPI_IN_PLACE, &num_total_elements, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    num_elem_points = std::pow(poly_degree + 1, dim);
    num_local_points = num_local_elements * num_elem_points;
    num_total_points = num_total_elements * num_elem_points;

    // Initialize work arrays
    int num_work_hst = dim;
    work_hst.resize(num_work_hst);
    for (int w = 0; w < num_work_hst; w++) work_hst[w].resize(num_local_points);

    int num_work_dev = dim;
    work_dev.resize(num_work_dev);
    for (int w = 0; w < num_work_dev; w++) work_dev[w] = device.malloc<DType>(num_local_points);

    for (int w = 0; w < num_work_dev; w++) ((DType**)(work_hst[0].data()))[w] = (DType*)(work_dev[w].ptr());
    work_dev_ptr = device.malloc<DType*>(num_work_dev);
    work_dev_ptr.copyFrom(work_hst[0].data(), num_work_dev * sizeof(DType*));

    // Initialize elements
    elements.reserve(num_local_elements);

    for (int e = 0; e < num_local_elements; e++)
    {
        elements.push_back(Element<DType>(e, dim, poly_degree));

        if (e > 0) elements[e].offset = elements[e - 1].offset + elements[e].num_points;
    }

    // Geometry
    if (dim >= 1)
    {
        sprintf(file_name, "%s/lx1_%d/x_%d.%d.dat", directory, poly_degree + 1, proc_id, poly_degree);
        file_ptr = fopen(file_name, (BINARY_INPUT) ? "rb" : "r");

        if (BINARY_INPUT)
        {
            for (auto &elem : elements)
                error = fread(elem.x.data(), sizeof(DType), elem.num_points, file_ptr);
        }
        else
        {
            for (auto &elem : elements)
                for (int v = 0; v < elem.num_points; v++)
                    error = fscanf(file_ptr, format, &(elem.x[v]));
        }

        fclose(file_ptr);
    }

    if (dim >= 2)
    {
        sprintf(file_name, "%s/lx1_%d/y_%d.%d.dat", directory, poly_degree + 1, proc_id, poly_degree);
        file_ptr = fopen(file_name, (BINARY_INPUT) ? "rb" : "r");

        if (BINARY_INPUT)
        {
            for (auto &elem : elements)
                error = fread(elem.y.data(), sizeof(DType), elem.num_points, file_ptr);
        }
        else
        {
            for (auto &elem : elements)
                for (int v = 0; v < elem.num_points; v++)
                    error = fscanf(file_ptr, format, &(elem.y[v]));
        }

        fclose(file_ptr);
    }

    if (dim >= 3)
    {
        sprintf(file_name, "%s/lx1_%d/z_%d.%d.dat", directory, poly_degree + 1, proc_id, poly_degree);
        file_ptr = fopen(file_name, (BINARY_INPUT) ? "rb" : "r");

        if (BINARY_INPUT)
        {
            for (auto &elem : elements)
                error = fread(elem.z.data(), sizeof(DType), elem.num_points, file_ptr);
        }
        else
        {
            for (auto &elem : elements)
                for (int v = 0; v < elem.num_points; v++)
                    error = fscanf(file_ptr, format, &(elem.z[v]));
        }

        fclose(file_ptr);
    }

    // Connectivity
    sprintf(file_name, "%s/lx1_%d/glo_num_%d.%d.dat", directory, poly_degree + 1, proc_id, poly_degree);
    file_ptr = fopen(file_name, (BINARY_INPUT) ? "rb" : "r");

    if (BINARY_INPUT)
    {
        for (auto &elem : elements)
            error = fread(elem.glo_num.data(), sizeof(long long), elem.num_points, file_ptr);
    }
    else
    {
        for (auto &elem : elements)
            for (int v = 0; v < elem.num_points; v++)
                error = fscanf(file_ptr, "%lld", &(elem.glo_num[v]));
    }

    fclose(file_ptr);

    for (auto &elem : elements)
        for (int v = 0; v < elem.num_points; v++)
            elem.loc_num[v] = elem.offset + v;

    // Node degree
    std::vector<int> node_degree(num_local_points);
    sprintf(file_name, "%s/lx1_%d/node_degree_%d.%d.dat", directory, poly_degree + 1, proc_id, poly_degree);
    file_ptr = fopen(file_name, (BINARY_INPUT) ? "rb" : "r");

    if (BINARY_INPUT)
        error = fread(node_degree.data(), sizeof(int), num_local_points, file_ptr);
    else
        for (int idx = 0; idx < num_local_points; idx++) error = fscanf(file_ptr, "%d", node_degree.data() + idx);

    fclose(file_ptr);

    // Dirichlet boundary conditions
    sprintf(file_name, "%s/lx1_%d/p_mask_%d.%d.dat", directory, poly_degree + 1, proc_id, poly_degree);
    file_ptr = fopen(file_name, (BINARY_INPUT) ? "rb" : "r");

    if (BINARY_INPUT)
    {
        for (auto &elem : elements)
            error = fread(elem.dirichlet_mask.data(), sizeof(DType), elem.num_points, file_ptr);
    }
    else
    {
        for (auto &elem : elements)
            for (int v = 0; v < elem.num_points; v++)
                error = fscanf(file_ptr, format, &(elem.dirichlet_mask[v]));
    }

    fclose(file_ptr);

    for (auto &elem : elements) memcpy(work_hst[0].data() + elem.offset, elem.dirichlet_mask.data(), elem.num_points * sizeof(DType));
    dirichlet_mask = device.malloc<DType>(num_local_points);
    dirichlet_mask.copyFrom(work_hst[0].data(), num_local_points * sizeof(DType));

    // Geometric factors
    for (int g = 0; g < NUM_GEOM_FACTS; g++)
    {
        sprintf(file_name, "%s/lx1_%d/g_%d_%d.%d.dat", directory, poly_degree + 1, g + 1, proc_id, poly_degree);
        file_ptr = fopen(file_name, (BINARY_INPUT) ? "rb" : "r");

        if (BINARY_INPUT)
        {
            for (auto &elem : elements)
                error = fread(elem.geom_fact[g].data(), sizeof(DType), elem.num_points, file_ptr);
        }
        else
        {
            for (auto &elem : elements)
                for (int v = 0; v < elem.num_points; v++)
                    error = fscanf(file_ptr, format, &(elem.geom_fact[g][v]));
        }

        fclose(file_ptr);

        for (auto &elem : elements) memcpy(work_hst[0].data() + elem.offset, elem.geom_fact[g].data(), elem.num_points * sizeof(DType));
        geom_fact[g] = device.malloc<DType>(num_local_points);
        geom_fact[g].copyFrom(work_hst[0].data(), num_local_points * sizeof(DType));
    }

    std::vector<DType*> geom_fact_ptr_hst(NUM_GEOM_FACTS);
    for (int g = 0; g < NUM_GEOM_FACTS; g++) geom_fact_ptr_hst[g] = (DType*)(geom_fact[g].ptr());
    geom_fact_ptr = device.malloc<DType*>(NUM_GEOM_FACTS);
    geom_fact_ptr.copyFrom(geom_fact_ptr_hst.data(), NUM_GEOM_FACTS * sizeof(DType*));

    // Done reading data
    if (error <= 0)
    {
        pstdout("ERROR: There was a problem reading Nek5000 data\n");
        quit();
    }

    // Communication
    if (proc_id == 0) printf("Setting up domain stitching handle...\n");

    std::unordered_map<long long, int> local_node_degree;

    for (auto &elem : elements)
    {
        for (int v = 0; v < elem.num_points; v++)
        {
            if (local_node_degree.find(elem.glo_num[v]) == local_node_degree.end())
                local_node_degree[elem.glo_num[v]] = 1;
            else
                local_node_degree[elem.glo_num[v]]++;
        }
    }

    std::unordered_map<long long, int> local_node_idx;
    std::vector<long long> boundary_nodes(num_local_points);
    int count = 0;

    for (auto &elem : elements)
    {
        for (int v = 0; v < elem.num_points; v++)
        {
            if (local_node_degree[elem.glo_num[v]] != node_degree[elem.offset + v])
            {
                if (local_node_idx.find(elem.glo_num[v]) == local_node_idx.end())
                {
                    boundary_nodes[count] = elem.glo_num[v];
                    local_node_idx[elem.glo_num[v]] = count;
                    count++;
                }
            }
        }
    }

    num_bdary_nodes = count;

    for (auto &elem : elements)
    {
        for (int v = 0; v < elem.num_points; v++)
        {
            if (local_node_idx.find(elem.glo_num[v]) == local_node_idx.end())
            {
                local_node_idx[elem.glo_num[v]] = count;
                count++;
            }
        }
    }

    comm_init(&gs_comm, MPI_COMM_WORLD);
    gs_handle = gslib_gs_setup(boundary_nodes.data(), num_bdary_nodes, &gs_comm, 0, gs_auto, 1);

    num_local_nodes = local_node_degree.size();
    Q.initialize(num_local_points, num_local_nodes);

    for (auto &elem : elements)
        for (int v = 0; v < elem.num_points; v++)
            Q.add_entry(elem.loc_num[v], local_node_idx[elem.glo_num[v]], 1.0);

    Q.assemble();
    Q.transpose(Qt);

    assembled_weight = device.malloc<DType>(num_local_nodes);
    math.set_to_value(work_dev[0], 1.0, num_local_points);
    Qt.multiply(assembled_weight, work_dev[0]);
    assembled_weight.copyTo(work_hst[0].data(), num_bdary_nodes * sizeof(DType));
    gslib_gs(work_hst[0].data(), gs_type, gs_add, 0, gs_handle, NULL);
    assembled_weight.copyFrom(work_hst[0].data(), num_bdary_nodes * sizeof(DType));
    math.invert_vector_elements(assembled_weight, num_local_nodes);

    // Operator
    int num_gll_points = poly_degree + 1;
    std::vector<double> r_gll(num_gll_points);
    std::vector<double> w_gll(num_gll_points);
    std::vector<double> D_gll(num_gll_points * num_gll_points);
    std::vector<double> Dt_gll(num_gll_points * num_gll_points);

    zwgll_(r_gll.data(), w_gll.data(), &num_gll_points);
    dgll_(Dt_gll.data(), D_gll.data(), r_gll.data(), &num_gll_points, &num_gll_points);

    for (int ij = 0; ij < num_gll_points * num_gll_points; ij++) ((DType*)(work_hst[0].data()))[ij] = (DType)(D_gll[ij]);
    D_hat = device.malloc<DType>(num_gll_points * num_gll_points);
    D_hat.copyFrom(work_hst[0].data(), num_gll_points * num_gll_points * sizeof(DType));

    // Solver
    r_k = device.malloc<DType>(num_local_points);
    r_kp1 = device.malloc<DType>(num_local_points);
    q_k = device.malloc<DType>(num_local_points);
    z_k = device.malloc<DType>(num_local_points);
    p_k = device.malloc<DType>(num_local_points);

    V.resize(num_vectors + 1); for (int i = 0; i < num_vectors + 1; i++) V[i] = device.malloc<DType>(num_local_points);
    Z.resize(num_vectors); for (int i = 0; i < num_vectors; i++) Z[i] = device.malloc<DType>(num_local_points);
    H.resize(num_vectors); for (int i = 0; i < num_vectors; i++) H[i].resize(num_vectors);
    c_gmres.resize(num_vectors);
    s_gmres.resize(num_vectors);
    gamma.resize(num_vectors + 1);

    // Kernels
    occa::properties properties;

    num_blocks = (num_local_points + BLOCK_SIZE - 1) / BLOCK_SIZE;

    properties["defines/DType"] = data_type;
    properties["defines/DIM"] = dim;
    properties["defines/OCCA_TYPE"] = OCCA_TYPE;
    properties["defines/BLOCK_SIZE"] = BLOCK_SIZE;

    if (proc_id == 0)
    {
        stiffness_matrix_1_kernel = device.buildKernel("domain.okl", "stiffness_matrix_1", properties);
        stiffness_matrix_2_kernel = device.buildKernel("domain.okl", "stiffness_matrix_2", properties);
        initialize_arrays_kernel = device.buildKernel("domain.okl", "initialize_arrays", properties);
        residual_norm_kernel = device.buildKernel("domain.okl", "residual_norm", properties);
        projection_inner_products_kernel = device.buildKernel("domain.okl", "projection_inner_products", properties);
        solution_and_residual_update_kernel = device.buildKernel("domain.okl", "solution_and_residual_update", properties);
        inner_product_flexible_kernel = device.buildKernel("domain.okl", "inner_product_flexible", properties);
        residual_and_search_update_kernel = device.buildKernel("domain.okl", "residual_and_search_update", properties);
        inner_product_kernel = device.buildKernel("domain.okl", "inner_product", properties);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (proc_id > 0)
    {
        stiffness_matrix_1_kernel = device.buildKernel("domain.okl", "stiffness_matrix_1", properties);
        stiffness_matrix_2_kernel = device.buildKernel("domain.okl", "stiffness_matrix_2", properties);
        initialize_arrays_kernel = device.buildKernel("domain.okl", "initialize_arrays", properties);
        residual_norm_kernel = device.buildKernel("domain.okl", "residual_norm", properties);
        projection_inner_products_kernel = device.buildKernel("domain.okl", "projection_inner_products", properties);
        solution_and_residual_update_kernel = device.buildKernel("domain.okl", "solution_and_residual_update", properties);
        inner_product_flexible_kernel = device.buildKernel("domain.okl", "inner_product_flexible", properties);
        residual_and_search_update_kernel = device.buildKernel("domain.okl", "residual_and_search_update", properties);
        inner_product_kernel = device.buildKernel("domain.okl", "inner_product", properties);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Visit output
template<typename DType>
void Domain<DType>::output(std::string output_name, int num_fields, ...)
{
    // Silo database
    DBSetDeprecateWarnings(0);
    DBfile *silo_file = NULL;
    char silo_name[80];

    if (proc_id == 0)
    {
        sprintf(silo_name, "%s.silo", output_name.c_str());
        silo_file = DBCreate(silo_name, DB_CLOBBER, DB_LOCAL, "Field data", DB_PDB);
    }

    if ((silo_file == NULL) and (proc_id == 0))
    {
        printf("ERROR: Couldn't create Silo file\n");

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Mesh
    int num_vertices = (dim == 2) ? 4 : 8;
    int num_low_order_elems = num_total_elements * std::pow(poly_degree, dim);
    int num_low_order_points = num_low_order_elems * num_vertices;

    std::vector<int> low_order_elements(num_low_order_points);
    int offset = 0;

    if (proc_id == 0)
    {
        if (dim == 2)
        {
            int n_x = poly_degree + 1;

            for (int e = 0; e < num_total_elements; e++)
            {
                for (int s_y = 0; s_y < poly_degree; s_y++)
                {
                    for (int s_x = 0; s_x < poly_degree; s_x++)
                    {
                        low_order_elements[offset++] = e * num_elem_points + (s_x + 0) + (s_y + 0) * n_x;
                        low_order_elements[offset++] = e * num_elem_points + (s_x + 1) + (s_y + 0) * n_x;
                        low_order_elements[offset++] = e * num_elem_points + (s_x + 1) + (s_y + 1) * n_x;
                        low_order_elements[offset++] = e * num_elem_points + (s_x + 0) + (s_y + 1) * n_x;
                    }
                }
            }
        }
        else
        {
            int n_x = poly_degree + 1;
            int n_xy = n_x * n_x;

            for (int e = 0; e < num_total_elements; e++)
            {
                for (int s_z = 0; s_z < poly_degree; s_z++)
                {
                    for (int s_y = 0; s_y < poly_degree; s_y++)
                    {
                        for (int s_x = 0; s_x < poly_degree; s_x++)
                        {
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 0) + (s_y + 0) * n_x + (s_z + 0) * n_xy;
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 1) + (s_y + 0) * n_x + (s_z + 0) * n_xy;
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 1) + (s_y + 1) * n_x + (s_z + 0) * n_xy;
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 0) + (s_y + 1) * n_x + (s_z + 0) * n_xy;
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 0) + (s_y + 0) * n_x + (s_z + 1) * n_xy;
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 1) + (s_y + 0) * n_x + (s_z + 1) * n_xy;
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 1) + (s_y + 1) * n_x + (s_z + 1) * n_xy;
                            low_order_elements[offset++] = e * num_elem_points + (s_x + 0) + (s_y + 1) * n_x + (s_z + 1) * n_xy;
                        }
                    }
                }
            }
        }
    }

    std::vector<DType> x_total, y_total, z_total;

    if (proc_id == 0)
    {
        if (dim >= 1) x_total.resize(num_total_points);
        if (dim >= 2) y_total.resize(num_total_points);
        if (dim >= 3) z_total.resize(num_total_points);
    }

    std::vector<int> proc_offset(num_procs);
    std::vector<int> proc_count(num_procs);
    proc_count[proc_id] = num_local_points;

    MPI_Gather(&num_local_points, 1, MPI_INT, proc_count.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (proc_id == 0) for (int p = 1; p < num_procs; p++) proc_offset[p] = proc_offset[p - 1] + proc_count[p - 1];

    if (dim >= 1)
    {
        for (auto &elem : elements) memcpy(work_hst[0].data() + elem.offset, elem.x.data(), elem.num_points * sizeof(DType));

        MPI_Gatherv(work_hst[0].data(), num_local_points, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, x_total.data(), proc_count.data(), proc_offset.data(), (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (dim >= 2)
    {
        for (auto &elem : elements) memcpy(work_hst[0].data() + elem.offset, elem.y.data(), elem.num_points * sizeof(DType));

        MPI_Gatherv(work_hst[0].data(), num_local_points, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, y_total.data(), proc_count.data(), proc_offset.data(), (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    if (dim >= 3)
    {
        for (auto &elem : elements) memcpy(work_hst[0].data() + elem.offset, elem.z.data(), elem.num_points * sizeof(DType));

        MPI_Gatherv(work_hst[0].data(), num_local_points, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, z_total.data(), proc_count.data(), proc_offset.data(), (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    DType *coordinates[] = { x_total.data(), y_total.data(), z_total.data() };

    if (proc_id == 0)
    {
        DBPutZonelist(silo_file, "elements", num_low_order_elems, dim, low_order_elements.data(), num_low_order_points, 0, &num_vertices, &num_low_order_elems, 1);
        DBPutUcdmesh(silo_file, "mesh", dim, NULL, coordinates, num_total_points, num_low_order_elems, "elements", NULL, (typeid(DType) == typeid(double)) ? DB_DOUBLE : DB_FLOAT, NULL);
    }

    // Fields
    char *field_name;
    occa::memory field_data;
    std::vector<DType> field_data_total;

    if (proc_id == 0) field_data_total.resize(num_total_points);

    va_list args;
    va_start(args, num_fields);

    for (int field = 0; field < num_fields; field++)
    {
        field_name = va_arg(args, char*);
        field_data = va_arg(args, occa::memory);
        field_data.copyTo(work_hst[0].data(), num_local_points * sizeof(DType));

        MPI_Gatherv(work_hst[0].data(), num_local_points, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, field_data_total.data(), proc_count.data(), proc_offset.data(), (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (proc_id == 0) DBPutUcdvar1(silo_file, field_name, "mesh", field_data_total.data(), num_total_points, NULL, 0, (typeid(DType) == typeid(double)) ? DB_DOUBLE : DB_FLOAT, DB_NODECENT, NULL);
    }

    va_end(args);

    // Free data
    if (proc_id == 0) DBClose(silo_file);

    MPI_Barrier(MPI_COMM_WORLD);
}

// Member functions
template<typename DType>
void Domain<DType>::initial_function(occa::memory &u, int function_id)
{
    // Set function
    if (dim == 2)
    {
        for (auto &elem : elements)
        {
            for (int v = 0; v < elem.num_points; v++)
            {
                if (function_id == 0)
                    work_hst[0][elem.loc_num[v]] = sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]);

                else if (function_id == 1)
                    work_hst[0][elem.loc_num[v]] = sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]) + sin(2.0 * M_PI * elem.x[v]) * sin(M_PI * elem.y[v]);

                else if (function_id == 2)
                    work_hst[0][elem.loc_num[v]] = exp(elem.x[v]) * sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]);

                else if (function_id == 3)
                    work_hst[0][elem.loc_num[v]] = sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]) + (1.0 / 5.0) * ((DType)(rand()) / (DType)(RAND_MAX));

                else if (function_id == 4)
                    work_hst[0][elem.loc_num[v]] = (DType)(rand()) / (DType)(RAND_MAX);
            }
        }
    }
    else
    {
        for (auto &elem : elements)
        {
            for (int v = 0; v < elem.num_points; v++)
            {
                if (function_id == 0)
                    work_hst[0][elem.loc_num[v]] = sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]) * sin(M_PI * elem.z[v]);

                else if (function_id == 1)
                    work_hst[0][elem.loc_num[v]] = sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]) * sin(M_PI * elem.z[v]) + sin(2.0 * M_PI * elem.x[v]) * sin(M_PI * elem.y[v]) * sin(M_PI * elem.z[v]);

                else if (function_id == 2)
                    work_hst[0][elem.loc_num[v]] = exp(elem.x[v]) * sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]) * sin(M_PI * elem.z[v]);

                else if (function_id == 3)
                    work_hst[0][elem.loc_num[v]] = sin(M_PI * elem.x[v]) * sin(M_PI * elem.y[v]) * sin(M_PI * elem.z[v]) + (1.0 / 5.0) * ((DType)(rand()) / (DType)(RAND_MAX));

                else if (function_id == 4)
                    work_hst[0][elem.loc_num[v]] = (DType)(rand()) / (DType)(RAND_MAX);
            }
        }
    }

    u.copyFrom(work_hst[0].data(), num_local_points * sizeof(DType));
    direct_stiffness_summation(u, u, true, true);
}

template<typename DType>
void Domain<DType>::direct_stiffness_summation(occa::memory &QQtu, occa::memory &u, bool apply_dirichlet_mask, bool apply_assembled_weight)
{
    if (apply_assembled_weight)
        Qt.multiply_weight(work_dev[0], u, assembled_weight);
    else
        Qt.multiply(work_dev[0], u);

    work_dev[0].copyTo(work_hst[0].data(), num_bdary_nodes * sizeof(DType));

    gslib_gs(work_hst[0].data(), gs_type, gs_add, 0, gs_handle, NULL);

    work_dev[0].copyFrom(work_hst[0].data(), num_bdary_nodes * sizeof(DType));

    if (apply_dirichlet_mask)
        Q.multiply_weight(QQtu, work_dev[0], dirichlet_mask);
    else
        Q.multiply(QQtu, work_dev[0]);
}

template<typename DType>
void Domain<DType>::stiffness_matrix(occa::memory &Au, occa::memory &u, bool apply_dssum)
{
    stiffness_matrix_1_kernel(work_dev_ptr, u, D_hat, geom_fact_ptr, num_local_points, poly_degree);
    stiffness_matrix_2_kernel(Au, work_dev_ptr, D_hat, num_local_points, poly_degree);

    if (apply_dssum) direct_stiffness_summation(Au, Au, true, false);
}

template<typename DType>
template<typename PType>
void Domain<DType>::flexible_conjugate_gradient(occa::memory &u, occa::memory &f, PType &subdomain, bool use_relative)
{
    // Initialize arrays
    timer.start("domain.vector_operations");
    occa::memory &u_k = u;
    initialize_arrays_kernel(u_k, r_k, f, num_local_points);
    timer.stop("domain.vector_operations");

    // Compute initial residual
    DType r_norm;
    DType r_0_norm;

    timer.start("domain.residual_norm");
    residual_norm(r_0_norm, r_k);
    timer.stop("domain.residual_norm");

    rstdout("Iter %2d: | residual_norm = %24.16g | relative_residual_norm = %24.16g | \n", 0, r_0_norm, 1.0);

    // Iterative solver
    DType alpha_k;
    DType beta_k;
    DType gamma_k;
    DType theta_k;

    if (use_preconditioner)
    {
        if (preconditioner_type == 0)
            subdomain.flexible_conjugate_gradient(z_k, r_k);
        else
            subdomain.generalized_minimum_residual(z_k, r_k);

        timer.start("subdomain.stitching");
        direct_stiffness_summation(z_k, z_k, true, true);
        timer.stop("subdomain.stitching");
    }
    else
    {
        direct_stiffness_summation(z_k, r_k);
    }

    timer.start("domain.vector_operations");
    p_k.copyFrom(z_k, num_local_points * sizeof(DType));
    timer.stop("domain.vector_operations");

    num_iterations = 0;

    for (int iter = 0; iter < max_iterations; iter++)
    {
        // Projection
        timer.start("domain.operator_application");
        stiffness_matrix(q_k, p_k);
        timer.stop("domain.operator_application");

        // Inner products
        timer.start("domain.inner_products");
        projection_inner_products(gamma_k, theta_k, z_k, r_k, p_k, q_k);
        timer.stop("domain.inner_products");

        alpha_k = gamma_k / theta_k;

        // Update solution and residual
        timer.start("domain.vector_operations");
        solution_and_residual_update(u_k, r_kp1, r_k, p_k, q_k, alpha_k);
        timer.stop("domain.vector_operations");

        // Residual norm
        timer.start("domain.residual_norm");
        residual_norm(r_norm, r_kp1);
        timer.stop("domain.residual_norm");

        rstdout("Iter %2d: | residual_norm = %24.16g | relative_residual_norm = %24.16g | \n", iter + 1, r_norm, r_norm / r_0_norm);

        if (use_relative)
        {
            if (r_norm / r_0_norm < tolerance) break;
        }
        else
        {
            if (r_norm < tolerance) break;
        }

        if (std::isnan(r_norm)) break;

        // Update search direction
        if (use_preconditioner)
        {
            if (preconditioner_type == 0)
                subdomain.flexible_conjugate_gradient(z_k, r_kp1);
            else
                subdomain.generalized_minimum_residual(z_k, r_kp1);

            timer.start("subdomain.stitching");
            direct_stiffness_summation(z_k, z_k, true, true);
            timer.stop("subdomain.stitching");
        }
        else
        {
            direct_stiffness_summation(z_k, r_kp1);
        }

        timer.start("domain.inner_products");
        inner_product_flexible(theta_k, r_k, r_kp1, z_k);
        timer.stop("domain.inner_products");

        beta_k = theta_k / gamma_k;

        timer.start("domain.vector_operations");
        residual_and_search_update_kernel(p_k, r_k, z_k, r_kp1, beta_k, num_local_points);
        timer.stop("domain.vector_operations");

        num_iterations++;
    }
}

template<typename DType>
template<typename PType>
void Domain<DType>::generalized_minimum_residual(occa::memory &u, occa::memory &f, PType &subdomain, bool use_relative)
{
    // Initialize arrays
    timer.start("domain.vector_operations");
    occa::memory &u_k = u;
    initialize_arrays_kernel(u_k, r_k, f, num_local_points);
    timer.stop("domain.vector_operations");

    // Compute initial residual
    DType r_norm;
    DType r_0_norm;

    timer.start("domain.residual_norm");
    residual_norm(r_0_norm, r_k);
    timer.stop("domain.residual_norm");

    rstdout("Iter %2d: | residual_norm = %24.16g | relative_residual_norm = %24.16g | \n", 0, r_0_norm, 1.0);

    // Iterative solver
    bool converged = false;
    int iter = 0;
    int outer = 0;
    int j;

    DType alpha_j;
    DType beta_j;
    DType gamma_j;
    DType gamma_k;

    while (iter < max_iterations)
    {
        if (iter > 0)
        {
            timer.start("domain.operator_application");
            stiffness_matrix(r_k, u_k);
            timer.stop("domain.operator_application");

            timer.start("domain.vector_operations");
            math.vector_vector_addition(r_k, 1.0, f, - 1.0, r_k, num_local_points);
            timer.stop("domain.vector_operations");

            timer.start("domain.residual_norm");
            residual_norm(r_norm, r_k);
            timer.stop("domain.residual_norm");

            gamma[0] = r_norm;
        }
        else
        {
            gamma[0] = r_0_norm;
        }

        timer.start("domain.vector_operations");
        math.vector_scaling(V[0], 1.0 / gamma[0], r_k, num_local_points);
        timer.stop("domain.vector_operations");

        for (j = 0; j < num_vectors; j++)
        {
            if (use_preconditioner)
            {
                if (preconditioner_type == 0)
                    subdomain.flexible_conjugate_gradient(Z[j], V[j]);
                else
                    subdomain.generalized_minimum_residual(Z[j], V[j]);

                timer.start("subdomain.stitching");
                direct_stiffness_summation(Z[j], Z[j], true, true);
                timer.stop("subdomain.stitching");
            }
            else
            {
                timer.start("subdomain.preconditioner");
                direct_stiffness_summation(Z[j], V[j]);
                timer.stop("subdomain.preconditioner");
            }

            timer.start("domain.operator_application");
            stiffness_matrix(q_k, Z[j]);
            timer.stop("domain.operator_application");

            // 2-pass Gram-Schmidt (1st pass)
            for (int i = 0; i < j + 1; i++)
            {
                timer.start("domain.inner_products");
                assembled_inner_product(H[i][j], q_k, V[i]);
                timer.stop("domain.inner_products");
            }

            for (int i = 0; i < j + 1; i++)
            {
                timer.start("domain.vector_operations");
                math.vector_vector_addition(q_k, 1.0, q_k, - H[i][j], V[i], num_local_points);
                timer.stop("domain.vector_operations");
            }

            // Apply Given's rotation to new column
            for (int i = 0; i < j; i++)
            {
                DType h_ij = H[i][j];
                H[i][j] = c_gmres[i] * h_ij + s_gmres[i] * H[i + 1][j];
                H[i + 1][j] = - s_gmres[i] * h_ij + c_gmres[i] * H[i + 1][j];
            }

            timer.start("domain.residual_norm");
            residual_norm(alpha_j, q_k);
            timer.stop("domain.residual_norm");

            if (std::abs(alpha_j) == 0.0)
            {
                converged = true;
                break;
            }

            beta_j = std::sqrt(H[j][j] * H[j][j] + alpha_j * alpha_j);
            gamma_j = 1.0 / beta_j;
            c_gmres[j] = H[j][j] * gamma_j;
            s_gmres[j] = alpha_j * gamma_j;
            H[j][j] = beta_j;
            gamma[j + 1] = - s_gmres[j] * gamma[j];
            gamma[j] = c_gmres[j] * gamma[j];
    
            r_norm = std::abs(gamma[j + 1]);
            rstdout("Iter %2d: | residual_norm = %24.16g | relative_residual_norm = %24.16g | \n", iter + 1, r_norm, r_norm / r_0_norm);

            if (use_relative)
            {
                if (r_norm / r_0_norm < tolerance)
                {
                    converged = true;
                    break;
                }
            }
            else
            {
                if (r_norm < tolerance)
                {
                    converged = true;
                    break;
                }
            }

            if (iter >= max_iterations)
            {
                converged = true;
                break;
            }

            if (std::isnan(r_norm))
            {
                converged = true;
                break;
            }

            timer.start("domain.vector_operations");
            math.vector_scaling(V[j + 1], 1.0 / alpha_j, q_k, num_local_points);
            timer.stop("domain.vector_operations");

            iter++;
        }

        if (j == num_vectors) j--;

        for (int k = j; k >= 0; k--)
        {
            gamma_k = gamma[k];

            for (int i = j; i > k; i--)
                gamma_k -= H[k][i] * c_gmres[i];

            c_gmres[k] = gamma_k / H[k][k];
        }

        // Sum Arnoldi vectors
        for (int i = 0; i < j + 1; i++)
        {
            timer.start("domain.vector_operations");
            math.vector_vector_addition(u_k, 1.0, u_k, c_gmres[i], Z[i], num_local_points);
            timer.stop("domain.vector_operations");
        }

        if (converged) break;
        outer++;
    }

    num_iterations = iter;
}

template<typename DType>
void Domain<DType>::residual_norm(DType &r_norm, occa::memory &r)
{
    r_norm = 0.0;

    direct_stiffness_summation(work_dev[1], r);
    residual_norm_kernel(work_dev[0], r, work_dev[1], dirichlet_mask, num_local_points, num_blocks);

    work_dev[0].copyTo(work_hst[0].data(), num_blocks * sizeof(DType));

    for (int b = 0; b < num_blocks; b++) r_norm += work_hst[0][b];

    // Reduce globally
    MPI_Allreduce(MPI_IN_PLACE, &r_norm, 1, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    r_norm = std::sqrt(r_norm);
}

template<typename DType>
void Domain<DType>::assembled_inner_product(DType &uv, occa::memory &u, occa::memory &v)
{
    uv = 0.0;

    direct_stiffness_summation(work_dev[1], v);
    inner_product_kernel(work_dev[0], u, work_dev[1], dirichlet_mask, num_local_points, num_blocks);

    work_dev[0].copyTo(work_hst[0].data(), num_blocks * sizeof(DType));

    for (int b = 0; b < num_blocks; b++) uv += work_hst[0][b];

    // Reduce globally
    MPI_Allreduce(MPI_IN_PLACE, &uv, 1, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

template<typename DType>
void Domain<DType>::projection_inner_products(DType &gamma_k, DType &theta_k, occa::memory &z_k, occa::memory &r_k, occa::memory &p_k, occa::memory &q_k)
{
    // Reduce locally
    gamma_k = 0.0;
    theta_k = 0.0;

    projection_inner_products_kernel(work_dev[0], z_k, r_k, p_k, q_k, num_local_points, num_blocks);

    work_dev[0].copyTo(work_hst[0].data(), (2 * num_blocks) * sizeof(DType));

    for (int b = 0; b < num_blocks; b++)
    {
        gamma_k += work_hst[0][b];
        theta_k += work_hst[0][b + num_blocks];
    }

    // Reduce globally
    DType values[2] = { gamma_k, theta_k};

    MPI_Allreduce(MPI_IN_PLACE, &values, 2, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    gamma_k = values[0];
    theta_k = values[1];
}

template<typename DType>
void Domain<DType>::solution_and_residual_update(occa::memory &u_k, occa::memory &r_kp1, occa::memory &r_k, occa::memory &p_k, occa::memory &q_k, DType alpha_k)
{
    solution_and_residual_update_kernel(u_k, r_kp1, r_k, p_k, q_k, alpha_k, num_local_points);
}

template<typename DType>
void Domain<DType>::inner_product_flexible(DType &theta_k, occa::memory &r_k, occa::memory &r_kp1, occa::memory &z_k)
{
    // Reducte locally
    theta_k = 0.0;

    inner_product_flexible_kernel(work_dev[0], r_k, r_kp1, z_k, num_local_points, num_blocks);

    work_dev[0].copyTo(work_hst[0].data(), num_blocks * sizeof(DType));

    for (int b = 0; b < num_blocks; b++)
        theta_k += work_hst[0][b];

    // Reduce globally
    MPI_Allreduce(MPI_IN_PLACE, &theta_k, 1, (typeid(DType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

template<typename DType>
void Domain<DType>::residual_and_search_update(occa::memory &p_k, occa::memory &r_k, occa::memory &z_k, occa::memory &r_kp1, DType beta_k)
{
    residual_and_search_update_kernel(p_k, r_k, z_k, r_kp1, beta_k, num_local_points);
}
