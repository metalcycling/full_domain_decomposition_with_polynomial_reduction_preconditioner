/*
 * Domain header file
 */

// Headers
#include "config.hpp"

extern "C"
{
    #define PREFIX gslib_
    #define MPI
    #define GLOBAL_LONG_LONG
    #include "name.h"
    #include "fail.h"
    #include "c99.h"
    #include "types.h"
    #include "comm.h"
    #include "gs_defs.h"
    #include "mem.h"
    #include "gs.h"
}

#include "element.hpp"
#include "csr_matrix.hpp"
#include "math.hpp"
#include "special_functions.hpp"
#include "timer.hpp"

// Class declaration
#ifndef DOMAIN_HPP
#define DOMAIN_HPP

template<typename DType>
class Domain
{
    private:
        // Work arrays
        std::vector<std::vector<DType>> work_hst;
        std::vector<occa::memory> work_dev;
        occa::memory work_dev_ptr;

        // Dirichlet boundary conditions
        occa::memory dirichlet_mask;

        // Assembly
        CSR_Matrix<DType> Q;
        CSR_Matrix<DType> Qt;
        occa::memory assembled_weight;

        // Gather scatter
        int num_bdary_nodes;
        struct comm gs_comm;
        struct gs_data *gs_handle;
        gs_dom gs_type = (typeid(DType) == typeid(double)) ? gs_double : gs_float;

        // Solver
        occa::memory u_k;
        occa::memory r_k;
        occa::memory r_kp1;
        occa::memory q_k;
        occa::memory z_k;
        occa::memory p_k;

        std::vector<occa::memory> V;
        std::vector<occa::memory> Z;
        std::vector<std::vector<DType>> H;
        std::vector<DType> c_gmres;
        std::vector<DType> s_gmres;
        std::vector<DType> gamma;

        void residual_norm(DType&, occa::memory&);
        void projection_inner_products(DType&, DType&, occa::memory&, occa::memory&, occa::memory&, occa::memory&);
        void solution_and_residual_update(occa::memory&, occa::memory&, occa::memory&, occa::memory&, occa::memory&, DType);
        void inner_product_flexible(DType&, occa::memory&, occa::memory&, occa::memory&);
        void residual_and_search_update(occa::memory&, occa::memory&, occa::memory&, occa::memory&, DType);
        void assembled_inner_product(DType&, occa::memory&, occa::memory&);

        // Utility functions
        Math<DType> math;

        // Kernels
        occa::kernel stiffness_matrix_1_kernel;
        occa::kernel stiffness_matrix_2_kernel;
        occa::kernel initialize_arrays_kernel;
        occa::kernel residual_norm_kernel;
        occa::kernel projection_inner_products_kernel;
        occa::kernel solution_and_residual_update_kernel;
        occa::kernel inner_product_flexible_kernel;
        occa::kernel residual_and_search_update_kernel;
        occa::kernel inner_product_kernel;

    public:
        // Member variables
        char *directory;
        int poly_degree;
        const char *data_type = (typeid(DType) == typeid(double)) ? "double" : "float";

        int num_total_elements;
        int num_total_points;
        int num_total_nodes;

        int num_local_elements;
        int num_local_points;
        int num_local_nodes;

        int num_elem_points;

        // Elements
        std::vector<Element<DType>> elements;

        // Solver
        int num_blocks;
        int num_iterations = 0;
        int num_vectors = 20;
        int max_iterations = 500;
        int preconditioner_type = 1;
        bool use_preconditioner = true;
        DType tolerance = (typeid(DType) == typeid(double)) ? 1.0e-07 : 1.0e-04;

        // Operator
        occa::memory D_hat;
        occa::memory geom_fact[NUM_GEOM_FACTS];
        occa::memory geom_fact_ptr;

        // Constructor and destructor
        Domain();
        Domain(char*, int);
        ~Domain();

        void initialize(char*, int);

        // Member functions
        void initial_function(occa::memory&, int = 0);
        void direct_stiffness_summation(occa::memory&, occa::memory&, bool = true, bool = false);
        void stiffness_matrix(occa::memory&, occa::memory&, bool = false);

        template<typename PType>
        void flexible_conjugate_gradient(occa::memory&, occa::memory&, PType&, bool = true);

        template<typename PType>
        void generalized_minimum_residual(occa::memory&, occa::memory&, PType&, bool = true);

        // Visit output
        void output(std::string, int = 0, ...);
};

#include "domain.tpp"

#endif
