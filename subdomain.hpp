/*
 * Subdomain header file
 */

// Headers
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
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

extern "C"
{
    #include "_hypre_utilities.h"
    #include "HYPRE_parcsr_ls.h"
    #include "_hypre_parcsr_ls.h"
    #include "_hypre_IJ_mv.h"
    #include "HYPRE.h"
}

#include "domain.hpp"
#include "csr_matrix.hpp"
#include "math.hpp"
#include "timer.hpp"
#include "AMG/vector.hpp"
#include "AMG/csr_matrix.hpp"

// Class declaration
#ifndef SUBDOMAIN_HPP
#define SUBDOMAIN_HPP

template<typename DType>
struct Stiffness_Operator
{
    int num_dofs = 0;
    int num_points = 0;
    int num_extended_dofs = 0;

    CSR_Matrix<DType> Q;
    CSR_Matrix<DType> Qt;

    CSR_Matrix<DType> A;
    CSR_Matrix<DType> P;
    CSR_Matrix<DType> Pt;

    std::vector<occa::memory> D_hat;
    occa::memory D_hat_ptr;

    occa::memory geom_fact[NUM_GEOM_FACTS];
    occa::memory geom_fact_ptr;

    occa::memory element;
    occa::memory vertex;
    occa::memory level;
    occa::memory offset;
};

template<typename DType>
class Subdomain
{
    private:
        // Work arrays
        std::vector<std::vector<DType>> work_hst;
        std::vector<occa::memory> work_dev;
        occa::memory work_dev_ptr;

        // Geometry
        int poly_reduction;
        int subdomain_overlap;
        int superdomain_overlap;

        std::vector<int> poly_degree;
        int num_levels;

        struct Level
        {
            int num_points;
            int num_elements;
            int poly_degree;
            int offset;
        };

        std::vector<Level> levels;

        // Coarse to fine interpolator
        std::map<std::pair<int, int>, std::pair<std::vector<DType>, occa::memory>> J_cf;
        occa::memory J_cf_ptr;

        // Gather-scatter data
        std::vector<int> proc_count;
        std::vector<int> proc_offset;

        comm gs_comm;
        struct gs_data *gs_handle;
        gs_dom gs_type = (typeid(DType) == typeid(double)) ? gs_double : gs_float;

        // Reference operator
        std::vector<std::pair<std::vector<DType>, occa::memory>> D_hat;
        occa::memory D_hat_ptr;

        // Subdomain operator
        int num_subdomain_elems;
        int num_subdomain_points;
        int num_subdomain_extended_elems;
        int num_subdomain_extended_points;
        int subdomain_offset;
        Stiffness_Operator<DType> subdomain_operator;

        // Superdomain operator
        CSR_Matrix<DType> Qt_coarse;

        int num_superdomain_elems;
        int num_superdomain_points;
        int num_superdomain_extended_elems;
        int num_superdomain_extended_points;
        int superdomain_offset;
        Stiffness_Operator<DType> superdomain_operator;

        // Interface assembly
        int num_interface_dofs;
        CSR_Matrix<DType> Q_int;
        CSR_Matrix<DType> Qt_int;
        CSR_Matrix<DType> QQt_int;

        // Preconditioner
        int num_levels_fem;

        HYPRE_ParCSRMatrix A_fem_hst_csr;
        HYPRE_IJMatrix A_fem_hst;
        hypre_ParAMGData *amg_data;

        std::vector<amg::CSR_Matrix> A_fem;
        std::vector<amg::Vector> D_val_fem;
        std::vector<amg::Vector> coefs_fem;
        std::vector<amg::CSR_Matrix> P_fem;
        std::vector<amg::CSR_Matrix> R_fem;
        std::vector<amg::Vector> work_hst_fem;
        std::vector<amg::Vector> work_dev_fem;
        std::vector<amg::Vector> f_fem;
        std::vector<amg::Vector> u_fem;
        std::vector<amg::Vector> r_fem;
        std::vector<amg::Vector> v_fem;
        std::vector<amg::Vector> w_fem;

        cudaStream_t cuda_stream;
        cudaGraph_t down_leg_graph;
        cudaGraphExec_t down_leg_instance;
        cudaGraph_t up_leg_graph;
        cudaGraphExec_t up_leg_instance;

        void low_order_preconditioner(occa::memory&, occa::memory&);

        // Solver
        int num_dofs;
        int num_blocks;
        occa::memory norm_weight;
        occa::memory inner_weight;

        occa::memory f;
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

        void initialize_arrays(occa::memory&, occa::memory&, occa::memory&);
        void tree_operator(occa::memory&, occa::memory&);
        void residual_norm(DType&, occa::memory&);
        void inner_product(DType&, occa::memory&, occa::memory&);
        void projection_inner_products(DType&, DType&, occa::memory&, occa::memory&, occa::memory&, occa::memory&);
        void solution_and_residual_update(occa::memory&, occa::memory&, occa::memory&, occa::memory&, occa::memory&, DType);
        void search_update_inner_product(DType&, occa::memory&, occa::memory&, occa::memory&);
        void residual_and_search_update(occa::memory&, occa::memory&, occa::memory&, occa::memory&, DType);
        void assembled_inner_product(DType&, occa::memory&, occa::memory&);

        // Utility functions
        Math<DType> math;

        // Kernels
        occa::kernel copy_from_domain_data_kernel; 
        occa::kernel copy_to_domain_data_kernel;
        occa::kernel restriction_1_kernel;
        occa::kernel restriction_2_kernel;
        occa::kernel restriction_3_kernel;

        occa::kernel initialize_arrays_kernel;
        occa::kernel stiffness_matrix_1_kernel;
        occa::kernel stiffness_matrix_2_kernel;
        occa::kernel inner_product_kernel;
        occa::kernel weighted_inner_product_kernel;
        occa::kernel projection_inner_products_kernel;
        occa::kernel solution_and_residual_update_kernel;
        occa::kernel search_update_inner_product_kernel;
        occa::kernel residual_and_search_update_kernel;

    public:
        // Member variables
        const char *data_type = (typeid(DType) == typeid(double)) ? "double" : "float";

        // Constructor and destructor
        template<typename PType>
        Subdomain(std::unordered_map<int, PType>, int, int, int = 1, int = 1);
        ~Subdomain();

        // Solver
        int num_iterations = 0;
        int num_vectors = 4;
        int max_iterations = 4;
        bool use_preconditioner = true;
        DType tolerance = (typeid(DType) == typeid(double)) ? 1.0e-12 : 1.0e-06;
        DType epsilon = (typeid(DType) == typeid(double)) ? 1.0e-12 : 1.0e-06;

        // Preconditioner
        int num_vcycles = 1;
        int cheby_order = 2;
        int level_cutoff = 5;

        // Elements
        int num_values;
        std::vector<Element<DType>> elements;

        // Member functions
        void direct_stiffness_summation(occa::memory&, occa::memory&);
        void stiffness_matrix(occa::memory&, occa::memory&);
        void flexible_conjugate_gradient(occa::memory&, occa::memory&, bool = true, bool = false);
        void generalized_minimum_residual(occa::memory&, occa::memory&, bool = true, bool = false);

        // Visit output
        void output(std::string, int = 0, ...);
};

#include "subdomain.tpp"

#endif
