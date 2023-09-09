/*
 * Poisson solver
 * - Full domain decomposition with polynomial reduction preconditioner
 * - High-order spectral elements
 * - FCG supported
 * - GPU enabled
 *
 * Creation date: 07/13/2021
 */

// Headers
#include "config.hpp"
#include "domain.hpp"
#include "subdomain.hpp"

#include <cuda_runtime.h>

extern "C"
{
    #include "_hypre_utilities.h"
    #include "HYPRE_parcsr_ls.h"
    #include "_hypre_parcsr_ls.h"
    #include "_hypre_IJ_mv.h"
    #include "HYPRE.h"
}

// Namespaces
using namespace std;

// Functions declaration
void MPI_Initialize(int, char*[]);
void HYPRE_Initialize();
void set_parallel_print();
void library_banner();
void OCCA_Initialize();
void run_simulation(char*, int, int, int, int);
void simulation_data();

// Main function
int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Initialize(argc, argv);

    // Initialize Hypre
    HYPRE_Initialize();

    // Set parallel output printing
    set_parallel_print();

    // Set timer
    timer.initialize();

    // Library message
    library_banner();

    // Initialize OCCA
    OCCA_Initialize();

    // Check parameters passed
    if (argc < 6)
    {
        rstdout("ERROR: Use as 'poisson <directory> <polynomial degree> <polynomial reduction> <subdomain overlap> <superdomain overlap>'\n");
        quit();
    }

    // Run simulation
    run_simulation(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));

    // Print output measurements
    simulation_data();

    // Finalize Hypre
    HYPRE_Finalize();

    // Finalize MPI
    MPI_Finalize();

    // End of program
    return EXIT_SUCCESS;
}

// Functions
void MPI_Initialize(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
}

void HYPRE_Initialize()
{
    cudaSetDevice(0);
    hypre_bind_device(proc_id, num_procs, hypre_MPI_COMM_WORLD);

    HYPRE_Init();
    HYPRE_PrintDeviceInfo();
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
    HYPRE_SetSpGemmUseCusparse(false);

    MPI_Barrier(MPI_COMM_WORLD);
}

void set_parallel_print()
{
#if PRINT == 1
    sprintf(pstdout_name, "pstdout_%d.dat", proc_id);
    pstdout_file = fopen(pstdout_name, "w");
#endif
}

void library_banner()
{
    rstdout("----------------------------------------------------------------------------------\n");
    rstdout("|                                                                                |\n");
    rstdout("|               .--~~,__                                                         |\n");
    rstdout("|  :-....,-------`~~'._.'                                                        |\n");
    rstdout("|   `-,,,  ,_      ;'~U'                                                         |\n");
    rstdout("|    _,-' ,'`-__; '--.     Full domain decomposition with polynomial reduction   |\n");
    rstdout("|   (_/'~~      ''''(;     By: Pedro D. Bello-Maldonado                          |\n");
    rstdout("|                                                                                |\n");
    rstdout("----------------------------------------------------------------------------------\n");
    rstdout("\n");
}

void OCCA_Initialize()
{
    rstdout("Running OCCA with:\n");

    // Device
#if OCCA_TYPE == 0
    device.setup({{"mode", "Serial"}});
    rstdout("- Mode: 'Serial'\n");

#elif OCCA_TYPE == 1
    device.setup({{"mode",  "CUDA"}, {"device_id", 0}});
    rstdout("- Mode: 'CUDA'\n");
    rstdout("- Device id: 0\n");

#else
    rstdout("OCCA mode '%d' is not available\n", OCCA_TYPE);
    quit();

#endif

    rstdout("\n");
}

void run_simulation(char *directory, int poly_degree, int poly_reduction, int subdomain_overlap, int superdomain_overlap)
{
    // Types
    typedef STYPE SType;
    typedef PTYPE PType;

    // Check input directory
    FILE *file_ptr = fopen(directory, "r");

    if (file_ptr == NULL)
    {
        rstdout("Directory '%s' does not exist. Make sure the directory has all the 'lx1' subdirectories", directory);
        quit();
    }

    fclose(file_ptr);

    // Opening message
    rstdout("Running simulation with:\n");
    rstdout("- Directory: \"%s\"\n", directory);
    rstdout("- Polynomial degree: \"%d\"\n", poly_degree);
    rstdout("- Polynomial reduction: \"%d\"\n", poly_reduction);
    rstdout("- Subdomain overlap: \"%d\"\n", subdomain_overlap);
    rstdout("- Superdomain overlap: \"%d\"\n", superdomain_overlap);

    // Create local domain
    std::unordered_map<int, Domain<SType>> domains;
    int poly_degree_level = poly_degree;

    rstdout("\nSetting up domain \"N = %d\" object...\n", poly_degree);
    domains[poly_degree].initialize(directory, poly_degree);
    rstdout("\n");

    while (poly_degree_level > 1)
    {
        poly_degree_level -= poly_reduction;

        if (poly_degree_level >= 1)
        {
            rstdout("Setting up domain \"N = %d\" object...\n", poly_degree_level);
            domains[poly_degree_level].initialize(directory, poly_degree_level);
        }
        else
        {
            rstdout("Setting up domain \"N = %d\" object...\n", 1);
            domains[1].initialize(directory, 1);
        }

        rstdout("\n");
    }

    Domain<SType> &domain = domains[poly_degree];

    // Setup preconditioner
    rstdout("Setting up subdomain object...\n");

    Subdomain<PType> subdomain(domains, poly_degree, poly_reduction, subdomain_overlap, superdomain_overlap);

    // Set exact solution
    rstdout("\nSetting up exact function...\n");

    int function_id = 4;
    occa::memory u_star = device.malloc<SType>(domain.num_local_points);
    domain.initial_function(u_star, function_id);

    // Construct right-hand-side
    rstdout("Setting up right-hand-side...\n");

    occa::memory f = device.malloc<SType>(domain.num_local_points);
    domain.stiffness_matrix(f, u_star);

    // Numerical solution
    rstdout("Solving Poisson problem...\n");

    int solver_id = 1;

    occa::memory u = device.malloc<SType>(domain.num_local_points);

    if (solver_id == 0)
        domain.flexible_conjugate_gradient(u, f, subdomain);
    else
        domain.generalized_minimum_residual(u, f, subdomain);

#if VISUALIZATION == 1
    domain.output("domain", 3, "u_star", u_star, "f", f, "u", u);
#endif

    rstdout("\nRun info:\n");
    rstdout("-------------------------------------------------------------------------\n");
    rstdout("Number of dimensions: %d\n", dim);
    rstdout("Total number of elements: %d\n", domain.num_total_elements);
    rstdout("Polynomial degree: %d\n", domain.poly_degree);
    rstdout("Function ID: %d\n", function_id);
    rstdout("Subdomain overlap: %d\n", subdomain_overlap);
    rstdout("Superdomain overlap: %d\n", superdomain_overlap);
    rstdout("Solver data precision: %s\n", domain.data_type);
    rstdout("Solver tolerance: %g\n", domain.tolerance);
    rstdout("Solver type: \"%s\"\n", (solver_id == 0) ? "FCG" : "GMRES");
    rstdout("Preconditioner data precision: %s\n", subdomain.data_type);
    rstdout("Preconditioner tolerance: %g\n", subdomain.tolerance);
    rstdout("Preconditioner type: \"%s\"\n", (domain.preconditioner_type == 0) ? "FCG" : "GMRES");
}

void simulation_data()
{
    typedef STYPE SType;
    const char *aggregation_type = "max";

    SType total_time = 0.0;
    SType inner_products_time = timer.total("domain.inner_products", aggregation_type);
    SType residual_norm_time = timer.total("domain.residual_norm", aggregation_type);
    SType vector_operations_time = timer.total("domain.vector_operations", aggregation_type);
    SType operator_application_time = timer.total("domain.operator_application", aggregation_type);
    SType subdomain_stitching_time = timer.total("subdomain.stitching", aggregation_type);
    SType tree_construction_time = 0.0;
    SType tree_exchange_time = 0.0;
    SType subdomain_solver_time = 0.0;

    std::string inner_products_string; timer.total("domain.inner_products", inner_products_string);
    std::string residual_norm_string; timer.total("domain.residual_norm", residual_norm_string);
    std::string vector_operations_string; timer.total("domain.vector_operations", vector_operations_string);
    std::string operator_application_string; timer.total("domain.operator_application", operator_application_string);
    std::string subdomain_stitching_string; timer.total("subdomain.stitching", subdomain_stitching_string);
    std::string tree_construction_string;
    std::string tree_exchange_string;
    std::string subdomain_solver_string;

    {
        // Subdomain solver time
        SType inner_products_time = timer.total("subdomain.inner_products");
        SType residual_norm_time = timer.total("subdomain.residual_norm");
        SType vector_operations_time = timer.total("subdomain.vector_operations");
        SType operator_application_time = timer.total("subdomain.operator_application");
        SType preconditioner_time = 0.0;

        preconditioner_time += timer.total("subdomain.preconditioner.assemble_subdomain");
        preconditioner_time += timer.total("subdomain.preconditioner.assemble_composite");
        preconditioner_time += timer.total("subdomain.preconditioner.memcpy");
        preconditioner_time += timer.total("subdomain.preconditioner.vector_operations");
        preconditioner_time += timer.total("subdomain.preconditioner.down_leg_gpu");
        preconditioner_time += timer.total("subdomain.preconditioner.coarse_grid_solver");
        preconditioner_time += timer.total("subdomain.preconditioner.up_leg_gpu");
        preconditioner_time += timer.total("subdomain.preconditioner.unassemble_subdomain");
        preconditioner_time += timer.total("subdomain.preconditioner.unassemble_composite");

        subdomain_solver_time += inner_products_time;
        subdomain_solver_time += residual_norm_time;
        subdomain_solver_time += vector_operations_time;
        subdomain_solver_time += operator_application_time;
        subdomain_solver_time += preconditioner_time;

        std::vector<SType> subdomain_solver_times(num_procs);
        subdomain_solver_times[proc_id] = subdomain_solver_time;

        MPI_Allreduce(MPI_IN_PLACE, subdomain_solver_times.data(), num_procs, (typeid(SType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        subdomain_solver_time = *std::max_element(subdomain_solver_times.begin(), subdomain_solver_times.end());

        char word[80];

        for (int p = 0; p < num_procs; p++)
        {
            if (p < num_procs - 1)
                sprintf(word, "%12.08f ", subdomain_solver_times[p]);
            else
                sprintf(word, "%12.08f", subdomain_solver_times[p]);

            subdomain_solver_string += word;
        }
    }

    {
        // Tree construction time
        SType gpu_to_gpu_time = timer.total("subdomain.tree_construction.gpu_to_gpu");
        SType subdomain_time = timer.total("subdomain.tree_construction.subdomain");
        SType gpu_to_cpu_time = timer.total("subdomain.tree_construction.gpu_to_cpu");
        SType assemble_coarse_time = timer.total("subdomain.tree_construction.assemble_coarse");
        SType superdomain_time = timer.total("subdomain.tree_construction.superdomain");

        tree_construction_time += gpu_to_gpu_time;
        tree_construction_time += subdomain_time;
        tree_construction_time += gpu_to_cpu_time;
        tree_construction_time += assemble_coarse_time;
        tree_construction_time += superdomain_time;

        std::vector<SType> tree_construction_times(num_procs);
        tree_construction_times[proc_id] = tree_construction_time;

        MPI_Allreduce(MPI_IN_PLACE, tree_construction_times.data(), num_procs, (typeid(SType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        tree_construction_time = *std::max_element(tree_construction_times.begin(), tree_construction_times.end());

        char word[80];

        for (int p = 0; p < num_procs; p++)
        {
            if (p < num_procs - 1)
                sprintf(word, "%12.08f ", tree_construction_times[p]);
            else
                sprintf(word, "%12.08f", tree_construction_times[p]);

            tree_construction_string += word;
        }
    }

    {
        // Tree exchange time
        SType superdomain_time = timer.total("subdomain.tree_exchange.superdomain");
        SType subdomain_time = timer.total("subdomain.tree_exchange.subdomain");
        SType cpu_to_gpu_time = timer.total("subdomain.tree_exchange.cpu_to_gpu");

        tree_exchange_time += superdomain_time;
        tree_exchange_time += subdomain_time;
        tree_exchange_time += cpu_to_gpu_time;

        std::vector<SType> tree_exchange_times(num_procs);
        tree_exchange_times[proc_id] = tree_exchange_time;

        MPI_Allreduce(MPI_IN_PLACE, tree_exchange_times.data(), num_procs, (typeid(SType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        tree_exchange_time = *std::max_element(tree_exchange_times.begin(), tree_exchange_times.end());

        char word[80];

        for (int p = 0; p < num_procs; p++)
        {
            if (p < num_procs - 1)
                sprintf(word, "%12.08f ", tree_exchange_times[p]);
            else
                sprintf(word, "%12.08f", tree_exchange_times[p]);

            tree_exchange_string += word;
        }
    }

    total_time += inner_products_time;
    total_time += residual_norm_time;
    total_time += vector_operations_time;
    total_time += operator_application_time;
    total_time += tree_construction_time;
    total_time += tree_exchange_time;
    total_time += subdomain_stitching_time;
    total_time += subdomain_solver_time;

    rstdout("\nTimings:\n");
    rstdout("-------------------------------------------------------------------------\n");
    rstdout("Total                 = %12.08f s ( %6.02f )\n", total_time, 100.0);
    rstdout("Inner products        = %12.08f s ( %6.02f ) [ %s ]\n", inner_products_time, 100.0 * inner_products_time / total_time, inner_products_string.c_str());
    rstdout("Residual norm         = %12.08f s ( %6.02f ) [ %s ]\n", residual_norm_time, 100.0 * residual_norm_time / total_time, residual_norm_string.c_str());
    rstdout("Vector operations     = %12.08f s ( %6.02f ) [ %s ]\n", vector_operations_time, 100.0 * vector_operations_time / total_time, vector_operations_string.c_str());
    rstdout("Operator application  = %12.08f s ( %6.02f ) [ %s ]\n", operator_application_time, 100.0 * operator_application_time / total_time, operator_application_string.c_str());
    rstdout("Tree construction     = %12.08f s ( %6.02f ) [ %s ]\n", tree_construction_time, 100.0 * tree_construction_time / total_time, tree_construction_string.c_str());
    rstdout("Tree exchange         = %12.08f s ( %6.02f ) [ %s ]\n", tree_exchange_time, 100.0 * tree_exchange_time / total_time, tree_exchange_string.c_str());
    rstdout("Subdomain stitching   = %12.08f s ( %6.02f ) [ %s ]\n", subdomain_stitching_time, 100.0 * subdomain_stitching_time / total_time, subdomain_stitching_string.c_str());
    rstdout("Subdomain solver      = %12.08f s ( %6.02f ) [ %s ]\n", subdomain_solver_time, 100.0 * subdomain_solver_time / total_time, subdomain_solver_string.c_str());

#if 0
    {
        char word[80];

        std::string number_of_values_string;
        std::vector<int> num_values(num_procs);
        num_values[proc_id] = subdomain.num_values;
        MPI_Allreduce(MPI_IN_PLACE, num_values.data(), num_procs, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int p = 0; p < num_procs; p++)
        {
            if (p < num_procs - 1)
                sprintf(word, "%12d            ", num_values[p]);
            else
                sprintf(word, "%12d           ", num_values[p]);

            number_of_values_string += word;
        }

        rstdout("- Number of values = [ %s ]\n", number_of_values_string.c_str());

        std::string number_of_iterations_string;
        std::vector<int> num_iterations(num_procs);
        num_iterations[proc_id] = subdomain.num_iterations;
        MPI_Allreduce(MPI_IN_PLACE, num_iterations.data(), num_procs, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int p = 0; p < num_procs; p++)
        {
            if (p < num_procs - 1)
                sprintf(word, "%12d            ", num_iterations[p]);
            else
                sprintf(word, "%12d           ", num_iterations[p]);

            number_of_iterations_string += word;
        }

        rstdout("- Number of iterations = [ %s ]\n", number_of_iterations_string.c_str());
        
        for (int i = 0; i < 5; i++)
        {
            std::string output_string;
            std::vector<SType> timings(num_procs);

            if (i == 0) timings[proc_id] = timer.total("subdomain.inner_products");
            if (i == 1) timings[proc_id] = timer.total("subdomain.residual_norm");
            if (i == 2) timings[proc_id] = timer.total("subdomain.vector_operations");
            if (i == 3) timings[proc_id] = timer.total("subdomain.operator_application");
            if (i == 4) timings[proc_id] = timer.total("subdomain.preconditioner");

            MPI_Allreduce(MPI_IN_PLACE, timings.data(), num_procs, (typeid(SType) == typeid(double)) ? MPI_DOUBLE : MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            for (int p = 0; p < num_procs; p++)
            {
                SType percentage = 100.0 * timings[p] / subdomain_solver_times[p];

                if (p < num_procs - 1)
                    sprintf(word, "%12.08f ( %6.02f ) ", timings[p], percentage);
                else
                    sprintf(word, "%12.08f ( %6.02f )", timings[p], percentage);

                output_string += word;
            }

            if (i == 0) rstdout("- Inner products       = [ %s ]\n", output_string.c_str());
            if (i == 1) rstdout("- Residual norm        = [ %s ]\n", output_string.c_str());
            if (i == 2) rstdout("- Vector operations    = [ %s ]\n", output_string.c_str());
            if (i == 3) rstdout("- Operator application = [ %s ]\n", output_string.c_str());
            if (i == 4) rstdout("- Preconditioner       = [ %s ]\n", output_string.c_str());
        }
    }

    rstdout("\nWriting output files...\n");

    //domain.output("domain", 3, "u_star", u_star, "f", f, "u", u);
#endif
}
