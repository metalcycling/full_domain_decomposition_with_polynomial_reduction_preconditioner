/*
 * Configuration file with headers and variables commont to all files
 */
#include <iostream>
#include <unordered_map>
#include <mpi.h>
#include <occa.hpp>

extern "C"
{
    #include "_hypre_utilities.h"
    #include "HYPRE_parcsr_ls.h"
    #include "_hypre_parcsr_ls.h"
    #include "_hypre_IJ_mv.h"
    #include "HYPRE.h"
}

#include "AMG/config.hpp"
#define STYPE double
#define PTYPE Float

#ifndef PRINT
#define PRINT 1
#endif

#if PRINT == 1
#define pstdout(...) { fprintf(pstdout_file, __VA_ARGS__); fflush(pstdout_file); }
#else
#define pstdout(...) 
#endif

#define rstdout(...) { if (proc_id == 0) { printf(__VA_ARGS__); fflush(stdout); } }

#ifndef OCCA_TYPE
#define OCCA_TYPE 1
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#define BINARY_INPUT true

#define VISUALIZATION 0

#ifndef GLOBALS_READY
#define GLOBALS_READY
int dim;
int proc_id;
int num_procs;

occa::device device;

char pstdout_name[80];
FILE *pstdout_file;

void quit()
{
    HYPRE_Finalize();
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}

#include "timer.hpp"
Timer<double> timer;

#else
extern int dim;
extern int proc_id;
extern int num_procs;

extern occa::device device;

extern char pstdout_name[80];
extern FILE *pstdout_file;

extern Timer<STYPE> timer;

void quit();

#endif
