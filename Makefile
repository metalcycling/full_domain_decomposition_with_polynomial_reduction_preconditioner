# Unix Shell
SHELL = /bin/bash
HOSTNAME = $(shell hostname)

# Definitions
DEFINITIONS = -DHYPRE_TIMING

ifeq (${HOSTNAME}, kitsune)
	DEFINITIONS += -DHOSTNAME=0
else
	DEFINITIONS += -DHOSTNAME=1
endif

# Libraries
ifeq (${HOSTNAME}, kitsune)
	LIB_DIR = /home/metalcycling/Programs
else
	LIB_DIR = /gpfs/alpine/scratch/belloma2/csc262/Programs
endif

OCCA_INC = -I$(LIB_DIR)/OCCA/Master/include
OCCA_LIB = -L$(LIB_DIR)/OCCA/Master/lib -locca

SILO_INC = -I$(LIB_DIR)/Silo/build/opt/include
SILO_LIB = -L$(LIB_DIR)/Silo/build/opt/lib -lsilo

GSLIB_INC = -isystem$(LIB_DIR)/GSLib/build/opt/include
GSLIB_LIB = -L$(LIB_DIR)/GSLib/build/opt/lib -lgslib

HYPRE_INC = -I$(LIB_DIR)/Hypre/Master/src/build/cpu/double/opt/include
HYPRE_LIB = -L$(LIB_DIR)/Hypre/Master/src/build/cpu/double/opt/lib -lHYPRE

CUDA_INC = -I$(CUDA_HOME)/include
CUDA_LIB = -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse -lcurand -lcuda

# Compiler flags and variables
DEBUG = -O2
CUDA_ARCH = 70

CC = mpicc
CC_FLAGS = -Wall
CC_INCLUDES = -I./

CXX = mpic++
CXX_FLAGS = -Wall -Wno-unused-result
CXX_INCLUDES = -I./ $(OCCA_INC) $(SILO_INC) $(GSLIB_INC) $(HYPRE_INC) $(CUDA_INC)

CU = nvcc
CU_FLAGS =  
CU_INCLUDES = -I./

FC = mpifort
FC_FLAGS = -fdefault-real-8 -std=legacy
FC_INCLUDES =

LD = nvcc
LD_FLAGS = 
LD_LIBRARIES = $(OCCA_LIB) $(SILO_LIB) $(GSLIB_LIB) $(HYPRE_LIB) $(CUDA_LIB) -ccbin=$(CXX) -gencode arch=compute_$(CUDA_ARCH),"code=sm_$(CUDA_ARCH)" -lstdc++ -lm

ifeq (${HOSTNAME}, kitsune)
	LD_LIBRARIES += -L/usr/lib/gcc/x86_64-linux-gnu/9 -lgfortran
else
	LD_LIBRARIES += -L/sw/summit/gcc/9.1.0-alpha+20190716/bin/gfortran -lgfortran
endif

# Source files
SRCS_CC := $(shell find . -name '*.c')
SRCS_CXX := $(shell find . -name '*.cpp')
SRCS_CU := $(shell find . -name '*.cu')
SRCS_FC := $(shell find . -name '*.f')
OBJS = $(SRCS_CC:.c=.o) $(SRCS_CXX:.cpp=.o) $(SRCS_FC:.f=.o) $(SRCS_CU:.cu=.o)

# Executable
EXE = poisson

# Make rules
all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(DEBUG) $(DEFINITIONS) $(LD_FLAGS) -o $@ $(OBJS) $(LD_LIBRARIES)

%.o: %.c
	$(CC) $(DEBUG) $(DEFINITIONS) $(CC_FLAGS) $(CC_INCLUDES) -c $< -o $@

%.o: %.cu
	$(CU) $(DEBUG) $(DEFINITIONS) $(CU_FLAGS) $(CU_INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(DEBUG) $(DEFINITIONS) $(CXX_FLAGS) $(CXX_INCLUDES) -c $< -o $@

%.o: %.f
	$(FC) $(DEBUG) $(DEFINITIONS) $(FC_FLAGS) $(FC_INCLUDES) -c $< -o $@

clean:
	rm $(EXE);
	find . -name "*.o" -delete
