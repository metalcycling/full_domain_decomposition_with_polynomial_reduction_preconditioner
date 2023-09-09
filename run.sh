#!/bin/bash
clean=1
build=1
run=1

if [ ${clean} -eq 1 ]; then
    rm pstdout*.dat
    rm *.silo
    rm *.D.*
    rm *.INFO.*
    rm *.00000
    rm core.*
fi

if [ ${build} -eq 1 ]; then
    if [ ${clean} -eq 1 ]; then
        make clean
    fi

    make
fi

num_procs=8
poly_degree=2
poly_reduction=1
subdomain_overlap=1
superdomain_overlap=1

if [ ${run} -eq 1 ]; then
    path="/home/metalcycling/Dropbox/University_of_Illinois/Research/Unstructured_Range_Decomposition/Code/Nek5000/Kershaw/Meshes/eps_0.3/16x16x16/P_${num_procs}"
    args="${path} ${poly_degree} ${poly_reduction} ${subdomain_overlap} ${superdomain_overlap}"

    if [ $(hostname) = kitsune ]; then
        mpirun -np ${num_procs} ./poisson ${args}
    else
        jsrun -n ${num_procs} -c 1 -a 1 -g 1 ./poisson ${args}
    fi

    echo
    python3 pstdout.py
fi
