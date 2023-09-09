proc_id=${OMPI_COMM_WORLD_RANK}
num_procs=${OMPI_COMM_WORLD_SIZE}
filename=profile.${proc_id}.nvvp

poly_degree=7
poly_reduction=3
subdomain_overlap=1
superdomain_overlap=1

rm ${filename}
nvprof -s --profile-from-start off -o ${filename} ./poisson "/gpfs/alpine/scratch/belloma2/csc262/Research/Unstructured_Range_Decomposition/Code/Nek5000/Kershaw/eps_0.3/64x64x64/P_${num_procs}" ${poly_degree} ${poly_reduction} ${subdomain_overlap} ${superdomain_overlap}
