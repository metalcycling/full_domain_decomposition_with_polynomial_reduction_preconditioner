import os
import time
import glob
import subprocess

# Functions
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Problem parameters
polynomial_degree = 7
subdomain_overlap = 1
superdomain_overlap = 1

# Simulation parameters
output_dir = "Strong_Scaling_3"
output_file = "pr_fdd"

parameters = {}

"""
parameters["kershaw_a"] = {}
parameters["kershaw_a"]["in_dir"] = "/gpfs/alpine/scratch/belloma2/csc262/Research/Unstructured_Range_Decomposition/Code/Nek5000/Kershaw/Meshes/eps_1.0/64x64x64"
parameters["kershaw_a"]["out_dir"] = "Kershaw_A"
parameters["kershaw_a"]["num_procs"] = [64, 128, 256, 512, 1024, 32]
parameters["kershaw_a"]["preconditioners"] = ["GMRES"]
parameters["kershaw_a"]["precision"] = ["double"]
parameters["kershaw_a"]["num_vcycles"] = [1]
parameters["kershaw_a"]["cheby_order"] = [1, 2]
parameters["kershaw_a"]["level_cutoff"] = [5]
parameters["kershaw_a"]["num_iterations"] = [1, 2, 4, 8]
#"""

#"""
parameters["kershaw_b"] = {}
parameters["kershaw_b"]["in_dir"] = "/gpfs/alpine/scratch/belloma2/csc262/Research/Unstructured_Range_Decomposition/Code/Nek5000/Kershaw/Meshes/eps_0.3/64x64x64"
parameters["kershaw_b"]["out_dir"] = "Kershaw_B"
parameters["kershaw_b"]["num_procs"] = [32, 128, 256, 512]
parameters["kershaw_b"]["preconditioners"] = ["GMRES"]
parameters["kershaw_b"]["precision"] = ["double"]
parameters["kershaw_b"]["num_vcycles"] = [1]
parameters["kershaw_b"]["cheby_order"] = [2]
parameters["kershaw_b"]["level_cutoff"] = [5]
parameters["kershaw_b"]["num_iterations"] = [4]
parameters["kershaw_b"]["polynomial_reduction"] = [6]
#"""

#"""
parameters["pb_146"] = {}
parameters["pb_146"]["in_dir"] = "/gpfs/alpine/scratch/belloma2/csc262/Research/Unstructured_Range_Decomposition/Code/Nek5000/Pebble_Bed/PB_146/Meshes"
parameters["pb_146"]["out_dir"] = "PB_146"
parameters["pb_146"]["num_procs"] = [32, 64, 128, 256, 512, 1024]
parameters["pb_146"]["preconditioners"] = ["GMRES"]
parameters["pb_146"]["precision"] = ["double"]
parameters["pb_146"]["num_vcycles"] = [1]
parameters["pb_146"]["cheby_order"] = [2]
parameters["pb_146"]["level_cutoff"] = [5]
parameters["pb_146"]["num_iterations"] = [4]
parameters["pb_146"]["polynomial_reduction"] = [6]
#"""

#"""
parameters["pb_1568"] = {}
parameters["pb_1568"]["in_dir"] = "/gpfs/alpine/scratch/belloma2/csc262/Research/Unstructured_Range_Decomposition/Code/Nek5000/Pebble_Bed/PB_1568/Meshes"
parameters["pb_1568"]["out_dir"] = "PB_1568"
parameters["pb_1568"]["num_procs"] = [256, 512, 1024, 2048]
parameters["pb_1568"]["preconditioners"] = ["GMRES"]
parameters["pb_1568"]["precision"] = ["double"]
parameters["pb_1568"]["num_vcycles"] = [1]
parameters["pb_1568"]["cheby_order"] = [2]
parameters["pb_1568"]["level_cutoff"] = [5]
parameters["pb_1568"]["num_iterations"] = [4]
parameters["pb_1568"]["polynomial_reduction"] = [6]
#"""

prec_id = { "FCG": 0, "GMRES": 1 }
precision_label = { "double": "Double", "float": "Single" }
num_gpus_node = 6

# Create output directory
create_directory("%s" % (output_dir))

# Run simulations
for name in parameters:
    create_directory("%s/%s" % (output_dir, parameters[name]["out_dir"]))

    for num_procs in parameters[name]["num_procs"]:
        create_directory("%s/%s/P_%d" % (output_dir, parameters[name]["out_dir"], num_procs))

        num_nodes = (num_procs + num_gpus_node - 1) // num_gpus_node
        total_runs  = len(parameters[name]["preconditioners"])
        total_runs *= len(parameters[name]["precision"])
        total_runs *= len(parameters[name]["num_vcycles"])
        total_runs *= len(parameters[name]["num_iterations"])
        total_runs *= len(parameters[name]["cheby_order"])
        total_runs *= len(parameters[name]["level_cutoff"])
        total_runs *= len(parameters[name]["polynomial_reduction"])

        while True:
            completed_runs = 0

            lsf_file = open("run.lsf", "w")
            lsf_file.write("#!/bin/bash\n\n")
            lsf_file.write("#BSUB -W 45\n")
            lsf_file.write("#BSUB -nnodes %d\n" % (num_nodes))
            lsf_file.write("#BSUB -P CSC262\n")
            lsf_file.write("#BSUB -J PR_FDD\n")
            lsf_file.write("#BSUB -N belloma2@illinois.edu\n\n")

            for prec_name in parameters[name]["preconditioners"]:
                path = ["%s" % (output_dir), "%s" % (parameters[name]["out_dir"]), "P_%d" % (num_procs), "%s" % (prec_name)]
                create_directory("/".join(path))

                for num_vcycles in parameters[name]["num_vcycles"]:
                    path.append("V_%d" % (num_vcycles))
                    create_directory("/".join(path))

                    for num_iter in parameters[name]["num_iterations"]:
                        path.append("I_%d" % (num_iter))
                        create_directory("/".join(path))

                        for precision in parameters[name]["precision"]:
                            path.append("%s" % (precision_label[precision]))
                            create_directory("/".join(path))

                            for cheby_order in parameters[name]["cheby_order"]:
                                path.append("C_%d" % (cheby_order))
                                create_directory("/".join(path))

                                for level_cutoff in parameters[name]["level_cutoff"]:
                                    path.append("L_%d" % (level_cutoff))
                                    create_directory("/".join(path))

                                    for polynomial_reduction in parameters[name]["polynomial_reduction"]:
                                        path.append("Nr_%d" % (polynomial_reduction))
                                        create_directory("/".join(path))

                                        not_found = True

                                        filename = "%s/%s.dat" % ("/".join(path), output_file)

                                        if os.path.exists(filename):
                                            proc = subprocess.Popen(["grep", "Total   ", filename], stdout = subprocess.PIPE)

                                            if len(proc.stdout.readlines()) > 0:
                                                not_found = False
                                                completed_runs += 1

                                        if not_found:
                                            lsf_file.write("sed -i \"116s/.*/        int preconditioner_type = %d;/\" domain.hpp\n" % (prec_id[prec_name]))
                                            lsf_file.write("sed -i \"229s/.*/        int num_vectors = %d;/\" subdomain.hpp\n" % (num_iter))
                                            lsf_file.write("sed -i \"230s/.*/        int max_iterations = %d;/\" subdomain.hpp\n" % (num_iter))
                                            lsf_file.write("sed -i \"236s/.*/        int num_vcycles = %d;/\" subdomain.hpp\n" % (num_vcycles))
                                            lsf_file.write("sed -i \"237s/.*/        int cheby_order = %d;/\" subdomain.hpp\n" % (cheby_order))
                                            lsf_file.write("sed -i \"238s/.*/        int level_cutoff = %d;/\" subdomain.hpp\n" % (level_cutoff))
                                            lsf_file.write("sed -i \"4s/.*/#define Float %s/\" AMG/config.hpp\n" % (precision))
                                            lsf_file.write("make clean\n")
                                            lsf_file.write("make\n")
                                            lsf_file.write("jsrun -n %d -c 1 -a 1 -g 1 ./poisson %s/P_%d %d %d %d %d > %s.dat\n" % (num_procs, parameters[name]["in_dir"], num_procs, polynomial_degree, polynomial_reduction, subdomain_overlap, superdomain_overlap, output_file))
                                            lsf_file.write("mv *.dat %s\n" % ("/".join(path)))
                                            lsf_file.write("\n")

                                        path.pop()
                                    path.pop()
                                path.pop()
                            path.pop()
                        path.pop()
                    path.pop()
                path.pop()

            lsf_file.close()

            if completed_runs == total_runs: break

            # Run
            command = ["bsub", "run.lsf"]
            proc = subprocess.Popen(command, stdout = subprocess.PIPE)
            response = proc.stdout.readlines()
            job_id = response[0].decode("utf-8").split()[1][1:-1]

            time.sleep(2)
            time_step = 5
            total_time = 0
            running = True

            while (running):
                command = ["bjobs", "-u", "belloma2"]
                proc = subprocess.Popen(command, stdout = subprocess.PIPE)
                response = proc.stdout.readlines()

                if (len(response) == 0):
                    running = False

                else:
                    found = False

                    for i in range(1, len(response)):
                        if job_id == response[i].decode("utf-8").split()[0]:
                            found = True

                    if not found:
                        running = False

                    time.sleep(time_step)
                    total_time += time_step
                    print("Total time: %d s" % (total_time))

            time.sleep(2)
