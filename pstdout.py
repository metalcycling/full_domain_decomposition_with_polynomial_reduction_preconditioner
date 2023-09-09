import glob
import numpy as np
import matplotlib.pyplot as plt

num_procs = len(glob.glob("pstdout_*.dat"))

#for p in range(num_procs):
for p in [0]:
    stdout_file = open("pstdout_%d.dat" % p, "r")
    data = stdout_file.readlines()
    stdout_file.close()

    print("pstdout_%d.dat:" % p)
    print("".join(data))
