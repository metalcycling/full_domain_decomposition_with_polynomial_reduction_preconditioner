import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
import matplotlib.pyplot as plt

def read_matrix(file_name):
    row_start, row_end, col_start, col_end = map(int, open(file_name).readline().split())
    num_rows = row_end - row_start + 1
    num_cols = col_end - col_start + 1
    data = np.loadtxt(file_name, skiprows = 1)
    return spsp.csr_matrix((data[:, 2], (data[:, 0].astype(int), data[:, 1].astype(int))), shape = (num_rows, num_cols))

nek_dir = "/home/metalcycling/Dropbox/University_of_Illinois/Research/Unstructured_Range_Decomposition/Code/Nek5000/Kershaw"
cpp_dir = "/home/metalcycling/Dropbox/University_of_Illinois/Research/Unstructured_Range_Decomposition/Code/Cpp/Full_Domain_Decomposition_with_Polynomial_Reduction/a1"

"""
nek_map = np.loadtxt("%s/mapping_0.dat" % (nek_dir), dtype = int)
cpp_map = np.loadtxt("%s/mapping_0.dat" % (cpp_dir), dtype = int)

num_elems, num_points = nek_map.shape
num_dofs = nek_map.max()
mapping = np.empty(num_dofs, dtype = int)

for e in range(num_elems):
    for v in range(num_points):
        if (nek_map[e, v] > 0):
            mapping[nek_map[e, v] - 1] = cpp_map[e, v] - 1

P = spsp.csr_matrix((np.ones(num_dofs), (np.arange(num_dofs, dtype = int), mapping)), shape = (num_dofs, num_dofs))
#"""

"""
A_nek = read_matrix("/home/metalcycling/Dropbox/University_of_Illinois/Research/Unstructured_Range_Decomposition/Code/Nek5000/Kershaw/A_0.00000")
A_cpp = read_matrix("/home/metalcycling/Dropbox/University_of_Illinois/Research/Unstructured_Range_Decomposition/Code/Cpp/Full_Domain_Decomposition_with_Polynomial_Reduction/a1/A_0.00000")
print(npla.norm(A_nek * np.ones(A_nek.shape[1])))
print(npla.norm(A_cpp * np.ones(A_cpp.shape[1])))

print(npla.norm((A_nek - P * A_cpp * P.T).toarray()))
plt.spy(A_nek - P * A_cpp * P.T, precision = 1.0e-8)
#plt.spy(A_nek - A_cpp, precision = 1.0e-8)
plt.show()
#"""

"""
A_l_nek = np.loadtxt("%s/pstdout_0.dat" % (nek_dir))
A_l_cpp = np.loadtxt("%s/pstdout_0.dat" % (cpp_dir))
print(npla.norm(A_l_nek - A_l_cpp))
#"""

"""
dof_num_cpp = np.loadtxt("%s/pstdout_0.dat" % (cpp_dir), dtype = int).reshape(-1)
dof_num_cpp.sort()

plt.figure(figsize = (12, 8))
plt.plot(dof_num_cpp[1:] - dof_num_cpp[:-1])
plt.show()
#"""

"""
nek_data = np.loadtxt("%s/debug_0.dat" % (nek_dir))
cpp_data = np.loadtxt("%s/debug_0.dat" % (cpp_dir))

print(npla.norm(nek_data))
print(npla.norm(cpp_data))

print(npla.norm(nek_data - P * cpp_data))
#"""

single = np.loadtxt("pstdout_0.dat")
double = np.loadtxt("Double/pstdout_0.dat")
print(npla.norm(single - double, ord = np.inf))
