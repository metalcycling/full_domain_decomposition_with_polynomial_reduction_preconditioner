/*
 * Element header file
 */

// Headers
#include <unordered_map>
#include <set>

// Definitions
#ifndef NUM_GEOM_FACTS
#define NUM_GEOM_FACTS 6
#endif

// Class declaration
#ifndef ELEMENT_HPP
#define ELEMENT_HPP

template<typename DType>
class Element
{
    public:
        // Descriptors
        int id;
        int dim;
        int poly_degree;
        int num_points;
        int offset;

        // Mesh
        int n_x;
        int n_y;
        int n_z;

        std::vector<DType> x;
        std::vector<DType> y;
        std::vector<DType> z;

        // Dirichlet boundary conditions
        std::vector<DType> dirichlet_mask;

        // Geometric factor
        std::vector<DType> geom_fact[NUM_GEOM_FACTS];

        // Connectivity
        std::vector<int> loc_num;
        std::vector<long long> glo_num;
        std::vector<long long> dof_num;
        std::vector<std::set<int>> vert_conn;
        std::vector<std::set<int>> edge_conn;
        std::vector<std::set<int>> face_conn;

        // Constructor and destructor
        Element(int, int, int);
        ~Element();
};

#include "element.tpp"

#endif
