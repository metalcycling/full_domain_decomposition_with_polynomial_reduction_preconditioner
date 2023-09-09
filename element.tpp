/*
 * Element template file
 */

// Class definition
// Constructor
template<typename DType>
Element<DType>::Element(int id_, int dim_, int poly_degree_)
{
    // Descriptors
    id = id_;
    dim = dim_;
    poly_degree = poly_degree_;
    num_points = std::pow(poly_degree + 1, dim);
    offset = 0;

    // Mesh
    n_x = poly_degree + 1;
    n_y = poly_degree + 1;
    n_z = poly_degree + 1;

    x.resize(num_points);
    y.resize(num_points);
    z.resize(num_points);

    // Dirichlet boundary conditions
    dirichlet_mask.resize(num_points);

    // Geometric factor
    for (int g = 0; g < NUM_GEOM_FACTS; g++)
        geom_fact[g].resize(num_points);

    // Connectivity
    loc_num.resize(num_points);
    glo_num.resize(num_points);
    dof_num.resize(num_points);

    vert_conn.resize((dim == 2) ? 4 : 8);
    edge_conn.resize((dim == 2) ? 4 : 12);
    face_conn.resize((dim == 2) ? 0 : 6);
}

template<typename DType>
Element<DType>::~Element()
{

}
