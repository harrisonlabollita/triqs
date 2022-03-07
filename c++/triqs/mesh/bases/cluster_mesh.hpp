/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2014 by O. Parcollet
 *
 * TRIQS is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * TRIQS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TRIQS. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#pragma once
#include <triqs/arrays.hpp>
#include <triqs/utility/index_generator.hpp>
#include <triqs/utility/arithmetic_ops_by_cast.hpp>

#include "../details/mesh_tools.hpp"

namespace triqs::mesh {

  /**
   * Compute dimensions of a parallelepiped cluster cell using the inverse of the periodization matrix
   *
   * This function computes the dimensions of a parallelepipedic super unit cell (i.e corresponding to the super vectors).
   *
   * for a given Bravais lattice (defined by unit vectors ${a_i}_{i=0\dots d-1}$), the periodic boundary conditions are uniquely
   * defined by the matrix $N$ such that the super vectors $\tilde{a}_i$ are given by:
   *
   * $$\tilde{a}_i = \sum_j N_{ij} a_j$$
   *
   * Example:
   *  If $N_{ij}$ is diag{n_k1, n_k2, n_k3}, this returns {n_k1, n_k2, n_k3}
   *
   * The algorithm used is the following:
   * let $C={(0,0,0)}$
   * for each dimension $d=1\dots 3$ :
   *   - Find the lowest nonzero integer $k_d$ such that there exists $mathbf{x}$ in C such for all $d_p$ in $1\dots 3$, $k_d mathbf{a}_d - mathbf{x}$ belongs to the superlattice.
   *   - Update $C = {mathbf{x} + q mathbf{a}_d, mathbf{x}\in C, q=0\dots k_d-1}$
   * return {k_d}_{k=1\dots d}
   *
   * @param periodization_matrix The periodization matrix
   * @return The dimensions of the parallelepiped unit cell
   */
  std::array<long, 3> find_cell_dims(nda::matrix<long> const &periodization_matrix);

  /// A lattice point
  struct lattice_point : public utility::arithmetic_ops_by_cast<lattice_point, nda::vector<double>> {
    std::array<long, 3> index;
    nda::matrix<double> units;

    lattice_point() : index({0, 0, 0}), units(nda::eye<double>(3)) {}
    lattice_point(std::array<long, 3>  const &index_, matrix<double> const &units_) : index(index_), units(units_) {}
    using cast_t = nda::vector<double>;
    operator cast_t() const {
      cast_t M(3);
      M() = 0.0;
      // slow, to be replaced with matrix vector multiplication
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) M(i) += index[j] * units(j, i);
      return M;
    }
    friend std::ostream &operator<<(std::ostream &out, lattice_point const &x) { return out << (cast_t)x; }
  };

  struct cluster_mesh : tag::mesh {

    public:
    cluster_mesh() = default;

    /**
     * Construct from basis vectors and periodization matrix
     *
     * @param units Matrix $B$ containing as rows the basis vectors that generate mesh points
     * @param periodization_matrix Matrix $N$ specifying the translation vectors for the
     *        periodic boundary conditions
     *      $$ \mathbf{x} \equiv \mathbf{x} + \mathbf{z} \cdot \mathbf{N} \forall \mathbf{n} in \Z^n$
     */
    cluster_mesh(matrix<double> const &units, matrix<long> const &periodization_matrix) : units_(units), periodization_matrix_(periodization_matrix) {
      EXPECTS((periodization_matrix.shape() == std::array{3l, 3l}));

      // The index_modulo operation currently assumes a diagonal periodization matrix by treating each index element separately.
      // It needs to be generalized to use only the periodicity as specified in the periodization matrix, i.e.
      //   $$ (i, j, k) -> (i, j, k) + (n1, n2, n3) * periodization_matrix $$
      if (nda::diag(nda::diagonal(periodization_matrix)) != periodization_matrix)
        throw std::runtime_error{"Non-diagonal periodization matrices are currently not supported."};

      dims_   = find_cell_dims(periodization_matrix_);
      size_   = dims_[0] * dims_[1] * dims_[2];
      stride0 = dims_[1] * dims_[2];
      stride1 = dims_[2];
    }

    int rank() const { return (dims_[2] > 1 ? 3 : (dims_[1] > 1 ? 2 : 1)); }

    /// The extent of each dimension
    std::array<long, 3> dims() const { return dims_; }

    /// Matrix containing the mesh basis vectors as rows
    matrix_const_view<double> units() const { return units_; }

    // The matrix defining the periodization on the mesh
    matrix_const_view<long> periodization_matrix() const { return periodization_matrix_; }

    /// ---------- Model the domain concept  ---------------------

    using domain_t = cluster_mesh;

    domain_t const &domain() const { return *this; }

    using point_t = nda::vector<double>; // domain concept. PUT on STACK

    /// ----------- Model the mesh concept  ----------------------

    using index_t = std::array<long, 3>;

    using linear_index_t = long;

    /// Reduce index modulo to the lattice
    index_t index_modulo(index_t const &r) const { return index_t{_modulo(r[0], 0), _modulo(r[1], 1), _modulo(r[2], 2)}; }

    /// The total number of points in the mesh
    size_t size() const { return size_; }

    /// from the index (n_i) to the cartesian coordinates
    /** for a point M of coordinates n_i in the {a_i} basis, the cartesian coordinates are
    *     $$ OM_i = \sum_j n_j X_{ji} $$
    * @param index_t the (integer) coordinates of the point (in basis a_i)
    * @warning can be made faster by writing this a matrix-vector multiplication
    */
    point_t index_to_point(index_t const &n) const {
      EXPECTS(n == index_modulo(n));
      point_t M(3);
      M() = 0.0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) M(i) += n[j] * units_(j, i);
      return M;
    }

    /// flatten the index
    linear_index_t index_to_linear(index_t const &i) const {
      EXPECTS(i == index_modulo(i));
      return i[0] * stride0 + i[1] * stride1 + i[2];
    }

    /// Is the point in the mesh ? Always true
    template <typename T> static constexpr bool is_within_boundary(T const &) { return true; }

    using mesh_point_t = mesh_point<cluster_mesh>;

    /// Accessing a point of the mesh from its index
    inline mesh_point_t operator[](index_t i) const; // impl below

    /// Iterating on all the points...
    using const_iterator = mesh_pt_generator<cluster_mesh>;
    inline const_iterator begin() const; // impl below
    inline const_iterator end() const;
    inline const_iterator cbegin() const;
    inline const_iterator cend() const;

    /// Mesh comparison
    bool operator==(cluster_mesh const &M) const {
      return ((dims_ == M.dims_) && (units_ == M.units_) && (periodization_matrix_ == M.periodization_matrix_));
    }
    bool operator!=(cluster_mesh const &M) const { return !(operator==(M)); }

    /// locate the closest point
    inline index_t closest_index(point_t const &x) const {
      auto idbl = transpose(inverse(units_)) * x;
      return {std::lround(idbl[0]), std::lround(idbl[1]), std::lround(idbl[2])};
    }

    protected:
    matrix<double> units_;
    matrix<long> periodization_matrix_;
    std::array<long, 3> dims_;
    size_t size_;
    long stride1, stride0;

    long _modulo(long r, int i) const {
      long res = r % dims_[i];
      return (res >= 0 ? res : res + dims_[i]);
    }

    // -------------- HDF5  --------------------------

    public:
    /// Write into HDF5
    friend void h5_write_impl(h5::group fg, std::string subgroup_name, cluster_mesh const &m, const char *_type) {
      h5::group gr = fg.create_group(subgroup_name);
      write_hdf5_format_as_string(gr, _type);
      h5_write(gr, "units", m.units_);
      h5_write(gr, "periodization_matrix", m.periodization_matrix_);
    }

    /// Read from HDF5
    friend void h5_read_impl(h5::group fg, std::string subgroup_name, cluster_mesh &m, const char *_type) {
      h5::group gr = fg.open_group(subgroup_name);
      assert_hdf5_format_as_string(gr, _type, true);
      auto units                = h5::h5_read<matrix<double>>(gr, "units");
      auto periodization_matrix = h5::h5_read<matrix<long>>(gr, "periodization_matrix");
      m                         = cluster_mesh(units, periodization_matrix);
    }

    friend std::ostream &operator<<(std::ostream &sout, cluster_mesh_base const &m) {
      return sout << "cluster_mesh of size " << m.dims() << "\n units = " << m.units() << "\n periodization_matrix = " << m.periodization_matrix()
                  << "\n";
    }
  };

  // ---------------------------------------------------------------------------
  //                     The mesh point
  // ---------------------------------------------------------------------------
  template <>
  struct mesh_point<cluster_mesh> : public utility::index3_generator,
                                    public utility::arithmetic_ops_by_cast<mesh_point<cluster_mesh>, cluster_mesh::index_t> {
    public:
    using mesh_t = cluster_mesh;

    private:
    mesh_t const *m = nullptr;

    public:
    using index_t        = mesh_t::index_t;
    using point_t        = mesh_t::point_t;
    using linear_index_t = mesh_t::linear_index_t;

    mesh_point() = default;
    explicit mesh_point(mesh_t const &mesh, mesh_t::index_t const &index) : index3_generator(mesh.dims(), index), m(&mesh) {}
    mesh_point(mesh_t const &mesh) : mesh_point(mesh, {0, 0, 0}) {}

    using cast_t = point_t; // FIXME : decide what we want.

    operator mesh_t::point_t() const { return m->index_to_point(index()); }
    operator lattice_point() const { return lattice_point(index(), m->units()); }
    operator mesh_t::index_t() const { return index(); }
    linear_index_t linear_index() const { return m->index_to_linear(index()); }
    // The mesh point behaves like a vector
    /// d: component (0, 1 or 2)
    double operator()(int d) const { return m->index_to_point(index())[d]; }
    double operator[](int d) const { return operator()(d); }
    friend std::ostream &operator<<(std::ostream &out, mesh_point const &x) { return out << (lattice_point)x; }
    mesh_point operator-() const { return mesh_point{*m, m->index_modulo({-index()[0], -index()[1], -index()[2]})}; }
    mesh_t const &mesh() const { return *m; }
  };

  // --- impl
  inline mesh_point<cluster_mesh> cluster_mesh::operator[](index_t i) const {
    EXPECTS(i == index_modulo(i));
    return mesh_point<cluster_mesh>{*this, i};
  }

  inline cluster_mesh::const_iterator cluster_mesh::begin() const { return const_iterator(this); }
  inline cluster_mesh::const_iterator cluster_mesh::end() const { return const_iterator(this, true); }
  inline cluster_mesh::const_iterator cluster_mesh::cbegin() const { return const_iterator(this); }
  inline cluster_mesh::const_iterator cluster_mesh::cend() const { return const_iterator(this, true); }
} // namespace triqs::mesh
