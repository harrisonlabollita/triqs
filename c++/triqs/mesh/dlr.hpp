/*******************************************************************************
 *
 * TRIQS: a Toolbox for Research in Interacting Quantum Systems
 *
 * Copyright (C) 2023 by Hugo U. R. Strand
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
#include "./details/mesh_tools.hpp"
#include "./bases/dlr.hpp"

#include <cppdlr/dlr_kernels.hpp>

namespace triqs::mesh {

  class dlr_coeffs;
  class dlr_imtime;
  
  /**
   *  Imaginary-time Discrete Lehmann representation mesh
   *
   *  Mesh for the imaginary-time axis from $0$ to $\beta$.
   *
   *  The time points are computed using the cppdlr library
   */
  class dlr_imtime : public dlr_mesh<matsubara_time_domain, tag::dlr_repr_imtime> {
    using B = dlr_mesh<matsubara_time_domain, tag::dlr_repr_imtime>;
    
    public:

    using domain_t       = matsubara_time_domain;
    using index_t        = long;
    using linear_index_t = long;
    using domain_pt_t    = typename domain_t::point_t;

    ///
    dlr_imtime() = default;

    ///
    dlr_imtime(dlr_imtime const &x) = default;
    
    template<typename M> dlr_imtime(M const &m) : B(m.domain(), m.lambda(), m.eps(), m.dlr_freq(), m.dlr()) {}

    /**
     * Construct a Mesh of Discrete Lehmann imaginary times
     * on a Matsubara time domain
     *
     * @param dom Matsubara time domain
     * @param lambda Lambda energy over beta parameter
     * @param eps Representation accuracy
     */
    dlr_imtime(matsubara_time_domain d, double lambda, double eps) : B(d, lambda, eps) {}

    /**
     * Construct a Mesh of imaginary times on the interval [0,beta]
     * including points at both edges.
     *
     * @param beta Inverse temperature
     * @param S Statistic (Fermion or Boson)
     * @param lambda Lambda energy over beta parameter
     * @param eps Representation accuracy
     */
    dlr_imtime(double beta, statistic_enum S, double lambda, double eps) : dlr_imtime({beta, S}, lambda, eps) {}

    // -------------------- print -------------------

    friend std::ostream &operator<<(std::ostream &sout, dlr_imtime const &m) {
      return sout << "DLR Imaginary Time Mesh of size " << m.size() << ", Domain: " << m.domain();
    }

    // -------------------- hdf5 -------------------

    static std::string hdf5_format() { return "MeshDLRImTime"; }

    friend void h5_write(h5::group fg, std::string const &subgroup_name, dlr_imtime const &m) { h5_write_impl(fg, subgroup_name, m, "MeshDLRImTime"); }

    friend void h5_read(h5::group fg, std::string const &subgroup_name, dlr_imtime &m) { h5_read_impl(fg, subgroup_name, m, "MeshDLRImTime"); }

  };

  /**
   *  Discrete Lehmann representation mesh with coefficients
   *
   *  Mesh for the imaginary-time axis from $0$ to $\beta$.
   *
   *  The coefficients are computed using the cppdlr library
   */
  class dlr_coeffs : public dlr_mesh<matsubara_time_domain, tag::dlr_repr_coeffs> {
    using B = dlr_mesh<matsubara_time_domain, tag::dlr_repr_coeffs>;
    
    public:

    using domain_t       = matsubara_time_domain;
    using index_t        = long;
    using linear_index_t = long;
    using domain_pt_t    = typename domain_t::point_t;

    ///
    dlr_coeffs() = default;

    ///
    dlr_coeffs(dlr_coeffs const &x) = default;

    template<typename M> dlr_coeffs(M const &m) : B(m.domain(), m.lambda(), m.eps(), m.dlr_freq(), m.dlr()) {}
    
    /**
     * Construct a Mesh of Discrete Lehmann imaginary times
     * on a Matsubara time domain
     *
     * @param dom Matsubara time domain
     * @param lambda Lambda energy over beta parameter
     * @param eps Representation accuracy
     */
    dlr_coeffs(matsubara_time_domain d, double lambda, double eps) : B(d, lambda, eps) {}

    /**
     * Construct a Mesh of imaginary times on the interval [0,beta]
     * including points at both edges.
     *
     * @param beta Inverse temperature
     * @param S Statistic (Fermion or Boson)
     * @param lambda Lambda energy over beta parameter
     * @param eps Representation accuracy
     */
    dlr_coeffs(double beta, statistic_enum S, double lambda, double eps) : dlr_coeffs({beta, S}, lambda, eps) {}

    // -------------------- print -------------------

    friend std::ostream &operator<<(std::ostream &sout, dlr_coeffs const &m) {
      return sout << "DLR coefficient Mesh of size " << m.size() << ", Domain: " << m.domain();
    }

    // -------------------- interpolation -------------------

    nda::vector<double> get_interpolation_data(double tau) const {
      auto res = nda::zeros<double>(size());
      for (auto l : range(size()))
	res[l] = cppdlr::kfun(tau / domain().beta, index_to_point(l));
      return res;
    }
    
    // -------------------- hdf5 -------------------

    static std::string hdf5_format() { return "MeshDLRCoeffs"; }

    friend void h5_write(h5::group fg, std::string const &subgroup_name, dlr_coeffs const &m) { h5_write_impl(fg, subgroup_name, m, "MeshDLRCoeffs"); }

    friend void h5_read(h5::group fg, std::string const &subgroup_name, dlr_coeffs &m) { h5_read_impl(fg, subgroup_name, m, "MeshDLRCoeffs"); }

  };
  
} // namespace triqs::mesh
