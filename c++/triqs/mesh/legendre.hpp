// Copyright (c) 2014-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2014-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018 Simons Foundation
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You may obtain a copy of the License at
//     https://www.gnu.org/licenses/gpl-3.0.txt
//
// Authors: Michel Ferrero, Olivier Parcollet, Nils Wentzell

#pragma once
#include "./domains/legendre.hpp"
#include "bases/linear.hpp"

namespace triqs::mesh {

  struct legendre : public linear_mesh<legendre_domain> {
    legendre() = default;
    legendre(legendre_domain domain) : linear_mesh(std::move(domain), 0, domain.n_max, domain.n_max + 1) {}
    legendre(double beta, statistic_enum S, long n_max) : linear_mesh(legendre_domain(beta, S, n_max), 0, n_max, n_max + 1) {}

    // -------------------- HDF5 -------------------
    static std::string hdf5_format() { return "MeshLegendre"; }
    friend void h5_write(h5::group fg, std::string const &subgroup_name, legendre const &m) { h5_write_impl(fg, subgroup_name, m, "MeshLegendre"); }
    friend void h5_read(h5::group fg, std::string const &subgroup_name, legendre &m) { h5_read_impl(fg, subgroup_name, m, "MeshLegendre"); }
  };

} // namespace triqs::mesh
