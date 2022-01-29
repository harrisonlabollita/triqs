// TRIQS: a Toolbox for Research in Interacting Quantum Systems
//
// Copyright (c) 2013-2018 Commissariat à l'énergie atomique et aux énergies alternatives (CEA)
// Copyright (c) 2013-2018 Centre national de la recherche scientifique (CNRS)
// Copyright (c) 2018-2021 Simons Foundation
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
// Authors: Michel Ferrero, Olivier Parcollet, Nils Wentzell, Philipp Dumitrescu

#pragma once

#include <h5/h5.hpp>
#include "../details/mesh_tools.hpp"

namespace triqs::mesh {

  struct legendre_domain {
    using point_t = int; // fix this to long (but then conflict of is within boundary)

    double beta              = 0.0;
    statistic_enum statistic = Fermion;
    long n_max               = 1; // Note: includes endpoint. Change from previous?

    [[nodiscard]] constexpr point_t min() const { return 0; }
    [[nodiscard]] point_t max() const { return n_max; }

    // Ctor
    legendre_domain() = default;
    legendre_domain(double beta_, statistic_enum statistic_, long n_max_) : beta{beta_}, statistic{statistic_}, n_max{n_max_} {};

    [[nodiscard]] bool contains(point_t const &pt) const { return (0 <= pt) && (pt <= n_max); };

    [[nodiscard]] long size() const { return n_max + 1; };

    bool operator==(legendre_domain const &) const = default;

    [[nodiscard]] std::string hdf5_format() const { return "LegendreDomain"; }

    /// Write into HDF5
    friend void h5_write(h5::group fg, std::string const &subgroup_name, legendre_domain const &d) {
      h5::group gr = fg.create_group(subgroup_name);
      h5_write(gr, "n_max", d.n_max);
      h5_write(gr, "beta", d.beta);
      h5_write(gr, "statistic", (d.statistic == Fermion ? "F" : "B"));
    }

    /// Read from HDF5
    friend void h5_read(h5::group fg, std::string const &subgroup_name, legendre_domain &d) {
      h5::group gr          = fg.open_group(subgroup_name);
      std::string statistic = " ";
      h5_read(gr, "n_max", d.n_max);
      h5_read(gr, "beta", d.beta);
      h5_read(gr, "statistic", statistic);
      d.statistic = "F" ? Fermion : Boson;
    }
  };
} // namespace triqs::mesh
