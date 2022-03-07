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
    using point_t = long;

    double beta              = 0.0;
    statistic_enum statistic = Fermion;
    size_t n_max             = 1; // Is this inclusive order not??

    // Do we need this ctor??
    legendre_domain() = default;
    legendre_domain(double beta_, statistic_enum statistic_, point_t n_max_) : beta{beta_}, statistic{statistic_}, n_max{n_max_} {};

    [[nodiscard]] bool is_in_domain(point_t const &pt) const { return (0 <= pt) && (pt <= n_max); };

    [[nodiscard]] size_t size() const { return n_max; };

    bool operator==(legendre_domain const &D) const {
      return ((std::abs(beta - D.beta) < 1.e-15) && (statistic == D.statistic) && (n_max == D.n_max));
    }
    bool operator!=(legendre_domain const &) const = default;

    /// Write into HDF5
    friend void h5_write(h5::group fg, std::string_view subgroup_name, legendre_domain const &d) {
      h5::group gr = fg.create_group(std::string{subgroup_name});
      h5_write(gr, "n_max", d.n_max);
      h5_write(gr, "beta", d.beta);
      h5_write(gr, "statistic", (d.statistic == Fermion ? "F" : "B"));
    }

    std::string hdf5_format() const { return "LegendreDomain"; }

    /// Read from HDF5
    friend void h5_read(h5::group fg, std::string_view subgroup_name, legendre_domain &d) {
      h5::group gr = fg.open_group(std::string{subgroup_name});
      size_t n;
      double beta;
      std::string statistic{};
      h5_read(gr, "n_max", n);
      h5_read(gr, "beta", beta);
      h5_read(gr, "statistic", statistic);
      d = legendre_domain{beta, (statistic == "F" ? Fermion : Boson), n};
    }

    //  BOOST Serialization
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive &ar, const unsigned int version) {
      ar &n_max;
      ar &beta;
      ar &statistic;
    }
  };
} // namespace triqs::mesh
