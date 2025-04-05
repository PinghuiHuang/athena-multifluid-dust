//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file default_pgen.cpp
//  \brief Provides default (empty) versions of all functions in problem generator files
//  This means user does not have to implement these functions if they are not needed.
//
// The attribute "weak" is used to ensure the loader selects the user-defined version of
// functions rather than the default version given here.
//
// The attribute "alias" may be used with the "weak" functions (in non-defining
// declarations) in order to have them refer to common no-operation function definition in
// the same translation unit. Target function must be specified by mangled name unless C
// linkage is specified.
//
// This functionality is not in either the C nor the C++ standard. These GNU extensions
// are largely supported by LLVM, Intel, IBM, but may affect portability for some
// architecutres and compilers. In such cases, simply define all 6 of the below class
// functions in every pgen/*.cpp file (without any function attributes).

// C headers

// C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if NON_BAROTROPIC_EOS
#error "This problem generator requires isothermal equation of state!"
#endif

// problem parameters which are useful to make global to this file
namespace {
// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real rho_g0, p0, u0, v0, w0, bx0, by0, bz0, dby, dbz;
Real delta_rho_gas_real, delta_rho_gas_imag;
Real delta_vel_gas_real, delta_vel_gas_imag;
Real user_dt;
Real amp, lambda, k_par;              // amplitude, Wavelength, 2*PI/wavelength
Real gam, gm1, iso_cs, vflow;
Real initial_D2G[NDUSTFLUIDS];
Real delta_rho_dust_real[NDUSTFLUIDS];
Real delta_rho_dust_imag[NDUSTFLUIDS];
Real delta_vel_dust_real[NDUSTFLUIDS];
Real delta_vel_dust_imag[NDUSTFLUIDS];
Real MyTimeStep(MeshBlock *pmb);
} // namespace

// 3x members of Mesh class:

namespace {
Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // read global parameters
  rho_g0   = pin->GetOrAddReal("problem", "rhog0",      1.0);
  user_dt = pin->GetOrAddReal("time",    "user_dt", 1.375e-2);
  amp     = pin->GetReal("problem",      "amp");
  vflow   = pin->GetOrAddReal("problem", "vflow",   0.0);
  iso_cs  = pin->GetReal("hydro",        "iso_sound_speed");

  delta_rho_gas_real = pin->GetReal("problem", "delta_rho_gas_real");
  delta_rho_gas_imag = pin->GetReal("problem", "delta_rho_gas_imag");
  delta_vel_gas_real = pin->GetReal("problem", "delta_vel_gas_real");
  delta_vel_gas_imag = pin->GetReal("problem", "delta_vel_gas_imag");

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      delta_rho_dust_real[n] = pin->GetReal("dust", "delta_rho_d_" + std::to_string(n+1) + "_real");
      delta_rho_dust_imag[n] = pin->GetReal("dust", "delta_rho_d_" + std::to_string(n+1) + "_imag");

      delta_vel_dust_real[n] = pin->GetReal("dust", "delta_vel_d_" + std::to_string(n+1) + "_real");
      delta_vel_dust_imag[n] = pin->GetReal("dust", "delta_vel_d_" + std::to_string(n+1) + "_imag");

      initial_D2G[n]         = pin->GetReal("dust", "Initial_D2G_" + std::to_string(n+1));
    }
  }

  if (NON_BAROTROPIC_EOS) {
    gam = pin->GetReal("hydro", "gamma");
    gm1 = (gam - 1.0);
  }

  Real x1size = mesh_size.x1max - mesh_size.x1min;
  Real x2size = mesh_size.x2max - mesh_size.x2min;
  Real x3size = mesh_size.x3max - mesh_size.x3min;

  Real x1 = x1size;
  Real x2 = 0.0;
  Real x3 = 0.0;

  lambda = x1;
  k_par  = 2.0*(PI)/lambda;
  u0     = vflow;
  v0     = 0.0;
  w0     = 0.0;
  p0     = SQR(iso_cs)*rho_g0;

  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);
  return;
}


void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  //Real inv_gm1 = 1./gm1;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x  = pcoord->x1v(i);
        Real sn = std::sin(k_par*x);
        Real cn = std::cos(k_par*x);

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        Real delta_rho = amp*rho_g0*(cn*delta_rho_gas_real - sn*delta_rho_gas_imag);
        Real delta_vel = amp*iso_cs*(cn*delta_vel_gas_real - sn*delta_vel_gas_imag);
        gas_dens       = rho_g0 + delta_rho;
        gas_mom1       = gas_dens*(u0 + delta_vel);
        gas_mom2       = 0.0;
        gas_mom3       = 0.0;

        if (NDUSTFLUIDS >0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
            Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
            Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
            Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

            Real rho_d0         = initial_D2G[dust_id] * rho_g0;
            Real delta_dust_rho = amp*rho_g0*(cn*delta_rho_dust_real[n] - sn*delta_rho_dust_imag[n]);
            Real delta_dust_vel = amp*iso_cs*(cn*delta_vel_dust_real[n] - sn*delta_vel_dust_imag[n]);

            dust_dens = rho_d0 + delta_dust_rho;
            dust_mom1 = dust_dens*(u0 + delta_dust_vel);
            dust_mom2 = 0.0;
            dust_mom3 = 0.0;
          }
        }
      }
    }
  }
  return;
}
