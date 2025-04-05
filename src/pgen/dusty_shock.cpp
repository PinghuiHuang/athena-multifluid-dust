//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shock_tube.cpp
//  \brief Problem generator for shock tube problems.
//
// Problem generator for shock tube (1-D Riemann) problems. Initializes plane-parallel
// shock along x1 (in 1D, 2D, 3D), along x2 (in 2D, 3D), and along x3 (in 3D).
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), freopen(), fprintf(), fclose()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if NON_BAROTROPIC_EOS
#error "This problem generator requires isothermal equation of state!"
#endif

#if (NDUSTFLUIDS == 0)
#error "This problem generator requires NDUSTFLUIDS > 1!"
#endif

// problem parameters which are useful to make global to this file
namespace {
Real user_dt, iso_cs, xshock, gamma_gas, vel_right;
Real MyTimeStep(MeshBlock *pmb);
Real wl[NHYDRO];
Real wr[NHYDRO];
Real K_para[NDUSTFLUIDS];
Real wl_d[4];
Real wr_d[4];
Real initial_D2G;
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);

} // namespace

void ShockInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void ShockOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real press(Real rho, Real T) {
  // Ionization fraction
  Real x = 2. /(1 + std::sqrt(1 + 4. * rho * std::exp(1. / T) * std::pow(T, -1.5)));
  return rho * T * (1. + x);
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  user_dt   = pin->GetOrAddReal("problem", "user_dt",         0.0);
  iso_cs    = pin->GetOrAddReal("hydro",   "iso_sound_speed", 1e-1);
  gamma_gas = pin->GetReal("hydro",        "gamma");

  for (int n=0; n<NDUSTFLUIDS; ++n)
    K_para[n] = pin->GetReal("dust", "K_para_" + std::to_string(n+1));

  initial_D2G = pin->GetReal("dust", "initial_D2G");

  xshock  = pin->GetReal("problem", "xshock");
  wl[IDN] = pin->GetReal("problem", "dl");
  wl[IVX] = pin->GetReal("problem", "ul");
  wl[IVY] = pin->GetReal("problem", "vl");
  wl[IVZ] = pin->GetReal("problem", "wl");

  wr[IDN] = pin->GetReal("problem", "dr");
  wr[IVX] = pin->GetReal("problem", "ur");
  wr[IVY] = pin->GetReal("problem", "vr");
  wr[IVZ] = pin->GetReal("problem", "wr");

  //wl_d[0] = pin->GetReal("dust", "dl_d");
  wl_d[0] = initial_D2G*wl[IDN];
  //wl_d[1] = pin->GetReal("dust", "ul_d");
  wl_d[1] = wl[IVX];
  wl_d[2] = pin->GetReal("dust", "vl_d");
  wl_d[3] = pin->GetReal("dust", "wl_d");

  //wr_d[0] = pin->GetReal("dust", "dr_d");
  wr_d[0] = initial_D2G*wr[IDN];
  //wr_d[1] = pin->GetReal("dust", "ur_d");
  wr_d[1] = wr[IVX];
  wr_d[2] = pin->GetReal("dust", "vr_d");
  wr_d[3] = pin->GetReal("dust", "wr_d");

  Real mach = wl[IVX]/iso_cs;
  vel_right = (1./(1. + initial_D2G))/(SQR(mach));
  std::cout << "vel_right = " << vel_right << std::endl;

  EnrollUserDustStoppingTime(MyStoppingTime);

  // Enroll user-defined time step
  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, ShockInnerX1);
  }

  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, ShockOuterX1);
  }

  return;
}

namespace {
Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      //int v1_id   = rho_id + 1;
      Real inv_K  = 1.0/K_para[dust_id];
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            //const Real &gas_rho  = prim(IDN, k, j, i);
            const Real &dust_rho = prim_df(rho_id, k, j, i);
            Real &st_time        = stopping_time(dust_id, k, j, i);
            //st_time              = inv_K*(dust_rho*gas_rho)/(dust_rho + gas_rho);
            st_time              = inv_K*dust_rho;

            //const Real &gas_vel1  = prim(IM1, k, j, i);
            //const Real &dust_vel1 = prim_df(v1_id, k, j, i);
            //st_time               = inv_K*dust_rho*gas_rho*(gas_vel1-dust_vel1)/(dust_rho+gas_rho);
          }
        }
      }
    }
  return;
}

}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the shock tube tests
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::stringstream msg;

  // parse shock direction: {1, 2, 3} -> {x1, x2, x3}
  int shk_dir = pin->GetOrAddInteger("problem","shock_dir", 1);

  // parse shock location (must be inside grid)
  Real xshock = pin->GetReal("problem","xshock");
  if (shk_dir == 1 && (xshock < pmy_mesh->mesh_size.x1min ||
                       xshock > pmy_mesh->mesh_size.x1max)) {
    msg << "### FATAL ERROR in Problem Generator" << std::endl << "xshock="
        << xshock << " lies outside x1 domain for shkdir=" << shk_dir << std::endl;
    ATHENA_ERROR(msg);
  }

  // Parse left state read from input file: dl, ul, vl, wl,[pl]
  Real wl[NHYDRO];
  Real wl_d[4];
  wl[IDN] = pin->GetReal("problem", "dl");
  wl[IVX] = pin->GetReal("problem", "ul");
  wl[IVY] = pin->GetReal("problem", "vl");
  wl[IVZ] = pin->GetReal("problem", "wl");

  //wl_d[0] = pin->GetReal("dust", "dl_d");
  //wl_d[1] = pin->GetReal("dust", "ul_d");
  wl_d[0] = initial_D2G*wl[IDN];
  wl_d[1] = wl[IVX];
  wl_d[2] = pin->GetReal("dust", "vl_d");
  wl_d[3] = pin->GetReal("dust", "wl_d");


  // Parse right state read from input file: dr, ur, vr, wr,[pr]
  Real wr[NHYDRO];
  Real wr_d[4];
  wr[IDN] = pin->GetReal("problem", "dr");
  wr[IVX] = pin->GetReal("problem", "ur");
  wr[IVY] = pin->GetReal("problem", "vr");
  wr[IVZ] = pin->GetReal("problem", "wr");

  //wr_d[0] = pin->GetReal("dust", "dr_d");
  //wr_d[1] = pin->GetReal("dust", "ur_d");
  wr_d[0] = initial_D2G*wr[IDN];
  wr_d[1] = wr[IVX];
  wr_d[2] = pin->GetReal("dust", "vr_d");
  wr_d[3] = pin->GetReal("dust", "wr_d");

  // Initialize the discontinuity in the Hydro and Dust fluids variables
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (pcoord->x1v(i) < xshock) {
          phydro->u(IDN, k, j, i) = wl[IDN];
          phydro->u(IM1, k, j, i) = wl[IVX]*wl[IDN];
          phydro->u(IM2, k, j, i) = wl[IVY]*wl[IDN];
          phydro->u(IM3, k, j, i) = wl[IVZ]*wl[IDN];

          if (NDUSTFLUIDS > 0) {
            for (int n = 0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;
              pdustfluids->df_u(rho_id, k, j, i) = wl_d[0];
              pdustfluids->df_u(v1_id, k, j, i)  = wl_d[1]*wl_d[0];
              pdustfluids->df_u(v2_id, k, j, i)  = wl_d[2]*wl_d[0];
              pdustfluids->df_u(v3_id, k, j, i)  = wl_d[3]*wl_d[0];
            }
          }
        } else {
          phydro->u(IDN, k, j, i) = wr[IDN];
          phydro->u(IM1, k, j, i) = wr[IVX]*wr[IDN];
          phydro->u(IM2, k, j, i) = wr[IVY]*wr[IDN];
          phydro->u(IM3, k, j, i) = wr[IVZ]*wr[IDN];

          if (NDUSTFLUIDS > 0) {
            for (int n = 0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;
              pdustfluids->df_u(rho_id, k, j, i) = wr_d[0];
              pdustfluids->df_u(v1_id,  k, j, i) = wr_d[1]*wr_d[0];
              pdustfluids->df_u(v2_id,  k, j, i) = wr_d[2]*wr_d[0];
              pdustfluids->df_u(v3_id,  k, j, i) = wr_d[3]*wr_d[0];
            }
          }
        }
      }
    }
  }

  return;
}


void ShockInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {

        prim(IDN, k, j, il-i) = wl[IDN];
        prim(IM1, k, j, il-i) = wl[IVX];
        prim(IM2, k, j, il-i) = wl[IVY];
        prim(IM3, k, j, il-i) = wl[IVZ];

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            prim_df(rho_id, k, j, il-i) = wl_d[0];
            prim_df(v1_id,  k, j, il-i) = wl_d[1];
            prim_df(v2_id,  k, j, il-i) = wl_d[2];
            prim_df(v3_id,  k, j, il-i) = wl_d[3];
          }
        }

      }
    }
  }
  return;
}

void ShockOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      for (int i=1; i<=ngh; ++i) {

        prim(IDN, k, j, iu+i) = wr[IDN];
        //prim(IM1, k, j, iu+i) = vel_right;
        prim(IM1, k, j, iu+i) = wr[IVX];
        prim(IM2, k, j, iu+i) = wr[IVY];
        prim(IM3, k, j, iu+i) = wr[IVZ];

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            prim_df(rho_id, k, j, iu+i) = wr_d[0];
            //prim_df(v1_id,  k, j, iu+i) = vel_right;
            prim_df(v1_id,  k, j, iu+i) = wr_d[1];
            prim_df(v2_id,  k, j, iu+i) = wr_d[2];
            prim_df(v3_id,  k, j, iu+i) = wr_d[3];
          }
        }

      }
    }
  }
  return;
}
