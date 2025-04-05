//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos_dustfluidss.cpp
//  \brief implements functions in EquationOfState class for dustfluidss

// C headers

// C++ headers
#include <cmath>   // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../dustfluids/dustfluids_diffusion_cc/cell_center_diffusions.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::DustFluidsConservedToPrimitive(AthenaArray<Real> &cons_df,
//!           const AthenaArray<Real> &prim_df_old,
//!           AthenaArray<Real> &r, Coordinates *pco,
//!           int il, int iu, int jl, int ju, int kl, int ku)
//! \brief Converts conserved into primitive dust fluids variables

void EquationOfState::DustFluidsConservedToPrimitive(
  AthenaArray<Real> &cons_df, const AthenaArray<Real> &cons_diff,
  const AthenaArray<Real> &prim_df_old, AthenaArray<Real> &prim_df,
  Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {

  MeshBlock  *pmb = pmy_block_;
  DustFluids *pdf = pmb->pdustfluids;

  bool diffusion_correction = ((pdf->dfdif.dustfluids_diffusion_defined) &&
                               (pdf->dfdif.Momentum_Diffusion_Flag));

  //bool diffusion_correction = false;

  if (diffusion_correction) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real &cons_dust_dens = cons_df(rho_id, k, j, i);
            Real &cons_dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &cons_dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &cons_dust_mom3 = cons_df(v3_id,  k, j, i);

            Real &prim_dust_rho  = prim_df(rho_id, k, j, i);
            Real &prim_dust_vel1 = prim_df(v1_id,  k, j, i);
            Real &prim_dust_vel2 = prim_df(v2_id,  k, j, i);
            Real &prim_dust_vel3 = prim_df(v3_id,  k, j, i);

            const Real &diff_dust_mom1 = cons_diff(v1_id, k, j, i);
            const Real &diff_dust_mom2 = cons_diff(v2_id, k, j, i);
            const Real &diff_dust_mom3 = cons_diff(v3_id, k, j, i);

            // apply dust fluids floor to conserved variable first, then transform:
            // (multi-D fluxes may have caused it to drop below floor)
            cons_dust_dens = (cons_dust_dens > dustfluids_floor_[dust_id]) ? cons_dust_dens : dustfluids_floor_[dust_id];
            prim_dust_rho  = cons_dust_dens;

            Real inv_dust_dens = 1.0/cons_dust_dens;
            prim_dust_vel1 = (cons_dust_mom1 - diff_dust_mom1)*inv_dust_dens;
            prim_dust_vel2 = (cons_dust_mom2 - diff_dust_mom2)*inv_dust_dens;
            prim_dust_vel3 = (cons_dust_mom3 - diff_dust_mom3)*inv_dust_dens;
          }
        }
      }
    }
  } else {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real &cons_dust_dens = cons_df(rho_id, k, j, i);
            Real &cons_dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &cons_dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &cons_dust_mom3 = cons_df(v3_id,  k, j, i);

            Real &prim_dust_rho  = prim_df(rho_id, k, j, i);
            Real &prim_dust_vel1 = prim_df(v1_id,  k, j, i);
            Real &prim_dust_vel2 = prim_df(v2_id,  k, j, i);
            Real &prim_dust_vel3 = prim_df(v3_id,  k, j, i);

            // apply dust fluids floor to conserved variable first, then transform:
            // (multi-D fluxes may have caused it to drop below floor)
            cons_dust_dens = (cons_dust_dens > dustfluids_floor_[dust_id]) ? cons_dust_dens : dustfluids_floor_[dust_id];
            prim_dust_rho  = cons_dust_dens;

            Real inv_dust_dens = 1.0/cons_dust_dens;
            prim_dust_vel1 = cons_dust_mom1*inv_dust_dens;
            prim_dust_vel2 = cons_dust_mom2*inv_dust_dens;
            prim_dust_vel3 = cons_dust_mom3*inv_dust_dens;
          }
        }
      }
    }
  }
  return;
}



void EquationOfState::DustFluidsConservedToPrimitiveCellAverage(
    AthenaArray<Real> &cons_df, const AthenaArray<Real> &prim_df_old, AthenaArray<Real> &prim_df,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  MeshBlock  *pmb = pmy_block_;
  DustFluids *pdf = pmb->pdustfluids;
  int nl = 0; int nu = NDUSTVARS - 1;
  // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so this should factor out
  // TODO(felker): also, this may need to be dx1v, since Laplacian is cell-centered
  Real h = pco->dx1f(il);  // pco->dx1f(i); inside loop
  Real C = (h*h)/24.0;

  // Fourth-order accurate approx to cell-centered conserved and primitive variables
  //AthenaArray<Real> &w_cc = ph->w_cc, &w = ph->w; // &u_cc = ph->u_cc;
  AthenaArray<Real> &prim_df_cc = pdf->df_w_cc, &cons_df_cc = pdf->df_u_cc;
  AthenaArray<Real> &cons_diff_cc = pdf->dfccdif.diff_mom_cc;
  // Laplacians of cell-averaged conserved and 2nd order accurate primitive variables
  AthenaArray<Real> &laplacian_cc = pdf->scr1_nkji_;

  // Compute and store Laplacian of cell-averaged conserved variables
  pco->Laplacian(cons_df, laplacian_cc, il, iu, jl, ju, kl, ku, nl, nu);

  // Compute fourth-order approximation to cell-centered conserved variables
  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // We do not actually need to store all cell-centered conserved variables,
          // but the ConservedToPrimitive() implementation operates on 4D arrays
          cons_df_cc(n,k,j,i) = cons_df(n,k,j,i) - C*laplacian_cc(n,k,j,i);
        }
      }
    }
  }

  // Compute Laplacian of 2nd-order approximation to cell-averaged primitive variables
  pco->Laplacian(prim_df, laplacian_cc, il, iu, jl, ju, kl, ku, nl, nu);

  // Convert cell-centered conserved values to cell-centered primitive values
  DustFluidsConservedToPrimitive(cons_df_cc, cons_diff_cc, prim_df_old, prim_df_cc, pco, il, iu,
                                jl, ju, kl, ku);

  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // Compute fourth-order approximation to cell-averaged primitive variables
          prim_df(n,k,j,i) = prim_df_cc(n,k,j,i) + C*laplacian_cc(n,k,j,i);
        }
      }
    }
  }

  // Reapply primitive variable floors
  // Cannot fuse w/ above loop since floors are applied to all NHYDRO variables at once
  for (int n=nl; n<=nu; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          ApplyDustFluidsPrimitiveConservedFloors(cons_df, prim_df, n, k, j, i);
        }
      }
    }
  }

  return;
}


//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::DustFluidsPrimitiveToConserved(const AthenaArray<Real> &prim_df
//           AthenaArray<Real> &cons_df, Coordinates *pco,
//           int il, int iu, int jl, int ju, int kl, int ku);
// \brief Converts primitive variables into conservative variables

void EquationOfState::DustFluidsPrimitiveToConserved(
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &cons_diff,
    AthenaArray<Real> &cons_df, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {

  MeshBlock  *pmb = pmy_block_;
  DustFluids *pdf = pmb->pdustfluids;

  bool diffusion_correction = ((pdf->dfdif.dustfluids_diffusion_defined) &&
                               (pdf->dfdif.Momentum_Diffusion_Flag));

  if (diffusion_correction) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real& dust_dens = cons_df(rho_id, k, j, i);
            Real& dust_mom1 = cons_df(v1_id,  k, j, i);
            Real& dust_mom2 = cons_df(v2_id,  k, j, i);
            Real& dust_mom3 = cons_df(v3_id,  k, j, i);

            const Real& dust_rho  = prim_df(rho_id, k, j, i);
            const Real& dust_vel1 = prim_df(v1_id,  k, j, i);
            const Real& dust_vel2 = prim_df(v2_id,  k, j, i);
            const Real& dust_vel3 = prim_df(v3_id,  k, j, i);

            const Real &diff_dust_mom1 = cons_diff(v1_id, k, j, i);
            const Real &diff_dust_mom2 = cons_diff(v2_id, k, j, i);
            const Real &diff_dust_mom3 = cons_diff(v3_id, k, j, i);

            dust_dens = dust_rho;
            dust_mom1 = dust_vel1*dust_dens + diff_dust_mom1;
            dust_mom2 = dust_vel2*dust_dens + diff_dust_mom2;
            dust_mom3 = dust_vel3*dust_dens + diff_dust_mom3;
          }
        }
      }
    }
  } else {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real& dust_dens = cons_df(rho_id, k, j, i);
            Real& dust_mom1 = cons_df(v1_id,  k, j, i);
            Real& dust_mom2 = cons_df(v2_id,  k, j, i);
            Real& dust_mom3 = cons_df(v3_id,  k, j, i);

            const Real& dust_rho  = prim_df(rho_id, k, j, i);
            const Real& dust_vel1 = prim_df(v1_id,  k, j, i);
            const Real& dust_vel2 = prim_df(v2_id,  k, j, i);
            const Real& dust_vel3 = prim_df(v3_id,  k, j, i);

            dust_dens = dust_rho;
            dust_mom1 = dust_vel1*dust_dens;
            dust_mom2 = dust_vel2*dust_dens;
            dust_mom3 = dust_vel3*dust_dens;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ApplyDustFluidsFloors(AthenaArray<Real> &prim, int n,
//                                                     int k, int j, int i)
// \brief Apply species concentration floor to cell-averaged DUSTFLUIDS or
// reconstructed L/R cell inteprim_dface states (if PPM is used, e.g.) along:
// (NDUSTVARS x) x1 slices

void EquationOfState::ApplyDustFluidsFloors(AthenaArray<Real> &prim_df, int n, int k, int j, int i) {

  int dust_id = n/4;
  int rho_id  = 4*dust_id;

  Real &rho_dust = prim_df(rho_id,i);
  // apply dust fluids density floor (df_w(rho_id)) WITHOUT adjusting DUSTFLUIDS
  // mass (conserved), unlike in floor in standard EOS
  rho_dust = (rho_dust > dustfluids_floor_[dust_id]) ?  rho_dust : dustfluids_floor_[dust_id];
  return;
}


void EquationOfState::ApplyDustFluidsPrimitiveConservedFloors(
    AthenaArray<Real> &cons_df, AthenaArray<Real> &prim_df,
    int n, int k, int j, int i) {

  int dust_id = n/4;
  int rho_id  = 4*dust_id;

  Real &den_dust = cons_df(rho_id, k, j, i);
  Real &rho_dust = prim_df(rho_id, k, j, i);

  den_dust = (den_dust > dustfluids_floor_[dust_id]) ?  den_dust : dustfluids_floor_[dust_id];
  rho_dust = den_dust;

  return;
}
