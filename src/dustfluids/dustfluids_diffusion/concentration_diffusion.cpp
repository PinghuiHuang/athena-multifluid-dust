//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file concentration_diffusion.cpp
//! \brief Compute dust fluids diffusive fluxes corresponding to concentration diffusion.

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <limits>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../defs.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

void DustFluidsDiffusion::DustFluidsConcentrationDiffusiveFlux(const AthenaArray<Real> &prim_df,
            const AthenaArray<Real> &w, AthenaArray<Real> *df_diff_flux) {
  DustFluids *pdf = pmb_->pdustfluids;
  const bool f2   = pmb_->pmy_mesh->f2;
  const bool f3   = pmb_->pmy_mesh->f3;
  AthenaArray<Real> &x1flux = df_diff_flux[X1DIR];

  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  AthenaArray<Real> &nu_dust = pdf->nu_dustfluids_array;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (f2) {
    if (!f3) // 2D
      jl = js - 1, ju = je + 1, kl = ks, ku = ke;
    else // 3D
      jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
  }

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          Real nu_face      = 0.5*(nu_dust(dust_id,k,j,i) + nu_dust(dust_id,k,j,i-1));
          Real gas_rho_face = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i-1));

          // dprim_df_dx1 = D(rho_d/rho_g)_x1/D(x1)
          Real dprim_df_dx1 = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k,j,i-1)/w(IDN,k,j,i-1))
                                      /pco_->dx1v(i-1);
          x1flux(rho_id,k,j,i) -= nu_face*gas_rho_face*dprim_df_dx1;
        }
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (!f3) // 2D
    il = is - 1, iu = ie + 1, kl = ks, ku = ke;
  else // 3D
    il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;

  if (f2) { // 2D or 3D
    AthenaArray<Real> &x2flux = df_diff_flux[X2DIR];
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je+1; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real nu_face      = 0.5*(nu_dust(dust_id,k,j,i) + nu_dust(dust_id,k,j-1,i));
            Real gas_rho_face = 0.5*(w(IDN,k,j,i) + w(IDN,k,j-1,i));

            // dprim_df_dx2 = D(rho_d/rho_g)_x2/D(x2)
            Real dprim_df_dx2 = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k,j-1,i)/w(IDN,k,j-1,i))
                                      /pco_->h2v(i)/pco_->dx2v(j-1);
            x2flux(rho_id,k,j,i) -= nu_face*gas_rho_face*dprim_df_dx2;
          }
        }
      }
    } // zero flux for 1D
  }

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (f2) // 2D or 3D
    il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
  else // 1D
    il = is - 1, iu = ie + 1;

  if (f3) { // 3D
    AthenaArray<Real> &x3flux = df_diff_flux[X3DIR];
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real nu_face      = 0.5*(nu_dust(dust_id,k,j,i) + nu_dust(dust_id,k-1,j,i));
            Real gas_rho_face = 0.5*(w(IDN,k,j,i) + w(IDN,k-1,j,i));

            // dprim_df_dx3 = D(rho_d/rho_g)_x3/D(x3)
            Real dprim_df_dx3 = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k-1,j,i)/w(IDN,k-1,j,i))
                                      /pco_->dx3v(k-1)/pco_->h31v(i)/pco_->h32v(j);
            x3flux(rho_id,k,j,i) -= nu_face*gas_rho_face*dprim_df_dx3;
          }
        }
      }
    } // zero flux for 1D/2D
  }
  return;
}
