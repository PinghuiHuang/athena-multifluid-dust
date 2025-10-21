//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_cellcenter_diffusion.cpp
//! \brief Compute dust fluids cell centered diffusive fluxes corresponding to concentration diffusion.

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
#include "../dustfluids_diffusion/dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


void DustFluidsCellCenterDiffusion::CalculateDiffusiveMomentum(const AthenaArray<Real> &prim_df,
            const AthenaArray<Real> &w) {
  DustFluids *pdf = pmb_->pdustfluids;
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;

  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  // i-direction
  const AthenaArray<Real> &x1flux = pdf->dfdif.dustfluids_diffusion_flux[X1DIR];

  // i-direction
  il = is - 2, iu = ie + 2, jl = js, ju = je, kl = ks, ku = ke;
  if (f2) {
    if (!f3) // 2D
      jl = js - 2, ju = je + 2, kl = ks, ku = ke;
    else // 3D
      jl = js - 2, ju = je + 2, kl = ks - 2, ku = ke + 2;
  }

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          diff_mom_cc(v1_id, k, j, i) = 0.5*(x1flux(rho_id, k, j, i) + x1flux(rho_id, k, j, i+1));
        }
      }
    }
  }

  // j-direction
  il = is, iu = ie, jl = js - 2, ju = je + 2, kl = ks, ku = ke;
  if (!f3) // 2D
    il = is - 2, iu = ie + 2, kl = ks, ku = ke;
  else // 3D
    il = is - 2, iu = ie + 2, kl = ks - 2, ku = ke + 2;

  if (f2) { // 2D or 3D
    const AthenaArray<Real> &x2flux = pdf->dfdif.dustfluids_diffusion_flux[X2DIR];
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v2_id   = rho_id + 2;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            diff_mom_cc(v2_id, k, j, i) = 0.5*(x2flux(rho_id, k, j, i) + x2flux(rho_id, k, j+1, i));
          }
        }
      }
    } // zero flux for 1D
  }

  // k-direction
  il = is, iu = ie, jl = js, ju = je, kl = ks - 2, ku = ke + 2;
  if (f2) // 2D or 3D
    il = is - 2, iu = ie + 2, jl = js - 2, ju = je + 2;
  else // 1D
    il = is - 2, iu = ie + 2;

  if (f3) { // 3D
    const AthenaArray<Real> &x3flux = pdf->dfdif.dustfluids_diffusion_flux[X3DIR];
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            diff_mom_cc(v3_id, k, j, i) = 0.5*(x3flux(rho_id, k, j, i) + x3flux(rho_id, k+1, j, i));
          }
        }
      }
    } // zero flux for 1D/2D
  }
  return;
}
