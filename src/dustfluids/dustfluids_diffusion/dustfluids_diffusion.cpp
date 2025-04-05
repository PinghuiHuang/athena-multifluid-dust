//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_diffusion.cpp
//! \brief Compute dustfluids fluxes corresponding to diffusion processes.

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <cstring>    // strcmp
#include <limits>
#include <sstream>
#include <string>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../defs.hpp"
#include "../../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../../mesh/mesh.hpp"
#include "../../orbital_advection/orbital_advection.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

class Hydro;
class HydroDiffusion;

DustFluidsDiffusion::DustFluidsDiffusion(DustFluids *pdf, ParameterInput *pin) :
  pmy_dustfluids_(pdf), pmb_(pmy_dustfluids_->pmy_block), pco_(pmb_->pcoord) {
  int nc1 = pmb_->ncells1, nc2 = pmb_->ncells2, nc3 = pmb_->ncells3;

  dustfluids_diffusion_defined = false;
  Diffusion_Flag = pin->GetBoolean("dust", "Diffusion_Flag");

  // Turn on dust diffusion
  if (Diffusion_Flag) {
    dustfluids_diffusion_defined = true;
    Momentum_Diffusion_Flag = pin->GetOrAddBoolean("dust", "Momentum_Diffusion_Flag", false);
  }

  // eddy time is set as 1.0 by default
  eddy_time_ = pin->GetOrAddReal("dust", "eddy_time", 1.0);

  if (dustfluids_diffusion_defined) {
    dustfluids_diffusion_flux[X1DIR].NewAthenaArray(NDUSTVARS, nc3,   nc2,   nc1+1); // Face centered x1 diffusive flux
    dustfluids_diffusion_flux[X2DIR].NewAthenaArray(NDUSTVARS, nc3,   nc2+1, nc1);   // Face centered x2 diffusive flux
    dustfluids_diffusion_flux[X3DIR].NewAthenaArray(NDUSTVARS, nc3+1, nc2,   nc1);   // Face centered x3 diffusive flux

    dx1_.NewAthenaArray(nc1);
    dx2_.NewAthenaArray(nc1);
    dx3_.NewAthenaArray(nc1);
    diff_tot_.NewAthenaArray(nc1);
  }
}


void DustFluidsDiffusion::CalcDustFluidsDiffusionFlux(
    const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &w_orb, const AthenaArray<Real> &prim_df_orb,
    const AthenaArray<Real> &u, const AthenaArray<Real> &cons_df) {
  DustFluids *pdf = pmy_dustfluids_;

  // Set the diffusive flux as zeros
  ClearDustFluidsFlux(dustfluids_diffusion_flux);

  // Calculate the concentration diffusive flux
  DustFluidsConcentrationDiffusiveFlux(prim_df_orb, w, dustfluids_diffusion_flux);

  // Calculate the correction of momentum diffusive flux due to concentration diffusion
  if (Momentum_Diffusion_Flag) {
    //// Calculate the advective diffusion and diffusive advection
    if (pmb_->porb->orbital_advection_defined)
      DustFluidsMomentumDiffusiveFlux(prim_df, w, dustfluids_diffusion_flux);
    else
      DustFluidsMomentumDiffusiveFlux(prim_df_orb, w_orb, dustfluids_diffusion_flux);

    // Calculate the cell center diffusive momentum
    pdf->dfccdif.diff_mom_cc.ZeroClear();
    pdf->dfccdif.CalculateDiffusiveMomentum(prim_df_orb, w_orb);
  }

  return;
}


// Add the dust diffusive fluxes into the dust fluxes
void DustFluidsDiffusion::AddDustFluidsDiffusionFlux(AthenaArray<Real> *flux_diff,
                    AthenaArray<Real> *flux_df) {
  // flux_diff: diffusive flux, flux_df: total flux
  int size1 = flux_df[X1DIR].GetSize();
#pragma omp simd
  for (int i=0; i<size1; ++i)
    flux_df[X1DIR](i) += flux_diff[X1DIR](i);

  if (pmb_->pmy_mesh->f2) {
    int size2 = flux_df[X2DIR].GetSize();
#pragma omp simd
    for (int i=0; i<size2; ++i)
      flux_df[X2DIR](i) += flux_diff[X2DIR](i);
  }

  if (pmb_->pmy_mesh->f3) {
    int size3 = flux_df[X3DIR].GetSize();
#pragma omp simd
    for (int i=0; i<size3; ++i)
      flux_df[X3DIR](i) += flux_diff[X3DIR](i);
  }

  return;
}


//! \fn void DustFluidsDiffusion::ClearDustFluidsFlux
//! \brief Reset dust diffusive fluxes back to zeros
void DustFluidsDiffusion::ClearDustFluidsFlux(AthenaArray<Real> *flux_diff) {
  flux_diff[X1DIR].ZeroClear();
  flux_diff[X2DIR].ZeroClear();
  flux_diff[X3DIR].ZeroClear();
  return;
}


// Calculate the parabolic time step due to dust diffusions
Real DustFluidsDiffusion::NewDiffusionDt() {
  DustFluids *pdf = pmy_dustfluids_;
  Real real_max   = std::numeric_limits<Real>::max();
  const bool f2   = pmb_->pmy_mesh->f2;
  const bool f3   = pmb_->pmy_mesh->f3;
  int il = pmb_->is - NGHOST; int jl = pmb_->js; int kl = pmb_->ks;
  int iu = pmb_->ie + NGHOST; int ju = pmb_->je; int ku = pmb_->ke;
  int dust_id, rho_id, v1_id, v2_id, v3_id;
  Real fac;
  if (f3) fac = 1.0/6.0;
  else if (f2) fac = 0.25;
  else fac = 0.5;

  Real dt_df_diff = real_max;

  AthenaArray<Real> &diff_t = diff_tot_;
  AthenaArray<Real> &len = dx1_, &dx2 = dx2_, &dx3 = dx3_;

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    dust_id = n;
    rho_id  = 4*dust_id;
    v1_id   = rho_id + 1;
    v2_id   = rho_id + 2;
    v3_id   = rho_id + 3;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          diff_t(i) = 0.0;
        }
#pragma omp simd
        for (int i=il; i<=iu; ++i)
          diff_t(i) += pdf->nu_dustfluids_array(dust_id,k,j,i);
        pco_->CenterWidth1(k, j, il, iu, len);
        pco_->CenterWidth2(k, j, il, iu, dx2);
        pco_->CenterWidth3(k, j, il, iu, dx3);
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          len(i) = (f2) ? std::min(len(i), dx2(i)) : len(i);
          len(i) = (f3) ? std::min(len(i), dx3(i)) : len(i);
        }
        if (dustfluids_diffusion_defined) {
          for (int i=il; i<=iu; ++i)
            dt_df_diff = std::min(dt_df_diff, static_cast<Real>(
                SQR(len(i))*fac/(diff_t(i) + TINY_NUMBER)));
        }
      }
    }
  }
  return dt_df_diff;
}
