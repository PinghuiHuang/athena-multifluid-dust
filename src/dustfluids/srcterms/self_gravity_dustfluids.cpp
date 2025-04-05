//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file self_gravity.cpp
//! \brief source terms due to self-gravity

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../gravity/gravity.hpp"
#include "../../mesh/mesh.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn void DustFluidsSourceTerms::SelfGravityDustFluids
//! \brief Adds source terms for self-gravitational acceleration to conserved variables
//! \note
//! This implements the source term formula in Mullen, Hanawa and Gammie 2020, but only
//! for the momentum part. The energy source term is not conservative in this version.
//! I leave the fully conservative formula for later as it requires design consideration.
//! Also note that this implementation is not exactly conservative when the potential
//! contains a residual error (Multigrid has small but non-zero residual).

void DustFluidsSourceTerms::SelfGravityDustFluids(const Real dt,
              const AthenaArray<Real> *flux_df, const AthenaArray<Real> &prim_df,
              AthenaArray<Real> &cons_df) {
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;
  Gravity *pgrav = pmb->pgrav;

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;

    // acceleration in 1-direction
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real dx1    = pmb->pcoord->dx1v(i);
          Real dtodx1 = dt/dx1;
          Real phic = pgrav->phi(k, j, i);
          Real phil = 0.5*(pgrav->phi(k, j, i-1)+pgrav->phi(k, j, i  ));
          Real phir = 0.5*(pgrav->phi(k, j, i  )+pgrav->phi(k, j, i+1));
          cons_df(v1_id, k, j, i) -= dtodx1*prim_df(rho_id, k, j, i)*(phir-phil);
        }
      }
    }

    if (pmb->block_size.nx2 > 1) {
      // acceleration in 2-direction
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real dx2     = pmb->pcoord->dx2v(j);
            Real hdtodx2 = 0.5*dt/dx2;
            Real dpl = -(pgrav->phi(k, j,   i) - pgrav->phi(k, j-1, i));
            Real dpr = -(pgrav->phi(k, j+1, i) - pgrav->phi(k, j,   i));
            cons_df(v2_id, k, j, i) += hdtodx2 * prim_df(rho_id, k, j, i) * (dpl + dpr);
          }
        }
      }
    }

    if (pmb->block_size.nx3 > 1) {
      // acceleration in 3-direction
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real dx3     = pmb->pcoord->dx3v(k);
            Real hdtodx3 = 0.5*dt/dx3;
            Real dpl = -(pgrav->phi(k,   j, i) - pgrav->phi(k-1, j, i));
            Real dpr = -(pgrav->phi(k+1, j, i) - pgrav->phi(k,   j, i));
            cons_df(v3_id, k, j, i) += hdtodx3 * prim_df(rho_id, k, j, i) * (dpl + dpr);
          }
        }
      }
    }
  }
  return;
}
