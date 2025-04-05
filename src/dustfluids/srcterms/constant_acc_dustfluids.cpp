//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file constant_acc_dustfluids.cpp
//! \brief source terms due to constant acceleration

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_srcterms.hpp"

//----------------------------------------------------------------------------------------
//! \fn void DustFluids::ConstantAccelerationDustFluids
//! \brief Adds source terms for constant acceleration to conserved variables

void DustFluidsSourceTerms::ConstantAccelerationDustFluids(const Real dt, const AthenaArray<Real> *flux_df,
                                            const AthenaArray<Real> &prim_df,
                                            AthenaArray<Real> &cons_df) {
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;

  // acceleration in 1-direction
  if (g1_!=0.0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real src = dt*prim_df(rho_id, k, j, i)*g1_;
            cons_df(v1_id, k, j, i) += src;
          }
        }
      }
    }
  }

  // acceleration in 2-direction
  if (g2_!=0.0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v2_id   = rho_id + 2;
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real src = dt*prim_df(rho_id, k, j, i)*g2_;
            cons_df(v2_id, k, j, i) += src;
          }
        }
      }
    }
  }

  // acceleration in 3-direction
  if (g3_!=0.0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v3_id   = rho_id + 3;
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real src = dt*prim_df(rho_id, k, j, i)*g3_;
            cons_df(v3_id, k, j, i) += src;
          }
        }
      }
    }
  }

  return;
}
