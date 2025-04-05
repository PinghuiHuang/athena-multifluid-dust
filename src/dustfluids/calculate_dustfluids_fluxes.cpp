//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file calculate_dust_fluid_fluxes.cpp
//! \brief Calculate dust fluids fluxes

// C headers

// C++ headers
#include <algorithm>   // min,max

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"   // reapply floors to face-centered reconstructed states
#include "../reconstruct/reconstruction.hpp"
#include "dustfluids.hpp"
#include "dustfluids_diffusion/dustfluids_diffusion.hpp"
#include "dustfluids_diffusion_cc/cell_center_diffusions.hpp"
#include "dustfluids_drags/dust_gas_drag.hpp"
#include "srcterms/dustfluids_srcterms.hpp"


// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void DustFluids::CalculateFluxes
//! \brief Calculate dust fluids fluxes using reconstruction

void DustFluids::CalculateDustFluidsFluxes(AthenaArray<Real> &prim_df, const int order) {
  MeshBlock *pmb = pmy_block;
  AthenaArray<Real> &x1flux = df_flux[X1DIR];
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int il, iu, jl, ju, kl, ku;

  AthenaArray<Real> &flux_fc          = scr1_nkji_;
  AthenaArray<Real> &laplacian_all_fc = scr2_nkji_;

  //--------------------------------------------------------------------------------------
  // i-direction

  // set the loop limits
  jl = js, ju = je, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
    if (pmb->block_size.nx2 > 1) {
      if (pmb->block_size.nx3 == 1) // 2D
        jl = js-1, ju = je+1, kl = ks, ku = ke;
      else // 3D
        jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
    }
  //}

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      // reconstruct L/R states
      if (order == 1) {
        pmb->precon->DonorCellX1_DustFluids(k, j, is-1, ie+1, prim_df, df_w_l_, df_w_r_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX1_DustFluids(k, j, is-1, ie+1, prim_df, df_w_l_, df_w_r_);
      } else {
        pmb->precon->PiecewiseParabolicX1_DustFluids(k, j, is-1, ie+1, prim_df, df_w_l_, df_w_r_);
        for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            pmb->peos->ApplyDustFluidsFloors(df_w_l_, n, k, j, i);
            pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
          }
        }
      }

      if (solver_id == 0)
        RiemannSolverDustFluids_Penetration(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);
      else if (solver_id == 1)
        RiemannSolverDustFluids_noPenetration(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);
      else if (solver_id == 2)
        HLLENoCsRiemannSolverDustFluids(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);
      else
        HLLERiemannSolverDustFluids(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);

      if (order == 4) {
        for (int n=0; n<NDUSTVARS; n++) {
          for (int i=is; i<=ie+1; i++) {
            df_w_l3d_(n, k, j, i) = df_w_l_(n, i);
            df_w_r3d_(n, k, j, i) = df_w_r_(n, i);
          }
        }
      }
    }
  }

  if (order == 4) {
    // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so this should factor out
    // TODO(felker): also, this may need to be dx1v, since Laplacian is cell-centered
    Real h = pmb->pcoord->dx1f(is);  // pco->dx1f(i); inside loop
    Real C = (h*h)/24.0;

    // construct Laplacian from x1flux
    pmb->pcoord->LaplacianX1All(x1flux, laplacian_all_fc, 0, NDUSTVARS-1,
        kl, ku, jl, ju, is, ie+1);

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        // Compute Laplacian of x1 face states
        for (int n=0; n<NDUSTVARS; ++n) {
          pmb->pcoord->LaplacianX1(df_w_l3d_, laplacian_l_df_fc_, n, k, j, is, ie+1);
          pmb->pcoord->LaplacianX1(df_w_r3d_, laplacian_r_df_fc_, n, k, j, is, ie+1);
#pragma omp simd
          for (int i=is; i<=ie+1; ++i) {
            df_w_l_(n,i) = df_w_l3d_(n,k,j,i) - C*laplacian_l_df_fc_(i);
            df_w_r_(n,i) = df_w_r3d_(n,k,j,i) - C*laplacian_r_df_fc_(i);
            pmb->peos->ApplyDustFluidsFloors(df_w_l_, n, k, j, i);
            pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
          }
        }

        // Compute x1 interface fluxes from face-centered primitive variables
        if (solver_id == 0)
          RiemannSolverDustFluids_Penetration(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);
        else if (solver_id == 1)
          RiemannSolverDustFluids_noPenetration(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);
        else if (solver_id == 2)
          HLLENoCsRiemannSolverDustFluids(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);
        else
          HLLERiemannSolverDustFluids(k, j, is, ie+1, 1, df_w_l_, df_w_r_, x1flux);

        // Apply Laplacian of second-order accurate face-averaged flux on x1 faces
        for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
          for (int i=is; i<=ie+1; i++)
            x1flux(n, k, j, i) = flux_fc(n, k, j, i) + C*laplacian_all_fc(n, k, j, i);
        }
      }
    }
  } // end if (order == 4)
  //------------------------------------------------------------------------------

  //--------------------------------------------------------------------------------------
  // j-direction

  if (pmb->pmy_mesh->f2) {
    AthenaArray<Real> &x2flux = df_flux[X2DIR];

    // set the loop limits
    il = is-1, iu = ie+1, kl = ks, ku = ke;
    //if (MAGNETIC_FIELDS_ENABLED) {
      if (pmb->block_size.nx3 == 1) // 2D
        kl = ks, ku = ke;
      else // 3D
        kl = ks-1, ku = ke+1;
    //}

    for (int k=kl; k<=ku; ++k) {
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX2_DustFluids(k, js-1, il, iu, prim_df, df_w_l_, df_w_r_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX2_DustFluids(k, js-1, il, iu, prim_df, df_w_l_, df_w_r_);
      } else {
        pmb->precon->PiecewiseParabolicX2_DustFluids(k, js-1, il, iu, prim_df, df_w_l_, df_w_r_);
        for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->peos->ApplyDustFluidsFloors(df_w_l_, n, k, js-1, i);
            //pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
          }
        }
      }

      for (int j=js; j<=je+1; ++j) {
        // reconstruct L/R states at j
        if (order == 1) {
          pmb->precon->DonorCellX2_DustFluids(k, j, il, iu, prim_df, df_w_lb_, df_w_r_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX2_DustFluids(k, j, il, iu, prim_df, df_w_lb_, df_w_r_);
        } else {
          pmb->precon->PiecewiseParabolicX2_DustFluids(k, j, il, iu, prim_df, df_w_lb_, df_w_r_);
          for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              pmb->peos->ApplyDustFluidsFloors(df_w_lb_, n, k, j, i);
              pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
            }
          }
        }

        if (solver_id == 0)
          RiemannSolverDustFluids_Penetration(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);
        else if (solver_id == 1)
          RiemannSolverDustFluids_noPenetration(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);
        else if (solver_id == 2)
          HLLENoCsRiemannSolverDustFluids(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);
        else
          HLLERiemannSolverDustFluids(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);

        if (order == 4) {
          for (int n=0; n<NDUSTVARS; n++) {
            for (int i=il; i<=iu; i++) {
              df_w_l3d_(n, k, j, i) = df_w_l_(n, i);
              df_w_r3d_(n, k, j, i) = df_w_r_(n, i);
            }
          }
        }

        // swap the arrays for the next step
        df_w_l_.SwapAthenaArray(df_w_lb_);
      }
    }
    if (order == 4) {
      // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so factor this out
      // TODO(felker): also, this may need to be dx2v, since Laplacian is cell-centered
      Real h = pmb->pcoord->dx2f(js);  // pco->dx2f(j); inside loop
      Real C = (h*h)/24.0;

      // construct Laplacian from x2flux
      pmb->pcoord->LaplacianX2All(x2flux, laplacian_all_fc, 0, NDUSTVARS-1,
          kl, ku, js, je+1, il, iu);

      // Approximate x2 face-centered states
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je+1; ++j) {
          // Compute Laplacian of x2 face states
          for (int n=0; n<NDUSTVARS; ++n) {
            pmb->pcoord->LaplacianX2(df_w_l3d_, laplacian_l_df_fc_, n, k, j, il, iu);
            pmb->pcoord->LaplacianX2(df_w_r3d_, laplacian_r_df_fc_, n, k, j, il, iu);
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              df_w_l_(n,i) = df_w_l3d_(n,k,j,i) - C*laplacian_l_df_fc_(i);
              df_w_r_(n,i) = df_w_r3d_(n,k,j,i) - C*laplacian_r_df_fc_(i);
              pmb->peos->ApplyDustFluidsFloors(df_w_l_, n, k, j, i);
              pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
            }
          }

          // Compute x2 interface fluxes from face-centered primitive variables
          if (solver_id == 0)
            RiemannSolverDustFluids_Penetration(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);
          else if (solver_id == 1)
            RiemannSolverDustFluids_noPenetration(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);
          else if (solver_id == 2)
            HLLENoCsRiemannSolverDustFluids(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);
          else
            HLLERiemannSolverDustFluids(k, j, il, iu, 2, df_w_l_, df_w_r_, x2flux);

          // Apply Laplacian of second-order accurate face-averaged flux on x1 faces
          for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
            for (int i=il; i<=iu; i++)
              x2flux(n,k,j,i) = flux_fc(n,k,j,i) + C*laplacian_all_fc(n,k,j,i);
          }
        }
      }
    } // end if (order == 4)
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->f3) {
    AthenaArray<Real> &x3flux = df_flux[X3DIR];

    // set the loop limits
    //if (MAGNETIC_FIELDS_ENABLED)
      il = is-1, iu = ie+1, jl = js-1, ju = je+1;

    for (int j=jl; j<=ju; ++j) { // this loop ordering is intentional
      // reconstruct the first row
      if (order == 1) {
        pmb->precon->DonorCellX3_DustFluids(ks-1, j, il, iu, prim_df, df_w_l_, df_w_r_);
      } else if (order == 2) {
        pmb->precon->PiecewiseLinearX3_DustFluids(ks-1, j, il, iu, prim_df, df_w_l_, df_w_r_);
      } else {
        pmb->precon->PiecewiseParabolicX3_DustFluids(ks-1, j, il, iu, prim_df, df_w_l_, df_w_r_);
        for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->peos->ApplyDustFluidsFloors(df_w_l_, n, ks-1, j, i);
            //pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
          }
        }
      }
      for (int k=ks; k<=ke+1; ++k) {
        // reconstruct L/R states at k
        if (order == 1) {
          pmb->precon->DonorCellX3_DustFluids(k, j, il, iu, prim_df, df_w_lb_, df_w_r_);
        } else if (order == 2) {
          pmb->precon->PiecewiseLinearX3_DustFluids(k, j, il, iu, prim_df, df_w_lb_, df_w_r_);
        } else {
          pmb->precon->PiecewiseParabolicX3_DustFluids(k, j, il, iu, prim_df, df_w_lb_, df_w_r_);
          for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              pmb->peos->ApplyDustFluidsFloors(df_w_lb_, n, k, j, i);
              pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
            }
          }
        }

        if (solver_id == 0)
          RiemannSolverDustFluids_Penetration(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);
        else if (solver_id == 1)
          RiemannSolverDustFluids_noPenetration(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);
        else if (solver_id == 2)
          HLLENoCsRiemannSolverDustFluids(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);
        else
          HLLERiemannSolverDustFluids(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);

        if (order == 4) {
          for (int n=0; n<NDUSTVARS; n++) {
            for (int i=il; i<=iu; i++) {
              df_w_l3d_(n, k, j, i) = df_w_l_(n, i);
              df_w_r3d_(n, k, j, i) = df_w_r_(n, i);
            }
          }
        }

        // swap the arrays for the next step
        df_w_l_.SwapAthenaArray(df_w_lb_);
      }
    }
    if (order == 4) {
      // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so factor this out
      // TODO(felker): also, this may need to be dx3v, since Laplacian is cell-centered
      Real h = pmb->pcoord->dx3f(ks);  // pco->dx3f(j); inside loop
      Real C = (h*h)/24.0;

      // construct Laplacian from x3flux
      pmb->pcoord->LaplacianX3All(x3flux, laplacian_all_fc, 0, NDUSTVARS-1,
                                  ks, ke+1, jl, ju, il, iu);

      // Approximate x3 face-centered states
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
          // Compute Laplacian of x3 face states
          for (int n=0; n<NDUSTVARS; ++n) {
            pmb->pcoord->LaplacianX3(df_w_l3d_, laplacian_l_df_fc_, n, k, j, il, iu);
            pmb->pcoord->LaplacianX3(df_w_r3d_, laplacian_r_df_fc_, n, k, j, il, iu);
#pragma omp simd
            for (int i=il; i<=iu; ++i) {
              df_w_l_(n, i) = df_w_l3d_(n, k, j, i) - C*laplacian_l_df_fc_(i);
              df_w_r_(n, i) = df_w_r3d_(n, k, j, i) - C*laplacian_r_df_fc_(i);
              pmb->peos->ApplyDustFluidsFloors(df_w_l_, n, k, j, i);
              pmb->peos->ApplyDustFluidsFloors(df_w_r_, n, k, j, i);
            }
          }

          // Compute x3 interface fluxes from face-centered primitive variables
          if (solver_id == 0)
            RiemannSolverDustFluids_Penetration(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);
          else if (solver_id == 1)
            RiemannSolverDustFluids_noPenetration(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);
          else if (solver_id == 2)
            HLLENoCsRiemannSolverDustFluids(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);
          else
            HLLERiemannSolverDustFluids(k, j, il, iu, 3, df_w_l_, df_w_r_, x3flux);

          // Apply Laplacian of second-order accurate face-averaged flux on x3 faces
          for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
            for (int i=il; i<=iu; i++)
              x3flux(n,k,j,i) = flux_fc(n,k,j,i) + C*laplacian_all_fc(n,k,j,i);
          }
        }
      }
    } // end if (order == 4)
  }

  if (!STS_ENABLED) {
    AddDiffusionFluxes();
  }
  return;
}


void DustFluids::CalculateDustFluidsFluxes_STS() {
  AddDiffusionFluxes();
}


void DustFluids::AddDiffusionFluxes() {
  // add diffusion fluxes
  if (dfdif.dustfluids_diffusion_defined)
    dfdif.AddDustFluidsDiffusionFlux(dfdif.dustfluids_diffusion_flux, df_flux);
  return;
}
