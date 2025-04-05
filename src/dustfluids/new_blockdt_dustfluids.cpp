//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt_dustfluids.cpp
//! \brief computes timestep using CFL condition on a MEshBlock

// C headers

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // fabs(), sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "dustfluids.hpp"
#include "dustfluids_diffusion/dustfluids_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
// \!fn void DusstFluids::NewBlockTimeStep_Hyperbolic()
// \brief calculate the minimum timestep within a MeshBlock

Real DustFluids::NewAdvectionDt() {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
  // hyperbolic timestep constraint in each (x1-slice) cell along coordinate direction:
  AthenaArray<Real> &dt1 = dt1_, &dt2 = dt2_, &dt3 = dt3_;  // (x1 slices)
  Real df_w_i[NDUSTVARS];

  Real real_max = std::numeric_limits<Real>::max();
  Real min_dt_hyperbolic_df = real_max;

  bool polar_inner = (pmb->pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar"));
  bool polar_outer = (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar"));

  bool polar_wedge_inner = (pmb->pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar_wedge"));
  bool polar_wedge_outer = (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar_wedge"));

  FluidFormulation fluid_status = pmb->pmy_mesh->fluid_setup;
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        pmb->pcoord->CenterWidth1(k, j, is, ie, dt1);
        pmb->pcoord->CenterWidth2(k, j, is, ie, dt2);
        pmb->pcoord->CenterWidth3(k, j, is, ie, dt3);

        // polar averaging
        if ((polar_inner || polar_wedge_inner)) {
          if (j==js)
#pragma omp simd
            for (int i=is; i<=ie; ++i)
              dt3(i) *= 4.0;
          if (j==js+1)
#pragma omp simd
            for (int i=is; i<=ie; ++i)
              dt3(i) *= 2.0;
        }
        if ((polar_outer || polar_wedge_outer)) {
          if (j==je)
#pragma omp simd
            for (int i=is; i<=ie; ++i)
              dt3(i) *= 4.0;
          if (j==je-1)
#pragma omp simd
            for (int i=is; i<=ie; ++i)
              dt3(i) *= 2.0;
        }

#pragma ivdep
        for (int i=is; i<=ie; ++i) {
          df_w_i[rho_id] = df_w(rho_id, k, j, i);
          df_w_i[v1_id]  = df_w(v1_id,  k, j, i);
          df_w_i[v2_id]  = df_w(v2_id,  k, j, i);
          df_w_i[v3_id]  = df_w(v3_id,  k, j, i);

          if ((fluid_status == FluidFormulation::evolve) && SoundSpeed_Flag) {
            dt1(i) /= (std::abs(df_w_i[v1_id]) + cs_dustfluids_array(dust_id, k, j, i));
            dt2(i) /= (std::abs(df_w_i[v2_id]) + cs_dustfluids_array(dust_id, k, j, i));
            dt3(i) /= (std::abs(df_w_i[v3_id]) + cs_dustfluids_array(dust_id, k, j, i));
          } else { // FluidFormulation::background or disabled. Assume scalar advection:
            dt1(i) /= (std::abs(df_w_i[v1_id]));
            dt2(i) /= (std::abs(df_w_i[v2_id]));
            dt3(i) /= (std::abs(df_w_i[v3_id]));
          }
        }

        // compute minimum of (v1 +/- C)
        for (int i=is; i<=ie; ++i) {
          Real& dt_1 = dt1(i);
          min_dt_hyperbolic_df = std::min(min_dt_hyperbolic_df, dt_1);
        }

        // if grid is 2D/3D, compute minimum of (v2 +/- C)
        if (pmb->block_size.nx2 > 1) {
          for (int i=is; i<=ie; ++i) {
            Real& dt_2 = dt2(i);
            min_dt_hyperbolic_df = std::min(min_dt_hyperbolic_df, dt_2);
          }
        }

        // if grid is 3D, compute minimum of (v3 +/- C)
        if (pmb->block_size.nx3 > 1) {
          for (int i=is; i<=ie; ++i) {
            Real& dt_3 = dt3(i);
            min_dt_hyperbolic_df = std::min(min_dt_hyperbolic_df, dt_3);
          }
        }
      }
    }
  }

  min_dt_hyperbolic_df *= pmb->pmy_mesh->cfl_number;

  return min_dt_hyperbolic_df;
}
