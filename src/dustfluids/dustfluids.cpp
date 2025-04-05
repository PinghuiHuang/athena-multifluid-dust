//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids.cpp
//! \brief implementation of functions in class DustFluids

// C headers

// C++ headers
#include <algorithm>
#include <cstring>    // strcmp
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "dustfluids.hpp"
#include "dustfluids_diffusion/dustfluids_diffusion.hpp"
#include "dustfluids_diffusion_cc/cell_center_diffusions.hpp"
#include "dustfluids_drags/dust_gas_drag.hpp"
#include "srcterms/dustfluids_srcterms.hpp"

class DustGasDrag;
class DustFluidsSourceTerms;
class DustFluidsDiffusion;
class DustFluidsCellCenterDiffusion;

//! constructor, initializes data structures and parameters
DustFluids::DustFluids(MeshBlock *pmb, ParameterInput *pin)  :
  pmy_block(pmb),
  df_u(NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_u1(NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_u_af_src(NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_w(NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_w1(NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_w_n(NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_flux{{NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {NDUSTVARS, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
          (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
          AthenaArray<Real>::DataStatus::empty)},
          {NDUSTVARS, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
          (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
          AthenaArray<Real>::DataStatus::empty)}},
  coarse_df_u_(NDUSTVARS, pmb->ncc3, pmb->ncc2, pmb->ncc1,
          (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
           AthenaArray<Real>::DataStatus::empty)),
  coarse_df_w_(NDUSTVARS, pmb->ncc3, pmb->ncc2, pmb->ncc1,
          (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
           AthenaArray<Real>::DataStatus::empty)),
  stopping_time_array(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  nu_dustfluids_array(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  cs_dustfluids_array(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  stopping_time_array_n(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  nu_dustfluids_array_n(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  cs_dustfluids_array_n(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  Stage_I_delta_mom1(NDUSTGAS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  Stage_I_delta_mom2(NDUSTGAS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  Stage_I_delta_mom3(NDUSTGAS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  Stage_I_vel1(NDUSTGAS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  Stage_I_vel2(NDUSTGAS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  Stage_I_vel3(NDUSTGAS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  dfbvar(pmb, &df_u, &coarse_df_u_, df_flux, DustFluidsBoundaryQuantity::cons_df),
  dfdrag(this, pin), dfdif(this, pin), dfccdif(pmy_block, this, pin), dfsrc(this, pin),
  pco_(pmb->pcoord) {

  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;

  Mesh *pm = pmy_block->pmy_mesh;
  pmb->RegisterMeshBlockData(df_u);

  int xorder  = pmb->precon->xorder;
  dust_xorder = pin->GetOrAddInteger("dust", "dust_xorder", xorder);
  if (dust_xorder > xorder)
    dust_xorder = xorder;

  solver_id = pin->GetOrAddInteger("dust", "solver_id", 0);
  if ((solver_id < 0) || (solver_id > 3)) {
    std::stringstream msg;
    msg << "The solver_id must be 0, 1, 2 or 3!" << std::endl;
    ATHENA_ERROR(msg);
  }

  SoundSpeed_Flag = pin->GetOrAddBoolean("dust", "Dust_SoundSpeed_Flag", false);

  // If dust is inviscid, then sound speed flag is set as false
  if (!dfdif.Diffusion_Flag) SoundSpeed_Flag = false;
  if (SoundSpeed_Flag) solver_id = 3;

  UserDefinedStoppingTime = pmy_block->pmy_mesh->UserStoppingTime_;
  if (UserDefinedStoppingTime == nullptr) {
    for (int n=0; n<NDUSTFLUIDS; ++n)
      const_stopping_time[n] = pin->GetReal("dust", "stopping_time_" + std::to_string(n+1));
  }

  if (dfdif.dustfluids_diffusion_defined) {
    UserDefinedDustDiffusivity = pmy_block->pmy_mesh->DustDiffusivity_;
    if (UserDefinedDustDiffusivity == nullptr) {
      for (int n=0; n<NDUSTFLUIDS; ++n)
        const_nu_dust[n] = pin->GetReal("dust", "nu_dust_" + std::to_string(n+1));
    }
  }

  // Allocate optional dustfluids variable memory registers for time-integrator
  if (dust_xorder == 4) {
    // fourth-order cell-centered approximations
    df_u_cc.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
    df_w_cc.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
  }

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");

  if (integrator == "ssprk5_4" || STS_ENABLED) {
    // future extension may add "int nregister" to Hydro class
    df_u2.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
  }

  // If STS RKL2, allocate additional memory registers
  if (STS_ENABLED) {
    std::string sts_integrator = pin->GetOrAddString("time", "sts_integrator", "rkl2");
    if (sts_integrator == "rkl2") {
      df_u0.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
      df_u_fl_div.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
    }
  }

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&df_u, &coarse_df_u_);
  }

  // enroll DustFluidsBoundaryVariable object
  dfbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&dfbvar);
  pmb->pbval->bvars_main_int.push_back(&dfbvar);

  if (STS_ENABLED) {
    if (dfdif.dustfluids_diffusion_defined) {
      pmb->pbval->bvars_sts.push_back(&dfbvar);
    }
  }

  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nc1);
  dt2_.NewAthenaArray(nc1);
  dt3_.NewAthenaArray(nc1);
  //dx_df_w_.NewAthenaArray(nc1);
  df_w_l_.NewAthenaArray(NDUSTVARS, nc1);
  df_w_r_.NewAthenaArray(NDUSTVARS, nc1);
  df_w_lb_.NewAthenaArray(NDUSTVARS, nc1);
  x1face_area_.NewAthenaArray(nc1+1);

  if (pm->f2) {
    x2face_area_.NewAthenaArray(nc1);
    x2face_area_p1_.NewAthenaArray(nc1);
  }

  if (pm->f3) {
    x3face_area_.NewAthenaArray(nc1);
    x3face_area_p1_.NewAthenaArray(nc1);
  }

  cell_volume_.NewAthenaArray(nc1);
  dflx_.NewAthenaArray(NDUSTVARS, nc1);

  // fourth-order integration scheme
  if (dust_xorder == 4) {
    // 4D scratch arrays
    df_w_l3d_.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
    df_w_r3d_.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
    scr1_nkji_.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
    scr2_nkji_.NewAthenaArray(NDUSTVARS, nc3, nc2, nc1);
    // store all face-centered mass fluxes (all 3x coordinate directions) from Hydro:

    // 1D scratch arrays
    laplacian_l_df_fc_.NewAthenaArray(nc1);
    laplacian_r_df_fc_.NewAthenaArray(nc1);
  }
}


void DustFluids::ConstantStoppingTime(MeshBlock *pmb, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

  // Calculate the constant stopping time
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &st_time = stopping_time(dust_id, k, j, i);
          st_time       = const_stopping_time[dust_id];
        }
      }
    }
  }
  return;
}


void DustFluids::SetDustFluidsProperties(const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust) {

  int il = pmy_block->is - NGHOST;
  int iu = pmy_block->ie + NGHOST;

  int jl = pmy_block->js;
  int ju = pmy_block->je;

  int kl = pmy_block->ks;
  int ku = pmy_block->ke;

  if (pmy_block->block_size.nx2 > 1) {
    jl -= NGHOST; ju += NGHOST;
  }

  if (pmy_block->block_size.nx3 > 1) {
    kl -= NGHOST; ku += NGHOST;
  }

  // User-defined stopping time
  UserDefinedStoppingTime = pmy_block->pmy_mesh->UserStoppingTime_;
  if (UserDefinedStoppingTime != nullptr)
    UserDefinedStoppingTime(pmy_block, time, prim, prim_df, stopping_time, il, iu, jl, ju, kl, ku);
  else
    ConstantStoppingTime(pmy_block, stopping_time, il, iu, jl, ju, kl, ku);

  UserDefinedDustDiffusivity = pmy_block->pmy_mesh->DustDiffusivity_;
  if (dfdif.dustfluids_diffusion_defined) {
    if (UserDefinedDustDiffusivity != nullptr)
      UserDefinedDustDiffusivity(this, pmy_block, prim, prim_df, stopping_time,
          nu_dust, cs_dust, il, iu, jl, ju, kl, ku);
    else
      dfdif.ConstantDustDiffusivity(this, pmy_block, nu_dust, cs_dust, il, iu, jl, ju, kl, ku);
  }

  return;
}
