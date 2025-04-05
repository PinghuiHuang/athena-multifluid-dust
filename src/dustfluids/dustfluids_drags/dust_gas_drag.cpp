//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dust_gas_drag.cpp
//! Contains data and functions that implement physical (not coordinate) drag terms

// C++ headers
#include <algorithm>   // min,max
#include <limits>
#include <cstring>    // strcmp

// Athena++ headers
#include "../../defs.hpp"
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../hydro/hydro.hpp"
#include "../dustfluids.hpp"
#include "dust_gas_drag.hpp"


DustGasDrag::DustGasDrag(DustFluids *pdf, ParameterInput *pin) :
  pmy_dustfluids_(pdf), pmb_(pmy_dustfluids_->pmy_block),
  pco_(pmb_->pcoord), pmy_hydro_(pmb_->phydro),
  orb_advection_(pmb_->pmy_mesh->orbital_advection) {

  int nc1 = pmb_->ncells1;

  force_x1.NewAthenaArray(NDUSTGAS, nc1);
  force_x2.NewAthenaArray(NDUSTGAS, nc1);
  force_x3.NewAthenaArray(NDUSTGAS, nc1);

  force_x1_n.NewAthenaArray(NDUSTGAS, nc1);
  force_x2_n.NewAthenaArray(NDUSTGAS, nc1);
  force_x3_n.NewAthenaArray(NDUSTGAS, nc1);

  inv_gas_rho.NewAthenaArray(nc1);
  alpha.NewAthenaArray(NDUSTGAS, nc1);
  epsilon.NewAthenaArray(NDUSTGAS, nc1);

  inv_gas_rho_n.NewAthenaArray(nc1);
  alpha_n.NewAthenaArray(NDUSTGAS, nc1);
  epsilon_n.NewAthenaArray(NDUSTGAS, nc1);

  qvalue.NewAthenaArray(NDUSTGAS, nc1);
  weight_gas.NewAthenaArray(NDUSTGAS, nc1);
  weight_dust.NewAthenaArray(NDUSTGAS, nc1);

  delta_mom1.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom2.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom3.NewAthenaArray(NDUSTGAS, nc1);

  delta_mom1_im.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom2_im.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom3_im.NewAthenaArray(NDUSTGAS, nc1);

  delta_mom1_im_II.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom2_im_II.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom3_im_II.NewAthenaArray(NDUSTGAS, nc1);

  delta_mom1_src.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom2_src.NewAthenaArray(NDUSTGAS, nc1);
  delta_mom3_src.NewAthenaArray(NDUSTGAS, nc1);

  mom1_prim.NewAthenaArray(NDUSTGAS, nc1);
  mom2_prim.NewAthenaArray(NDUSTGAS, nc1);
  mom3_prim.NewAthenaArray(NDUSTGAS, nc1);

  mom1_prim_n.NewAthenaArray(NDUSTGAS, nc1);
  mom2_prim_n.NewAthenaArray(NDUSTGAS, nc1);
  mom3_prim_n.NewAthenaArray(NDUSTGAS, nc1);

  jacobi.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);
  jacobi_n.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);
  product.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);

  lambda.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);
  lambda_inv.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);

  biggest_arr.NewAthenaArray(nc1);
  det_arr.NewAthenaArray(nc1);
  mmax_arr.NewAthenaArray(nc1);
  sum_arr.NewAthenaArray(nc1);
  temp_arr.NewAthenaArray(nc1);
  idx_vector.NewAthenaArray(NDUSTGAS, nc1);
  scale_arr.NewAthenaArray(NDUSTGAS, nc1);
  xx_arr.NewAthenaArray(NDUSTGAS, nc1);
  lu_matrix.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);

  temp_rho.NewAthenaArray(nc1);
  temp_inv_rho.NewAthenaArray(nc1);

  temp_mom1.NewAthenaArray(nc1);
  temp_mom2.NewAthenaArray(nc1);
  temp_mom3.NewAthenaArray(nc1);

  temp_total_vel1.NewAthenaArray(nc1);
  temp_total_vel2.NewAthenaArray(nc1);
  temp_total_vel3.NewAthenaArray(nc1);

  temp_A.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);
  temp_B.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);
  temp_C.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);
  temp_D.NewAthenaArray(NDUSTGAS, NDUSTGAS, nc1);

  time_drag = pin->GetOrAddReal("dust", "time_drag", 0.0);
  if (time_drag < 0.0) time_drag = 0.0;

  DustFeedback_Flag = pin->GetBoolean("dust", "DustFeedback_Flag");
  Dissipation_Flag  = pin->GetOrAddBoolean("dust", "Dissipation_Flag", true);
  integrator_       = pin->GetOrAddString("time", "integrator", "vl2");
  drag_method       = pin->GetOrAddString("dust", "drag_method", "2nd-implicit");

  if      (drag_method == "2nd-implicit")  method_id_ = 1;
  else if (drag_method == "1st-implicit")  method_id_ = 2;
  else if (drag_method == "semi-implicit") method_id_ = 3;
  else if (drag_method == "explicit")      method_id_ = 4;
  else                                     method_id_ = 0;
}


void DustGasDrag::DragIntegrate(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  switch (method_id_) {
    case 1:
      if (integrator_ == "vl2") {
        if (DustFeedback_Flag)
          VL2ImplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          VL2ImplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else if (integrator_ == "rk2") {
        if (DustFeedback_Flag)
          RK2ImplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          RK2ImplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else {
        std::stringstream msg;
        msg << "The integrator combined with the 2nd-implicit methods must be \"VL2\" or \"RK2\"!" << std::endl;
        ATHENA_ERROR(msg);
      }
      break;

    case 2:
      if (integrator_ == "rk1") {
        if (DustFeedback_Flag)
          BackwardEulerFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          BackwardEulerNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else if (integrator_ == "vl2") {
        if (DustFeedback_Flag)
          VL2BackwardEulerFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          VL2BackwardEulerNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else if (integrator_ == "rk2") {
        if (DustFeedback_Flag)
          RK2BackwardEulerFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          RK2BackwardEulerNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else {
        std::stringstream msg;
        msg << "The integrator combined with the 1st-implicit methods must be \"RK1\" or \"VL2\" or \"RK2\"!" << std::endl;
        ATHENA_ERROR(msg);
      }
      break;

    case 3:
      if (integrator_ == "vl2") {
        if (DustFeedback_Flag)
          TRBDF2Feedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          TRBDF2NoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else if (integrator_ == "rk2") {
        if (DustFeedback_Flag)
          TrapezoidFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          TrapezoidNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else {
        std::stringstream msg;
        msg << "The integrator combined with the semi-implicit (2nd-order) methods must be \"VL2\" or \"RK2\"!" << std::endl;
        ATHENA_ERROR(msg);
      }
      break;

    case 4:
      if (integrator_ == "rk2") {
        if (DustFeedback_Flag)
          RK2ExplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          ExplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else if ( (integrator_ == "rk1") || (integrator_ == "vl2") ) {
        if (DustFeedback_Flag)
          ExplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
        else
          ExplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      } else {
        if (DustFeedback_Flag) {
          if (!NON_BAROTROPIC_EOS)
            ExplicitFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
          else {
            std::stringstream msg;
            msg << "For now, only \"VL2\", \"RK1\" and \"RK2\" can be used in non-isothermal cases when dust feedback is turned on!" << std::endl;
            ATHENA_ERROR(msg);
          }
        } else
            ExplicitNoFeedback(stage, dt, stopping_time, w, prim_df, u, cons_df);
      }
      break;

    default:
      std::stringstream msg;
      msg << "The drag-integrate method must be \"2nd-implicit\" or \"1st-implicit\" or \"semi-implicit\" or \"explicit\"!" << std::endl;
      ATHENA_ERROR(msg);
      break;
  }
  return;
}
