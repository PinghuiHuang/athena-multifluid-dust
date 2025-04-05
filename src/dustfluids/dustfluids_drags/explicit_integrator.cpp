//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file explicit_integrator.cpp
//! Explicit drag time integrators

// C++ headers
#include <algorithm>   // min, max
#include <cstring>    // strcmp
#include <iostream>   // endl
#include <limits>
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../defs.hpp"
#include "../../mesh/mesh.hpp"
#include "../dustfluids.hpp"
#include "dust_gas_drag.hpp"


void DustGasDrag::ExplicitFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  bool Drag_Work = (NON_BAROTROPIC_EOS && (!Dissipation_Flag));
  bool Drag_WorkDissipation = (NON_BAROTROPIC_EOS && Dissipation_Flag);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      force_x1.ZeroClear();
      force_x2.ZeroClear();
      force_x3.ZeroClear();
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        // Alias the primitives of gas at current stage
        const Real &gas_rho  = w(IDN, k, j, i);
        const Real &gas_vel1 = w(IM1, k, j, i);
        const Real &gas_vel2 = w(IM2, k, j, i);
        const Real &gas_vel3 = w(IM3, k, j, i);

        mom1_prim(0, i) = gas_rho*gas_vel1;
        mom2_prim(0, i) = gas_rho*gas_vel2;
        mom3_prim(0, i) = gas_rho*gas_vel3;
        inv_gas_rho(i)  = 1.0/gas_rho;
      }

      // Set the drag force
      for (int n=1; n<=NDUSTFLUIDS; ++n) {
        int dust_id = n - 1;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of dust at current stage
          const Real &dust_rho  = prim_df(rho_id, k, j, i);
          const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
          const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
          const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

          alpha(n, i)   = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
          epsilon(n, i) = dust_rho*inv_gas_rho(i);

          mom1_prim(n, i) = dust_rho*dust_vel1;
          mom2_prim(n, i) = dust_rho*dust_vel2;
          mom3_prim(n, i) = dust_rho*dust_vel3;

          force_x1(n, i) = alpha(n, i)*(epsilon(n, i)*mom1_prim(0, i) - mom1_prim(n, i));
          force_x2(n, i) = alpha(n, i)*(epsilon(n, i)*mom2_prim(0, i) - mom2_prim(n, i));
          force_x3(n, i) = alpha(n, i)*(epsilon(n, i)*mom3_prim(0, i) - mom3_prim(n, i));

          force_x1(0, i) -= force_x1(n, i);
          force_x2(0, i) -= force_x2(n, i);
          force_x3(0, i) -= force_x3(n, i);

          delta_mom1(n, i) = force_x1(n, i)*dt;
          delta_mom2(n, i) = force_x2(n, i)*dt;
          delta_mom3(n, i) = force_x3(n, i)*dt;

          // Alias the conserves of dust at current stage
          Real &dust_dens = cons_df(rho_id, k, j, i);
          Real &dust_mom1 = cons_df(v1_id,  k, j, i);
          Real &dust_mom2 = cons_df(v2_id,  k, j, i);
          Real &dust_mom3 = cons_df(v3_id,  k, j, i);

          // Calculate the dust velocity before drags
          Real inv_dust_dens     = 0.0;
          Real dust_vel1_bf_drag = 0.0;
          Real dust_vel2_bf_drag = 0.0;
          Real dust_vel3_bf_drag = 0.0;

          if (Drag_WorkDissipation) {
            inv_dust_dens     = 1.0/dust_dens;
            dust_vel1_bf_drag = dust_mom1*inv_dust_dens;
            dust_vel2_bf_drag = dust_mom2*inv_dust_dens;
            dust_vel3_bf_drag = dust_mom3*inv_dust_dens;
          }

          // Add the delta momentum caused by drags on dust conserves
          dust_mom1 += delta_mom1(n, i);
          dust_mom2 += delta_mom2(n, i);
          dust_mom3 += delta_mom3(n, i);

          // Add the energy dissipation of drags if gas is non barotropic.
          // dE_gas = dM_dust*(v_dust_before + v_dust_after)/2
          if (Drag_WorkDissipation) {
            // Calculate the dust velocity after drags
            Real dust_vel1_af_drag = dust_mom1*inv_dust_dens;
            Real dust_vel2_af_drag = dust_mom2*inv_dust_dens;
            Real dust_vel3_af_drag = dust_mom3*inv_dust_dens;

            Real dissipation = 0.5*(delta_mom1(n, i)*(dust_vel1_bf_drag + dust_vel1_af_drag) +
                                    delta_mom2(n, i)*(dust_vel2_bf_drag + dust_vel2_af_drag) +
                                    delta_mom3(n, i)*(dust_vel3_bf_drag + dust_vel3_af_drag));

            Real &gas_erg  = u(IEN, k, j, i);
            gas_erg       -= dissipation;
          }
        }
      }

#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        delta_mom1(0, i) = force_x1(0, i)*dt;
        delta_mom2(0, i) = force_x2(0, i)*dt;
        delta_mom3(0, i) = force_x3(0, i)*dt;

        // Alias the conserves of gas at current stage
        Real &gas_dens = u(IDN, k, j, i);
        Real &gas_mom1 = u(IM1, k, j, i);
        Real &gas_mom2 = u(IM2, k, j, i);
        Real &gas_mom3 = u(IM3, k, j, i);

        // Calculate the gas velocity before drags
        Real inv_gas_dens     = 0.0;
        Real gas_vel1_bf_drag = 0.0;
        Real gas_vel2_bf_drag = 0.0;
        Real gas_vel3_bf_drag = 0.0;

        if (Drag_Work) {
          inv_gas_dens     = 1.0/gas_dens;
          gas_vel1_bf_drag = gas_mom1*inv_gas_dens;
          gas_vel2_bf_drag = gas_mom2*inv_gas_dens;
          gas_vel3_bf_drag = gas_mom3*inv_gas_dens;
        }

        // Add the delta momentum caused by drags on gas conserves
        gas_mom1 += delta_mom1(0, i);
        gas_mom2 += delta_mom2(0, i);
        gas_mom3 += delta_mom3(0, i);

        // Add the work done by drags if gas is non barotropic.
        // dE_gas = dM_gas*(v_gas_before + v_gas_after)/2
        if (Drag_Work) {
          // Calculate the gas velocity after drags
          Real gas_vel1_af_drag = gas_mom1*inv_gas_dens;
          Real gas_vel2_af_drag = gas_mom2*inv_gas_dens;
          Real gas_vel3_af_drag = gas_mom3*inv_gas_dens;

          Real work_drag = 0.5*(delta_mom1(0, i)*(gas_vel1_bf_drag + gas_vel1_af_drag) +
                                delta_mom2(0, i)*(gas_vel2_bf_drag + gas_vel2_af_drag) +
                                delta_mom3(0, i)*(gas_vel3_bf_drag + gas_vel3_af_drag));

          Real &gas_erg  = u(IEN, k, j, i);
          gas_erg       += work_drag;
        }
      }
    }
  }
  return;
}


void DustGasDrag::ExplicitNoFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      force_x1.ZeroClear();
      force_x2.ZeroClear();
      force_x3.ZeroClear();
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        // Alias the primitives of gas at current stage
        const Real &gas_rho  = w(IDN, k, j, i);
        const Real &gas_vel1 = w(IM1, k, j, i);
        const Real &gas_vel2 = w(IM2, k, j, i);
        const Real &gas_vel3 = w(IM3, k, j, i);

        mom1_prim(0, i) = gas_rho*gas_vel1;
        mom2_prim(0, i) = gas_rho*gas_vel2;
        mom3_prim(0, i) = gas_rho*gas_vel3;
        inv_gas_rho(i)  = 1.0/gas_rho;
      }

      // Set the drag force
      for (int n=1; n<=NDUSTFLUIDS; ++n) {
        int dust_id = n - 1;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of dust at current stage
          const Real &dust_rho  = prim_df(rho_id, k, j, i);
          const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
          const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
          const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

          alpha(n, i)   = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
          epsilon(n, i) = dust_rho*inv_gas_rho(i);

          mom1_prim(n, i) = dust_rho*dust_vel1;
          mom2_prim(n, i) = dust_rho*dust_vel2;
          mom3_prim(n, i) = dust_rho*dust_vel3;

          force_x1(n, i) = alpha(n, i)*(epsilon(n, i)*mom1_prim(0, i) - mom1_prim(n, i));
          force_x2(n, i) = alpha(n, i)*(epsilon(n, i)*mom2_prim(0, i) - mom2_prim(n, i));
          force_x3(n, i) = alpha(n, i)*(epsilon(n, i)*mom3_prim(0, i) - mom3_prim(n, i));

          delta_mom1(n, i) = force_x1(n, i)*dt;
          delta_mom2(n, i) = force_x2(n, i)*dt;
          delta_mom3(n, i) = force_x3(n, i)*dt;

          // Alias the conserves of dust at current stage
          Real &dust_mom1 = cons_df(v1_id, k, j, i);
          Real &dust_mom2 = cons_df(v2_id, k, j, i);
          Real &dust_mom3 = cons_df(v3_id, k, j, i);

          // Add the delta momentum caused by drags on dust conserves
          dust_mom1 += delta_mom1(n, i);
          dust_mom2 += delta_mom2(n, i);
          dust_mom3 += delta_mom3(n, i);
        }
      }
    }
  }
  return;
}


void DustGasDrag::RK2ExplicitFeedback(const int stage,
      const Real dt, const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df) {

  const AthenaArray<Real> &w_n       = pmy_hydro_->w_n;
  const AthenaArray<Real> &prim_df_n = pmy_dustfluids_->df_w_n;

  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  bool Stage_I = (((orb_advection_  < 2) && (stage == 1)) ||
                  ((orb_advection_ == 2) && (stage == 2)));
  bool Drag_Work = (NON_BAROTROPIC_EOS && (!Dissipation_Flag));
  bool Drag_WorkDissipation = (NON_BAROTROPIC_EOS && Dissipation_Flag);

  AthenaArray<Real> &Stage_I_delta_mom1 = pmy_dustfluids_->Stage_I_delta_mom1;
  AthenaArray<Real> &Stage_I_delta_mom2 = pmy_dustfluids_->Stage_I_delta_mom2;
  AthenaArray<Real> &Stage_I_delta_mom3 = pmy_dustfluids_->Stage_I_delta_mom3;

  AthenaArray<Real> &Stage_I_vel1 = pmy_dustfluids_->Stage_I_vel1;
  AthenaArray<Real> &Stage_I_vel2 = pmy_dustfluids_->Stage_I_vel2;
  AthenaArray<Real> &Stage_I_vel3 = pmy_dustfluids_->Stage_I_vel3;

  if (Stage_I) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of gas at current stage
          const Real &gas_rho  = w(IDN, k, j, i);
          const Real &gas_vel1 = w(IM1, k, j, i);
          const Real &gas_vel2 = w(IM2, k, j, i);
          const Real &gas_vel3 = w(IM3, k, j, i);

          mom1_prim(0, i) = gas_rho*gas_vel1;
          mom2_prim(0, i) = gas_rho*gas_vel2;
          mom3_prim(0, i) = gas_rho*gas_vel3;
          inv_gas_rho(i)  = 1.0/gas_rho;
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            // Alias the primitives of dust at current stage
            const Real &dust_rho  = prim_df(rho_id, k, j, i);
            const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            alpha(n, i)   = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
            epsilon(n, i) = dust_rho*inv_gas_rho(i);

            mom1_prim(n, i) = dust_rho*dust_vel1;
            mom2_prim(n, i) = dust_rho*dust_vel2;
            mom3_prim(n, i) = dust_rho*dust_vel3;

            force_x1(n, i) = alpha(n, i)*(epsilon(n, i)*mom1_prim(0, i) - mom1_prim(n, i));
            force_x2(n, i) = alpha(n, i)*(epsilon(n, i)*mom2_prim(0, i) - mom2_prim(n, i));
            force_x3(n, i) = alpha(n, i)*(epsilon(n, i)*mom3_prim(0, i) - mom3_prim(n, i));

            force_x1(0, i) -= force_x1(n, i);
            force_x2(0, i) -= force_x2(n, i);
            force_x3(0, i) -= force_x3(n, i);

            delta_mom1(n, i) = force_x1(n, i)*dt;
            delta_mom2(n, i) = force_x2(n, i)*dt;
            delta_mom3(n, i) = force_x3(n, i)*dt;

            // Alias the conserves of dust at current stage
            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            // Calculate the dust velocity before drags
            Real inv_dust_dens     = 0.0;
            Real dust_vel1_bf_drag = 0.0;
            Real dust_vel2_bf_drag = 0.0;
            Real dust_vel3_bf_drag = 0.0;

            if (Drag_WorkDissipation) {
              inv_dust_dens     = 1.0/dust_dens;
              dust_vel1_bf_drag = dust_mom1*inv_dust_dens;
              dust_vel2_bf_drag = dust_mom2*inv_dust_dens;
              dust_vel3_bf_drag = dust_mom3*inv_dust_dens;

              Stage_I_vel1(n, k, j, i) = dust_vel1_bf_drag;
              Stage_I_vel2(n, k, j, i) = dust_vel2_bf_drag;
              Stage_I_vel3(n, k, j, i) = dust_vel3_bf_drag;
            }

            Stage_I_delta_mom1(n, k, j, i) = 0.5*delta_mom1(n, i);
            Stage_I_delta_mom2(n, k, j, i) = 0.5*delta_mom2(n, i);
            Stage_I_delta_mom3(n, k, j, i) = 0.5*delta_mom3(n, i);

            // Add the delta momentum caused by drags on dust conserves
            dust_mom1 += delta_mom1(n, i);
            dust_mom2 += delta_mom2(n, i);
            dust_mom3 += delta_mom3(n, i);

            // Add the energy dissipation of drags if gas is non barotropic.
            // dE_gas = dM_dust*(v_dust_before + v_dust_after)/2
            if (Drag_WorkDissipation) {
              // Calculate the dust velocity after drags
              Real dust_vel1_af_drag = dust_mom1*inv_dust_dens;
              Real dust_vel2_af_drag = dust_mom2*inv_dust_dens;
              Real dust_vel3_af_drag = dust_mom3*inv_dust_dens;

              Real dissipation = 0.5*(delta_mom1(n, i)*(dust_vel1_bf_drag + dust_vel1_af_drag) +
                                      delta_mom2(n, i)*(dust_vel2_bf_drag + dust_vel2_af_drag) +
                                      delta_mom3(n, i)*(dust_vel3_bf_drag + dust_vel3_af_drag));

              Real &gas_erg  = u(IEN, k, j, i);
              gas_erg       -= dissipation;
            }
          }
        }

#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          delta_mom1(0, i) = force_x1(0, i)*dt;
          delta_mom2(0, i) = force_x2(0, i)*dt;
          delta_mom3(0, i) = force_x3(0, i)*dt;

          // Alias the conserves of gas at current stage
          Real &gas_dens = u(IDN, k, j, i);
          Real &gas_mom1 = u(IM1, k, j, i);
          Real &gas_mom2 = u(IM2, k, j, i);
          Real &gas_mom3 = u(IM3, k, j, i);

          // Calculate the gas velocity before drags
          Real inv_gas_dens     = 0.0;
          Real gas_vel1_bf_drag = 0.0;
          Real gas_vel2_bf_drag = 0.0;
          Real gas_vel3_bf_drag = 0.0;

          if (Drag_Work) {
            inv_gas_dens     = 1.0/gas_dens;
            gas_vel1_bf_drag = gas_mom1*inv_gas_dens;
            gas_vel2_bf_drag = gas_mom2*inv_gas_dens;
            gas_vel3_bf_drag = gas_mom3*inv_gas_dens;

            Stage_I_vel1(0, k, j, i) = gas_vel1_bf_drag;
            Stage_I_vel2(0, k, j, i) = gas_vel2_bf_drag;
            Stage_I_vel3(0, k, j, i) = gas_vel3_bf_drag;

            Stage_I_delta_mom1(0, k, j, i) = 0.5*delta_mom1(0, i);
            Stage_I_delta_mom2(0, k, j, i) = 0.5*delta_mom2(0, i);
            Stage_I_delta_mom3(0, k, j, i) = 0.5*delta_mom3(0, i);
          }

          // Add the delta momentum caused by drags on gas conserves
          gas_mom1 += delta_mom1(0, i);
          gas_mom2 += delta_mom2(0, i);
          gas_mom3 += delta_mom3(0, i);

          // Add the work done by drags if gas is non barotropic.
          // dE_gas = dM_gas*(v_gas_before + v_gas_after)/2
          if (Drag_Work) {
            // Calculate the gas velocity after drags
            Real gas_vel1_af_drag = gas_mom1*inv_gas_dens;
            Real gas_vel2_af_drag = gas_mom2*inv_gas_dens;
            Real gas_vel3_af_drag = gas_mom3*inv_gas_dens;

            Real work_drag = 0.5*(delta_mom1(0, i)*(gas_vel1_bf_drag + gas_vel1_af_drag) +
                                  delta_mom2(0, i)*(gas_vel2_bf_drag + gas_vel2_af_drag) +
                                  delta_mom3(0, i)*(gas_vel3_bf_drag + gas_vel3_af_drag));

            Real &gas_erg  = u(IEN, k, j, i);
            gas_erg       += work_drag;
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        force_x1.ZeroClear();
        force_x2.ZeroClear();
        force_x3.ZeroClear();
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of gas at current stage
          const Real &gas_rho  = w(IDN, k, j, i);
          const Real &gas_vel1 = w(IM1, k, j, i);
          const Real &gas_vel2 = w(IM2, k, j, i);
          const Real &gas_vel3 = w(IM3, k, j, i);

          mom1_prim(0, i) = gas_rho*gas_vel1;
          mom2_prim(0, i) = gas_rho*gas_vel2;
          mom3_prim(0, i) = gas_rho*gas_vel3;
          inv_gas_rho(i)  = 1.0/gas_rho;
        }

        for (int n=1; n<=NDUSTFLUIDS; ++n) {
          int dust_id = n - 1;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            // Alias the primitives of dust at current stage
            const Real &dust_rho  = prim_df(rho_id, k, j, i);
            const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            alpha(n, i)   = 1.0/(stopping_time(dust_id, k, j, i) + TINY_NUMBER);
            epsilon(n, i) = dust_rho*inv_gas_rho(i);

            mom1_prim(n, i) = dust_rho*dust_vel1;
            mom2_prim(n, i) = dust_rho*dust_vel2;
            mom3_prim(n, i) = dust_rho*dust_vel3;

            force_x1(n, i) = alpha(n, i)*(epsilon(n, i)*mom1_prim(0, i) - mom1_prim(n, i));
            force_x2(n, i) = alpha(n, i)*(epsilon(n, i)*mom2_prim(0, i) - mom2_prim(n, i));
            force_x3(n, i) = alpha(n, i)*(epsilon(n, i)*mom3_prim(0, i) - mom3_prim(n, i));

            force_x1(0, i) -= force_x1(n, i);
            force_x2(0, i) -= force_x2(n, i);
            force_x3(0, i) -= force_x3(n, i);

            delta_mom1(n, i) = force_x1(n, i)*dt;
            delta_mom2(n, i) = force_x2(n, i)*dt;
            delta_mom3(n, i) = force_x3(n, i)*dt;

            // Alias the conserves of dust at current stage
            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            // Add the delta momentum caused by drags on dust conserves
            dust_mom1 += delta_mom1(n, i);
            dust_mom2 += delta_mom2(n, i);
            dust_mom3 += delta_mom3(n, i);

            // Add the energy dissipation of drags if gas is non barotropic.
            // dE_gas = dM_dust*(v_dust_before + v_dust_after)/2
            if (Drag_WorkDissipation) {
              // Calculate the dust velocity after drags
              Real inv_dust_dens     = 1.0/dust_dens;
              Real dust_vel1_af_drag = dust_mom1*inv_dust_dens;
              Real dust_vel2_af_drag = dust_mom2*inv_dust_dens;
              Real dust_vel3_af_drag = dust_mom3*inv_dust_dens;

              Real dissipation_I = -0.5*(Stage_I_delta_mom1(n, k, j, i)*(Stage_I_vel1(n, k, j, i) + dust_vel1) +
                                         Stage_I_delta_mom2(n, k, j, i)*(Stage_I_vel2(n, k, j, i) + dust_vel2) +
                                         Stage_I_delta_mom3(n, k, j, i)*(Stage_I_vel3(n, k, j, i) + dust_vel3));

              Real dissipation = 0.5*((Stage_I_delta_mom1(n, k, j, i) + delta_mom1(n, i))*(Stage_I_vel1(n, k, j, i) + dust_vel1_af_drag) +
                                      (Stage_I_delta_mom2(n, k, j, i) + delta_mom2(n, i))*(Stage_I_vel2(n, k, j, i) + dust_vel2_af_drag) +
                                      (Stage_I_delta_mom3(n, k, j, i) + delta_mom3(n, i))*(Stage_I_vel3(n, k, j, i) + dust_vel3_af_drag));

              Real &gas_erg  = u(IEN, k, j, i);
              gas_erg       -= (dissipation_I + dissipation);
            }
          }
        }

#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // Alias the primitives of gas at current stage
          const Real &gas_vel1 = w(IM1, k, j, i);
          const Real &gas_vel2 = w(IM2, k, j, i);
          const Real &gas_vel3 = w(IM3, k, j, i);

          delta_mom1(0, i) = force_x1(0, i)*dt;
          delta_mom2(0, i) = force_x2(0, i)*dt;
          delta_mom3(0, i) = force_x3(0, i)*dt;

          // Alias the conserves of gas at current stage
          Real &gas_dens = u(IDN, k, j, i);
          Real &gas_mom1 = u(IM1, k, j, i);
          Real &gas_mom2 = u(IM2, k, j, i);
          Real &gas_mom3 = u(IM3, k, j, i);

          // Add the delta momentum caused by drags on gas conserves
          gas_mom1 += delta_mom1(0, i);
          gas_mom2 += delta_mom2(0, i);
          gas_mom3 += delta_mom3(0, i);

          // Add the work done by drags if gas is non barotropic.
          // dE_gas = dM_gas*(v_gas_before + v_gas_after)/2
          if (Drag_Work) {
            // Calculate the gas velocity after drags
            Real inv_gas_dens     = 1.0/gas_dens;
            Real gas_vel1_af_drag = gas_mom1*inv_gas_dens;
            Real gas_vel2_af_drag = gas_mom2*inv_gas_dens;
            Real gas_vel3_af_drag = gas_mom3*inv_gas_dens;

            Real work_drag_I = -0.5*(Stage_I_delta_mom1(0, k, j, i)*(Stage_I_vel1(0, k, j, i) + gas_vel1) +
                                     Stage_I_delta_mom2(0, k, j, i)*(Stage_I_vel2(0, k, j, i) + gas_vel2) +
                                     Stage_I_delta_mom3(0, k, j, i)*(Stage_I_vel3(0, k, j, i) + gas_vel3));

            Real work_drag = 0.5*((Stage_I_delta_mom1(0, k, j, i) + delta_mom1(0, i))*(Stage_I_vel1(0, k, j, i) + gas_vel1_af_drag) +
                                  (Stage_I_delta_mom2(0, k, j, i) + delta_mom2(0, i))*(Stage_I_vel2(0, k, j, i) + gas_vel2_af_drag) +
                                  (Stage_I_delta_mom3(0, k, j, i) + delta_mom3(0, i))*(Stage_I_vel3(0, k, j, i) + gas_vel3_af_drag));

            Real &gas_erg  = u(IEN, k, j, i);
            gas_erg       += (work_drag_I + work_drag);
          }
        }
      }
    }
  }
  return;
}
