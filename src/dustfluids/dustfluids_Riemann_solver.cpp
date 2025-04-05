//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_noCs_solver.cpp
//! \brief HLLE Riemann solver for dust fludis (no dust sound speed)
//!
//! Computes 1D fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.  This flux is
//! very diffusive, especially for contacts, and so it is not recommended for use in
//! applications.  However, as shown by Einfeldt et al.(1991), it is positively
//! conservative (cannot return negative densities or pressure), so it is a useful
//! option when other approximate solvers fail and/or when extra dissipation is needed.
//!
// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "dustfluids.hpp"

//----------------------------------------------------------------------------------------
//! \fn void DustFluids::RiemannSolver_DustFluids
//! \brief The Riemann solver for Dust Fluids (no dust sound speed)

void DustFluids::RiemannSolverDustFluids_Penetration(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &prim_df_l,
                          AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {

  Real df_w_li[(NDUSTVARS)], df_w_ri[(NDUSTVARS)];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int idust = n;
    int irho  = 4*idust;

    int v1_id = irho + 1;
    int v2_id = irho + 2;
    int v3_id = irho + 3;

    int ivx = irho + (IVX + (index-IVX)%3);
    int ivy = irho + (IVX + ((index-IVX)+1)%3);
    int ivz = irho + (IVX + ((index-IVX)+2)%3);

#pragma omp simd private(df_w_li, df_w_ri)
    for (int i=il; i<=iu; i++) {
      df_w_li[irho]  = prim_df_l(irho, i);
      df_w_li[v1_id] = prim_df_l(ivx,  i);
      df_w_li[v2_id] = prim_df_l(ivy,  i);
      df_w_li[v3_id] = prim_df_l(ivz,  i);

      df_w_ri[irho]  = prim_df_r(irho, i);
      df_w_ri[v1_id] = prim_df_r(ivx,  i);
      df_w_ri[v2_id] = prim_df_r(ivy,  i);
      df_w_ri[v3_id] = prim_df_r(ivz,  i);

      Real temp_li = (df_w_li[v1_id] > 0.0) ? 1.0 : 0.0;
      Real temp_ri = (df_w_ri[v1_id] < 0.0) ? 1.0 : 0.0;

      Real flx_rho_li = df_w_li[irho]*df_w_li[v1_id];
      Real flx_rho_ri = df_w_ri[irho]*df_w_ri[v1_id];

      dust_flux(irho, k, j, i) = temp_li*flx_rho_li + temp_ri*flx_rho_ri;
      dust_flux(ivx,  k, j, i) = temp_li*df_w_li[v1_id]*flx_rho_li + temp_ri*df_w_ri[v1_id]*flx_rho_ri;
      dust_flux(ivy,  k, j, i) = temp_li*df_w_li[v2_id]*flx_rho_li + temp_ri*df_w_ri[v2_id]*flx_rho_ri;
      dust_flux(ivz,  k, j, i) = temp_li*df_w_li[v3_id]*flx_rho_li + temp_ri*df_w_ri[v3_id]*flx_rho_ri;
    }
  }
  return;
}


void DustFluids::RiemannSolverDustFluids_noPenetration(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &prim_df_l,
                          AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {

  Real df_w_li[(NDUSTVARS)], df_w_ri[(NDUSTVARS)], df_w_roe[(NDUSTVARS)];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int idust = n;
    int irho  = 4*idust;

    int v1_id = irho + 1;
    int v2_id = irho + 2;
    int v3_id = irho + 3;

    int ivx = (IVX + ((index-IVX))%3)   + irho;
    int ivy = (IVX + ((index-IVX)+1)%3) + irho;
    int ivz = (IVX + ((index-IVX)+2)%3) + irho;

#pragma omp simd private(df_w_li, df_w_ri, df_w_roe)
    for (int i=il; i<=iu; i++) {
      df_w_li[irho]  = prim_df_l(irho, i);
      df_w_li[v1_id] = prim_df_l(ivx,  i);
      df_w_li[v2_id] = prim_df_l(ivy,  i);
      df_w_li[v3_id] = prim_df_l(ivz,  i);

      df_w_ri[irho]  = prim_df_r(irho, i);
      df_w_ri[v1_id] = prim_df_r(ivx,  i);
      df_w_ri[v2_id] = prim_df_r(ivy,  i);
      df_w_ri[v3_id] = prim_df_r(ivz,  i);

      Real sqrtdl  = std::sqrt(df_w_li[irho]);
      Real sqrtdr  = std::sqrt(df_w_ri[irho]);
      Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

      df_w_roe[v1_id] = (sqrtdl*df_w_li[v1_id] + sqrtdr*df_w_ri[v1_id])*isdlpdr;
      df_w_roe[v2_id] = (sqrtdl*df_w_li[v2_id] + sqrtdr*df_w_ri[v2_id])*isdlpdr;
      df_w_roe[v3_id] = (sqrtdl*df_w_li[v3_id] + sqrtdr*df_w_ri[v3_id])*isdlpdr;

      Real neg_li = df_w_li[v1_id] < 0.0 ? 1.0 : 0.0;
      Real pos_ri = df_w_ri[v1_id] > 0.0 ? 1.0 : 0.0;
      Real temp   = 1.0 - neg_li * pos_ri;

      Real fra_roe = (df_w_roe[v1_id] > 0.0) ? 1.0 : 0.0;
      if (df_w_roe[v1_id] == 0.0) fra_roe = 0.5;

      dust_flux(irho, k, j, i) = temp*(fra_roe*df_w_li[irho]*df_w_li[v1_id] +
                                 (1.0-fra_roe)*df_w_ri[irho]*df_w_ri[v1_id]);

      dust_flux(ivx, k, j, i) = temp*(fra_roe*df_w_li[v1_id]*dust_flux(irho, k, j, i) +
                                (1.0-fra_roe)*df_w_ri[v1_id]*dust_flux(irho, k, j, i));

      dust_flux(ivy, k, j, i) = temp*(fra_roe*df_w_li[v2_id]*dust_flux(irho, k, j, i) +
                                (1.0-fra_roe)*df_w_ri[v2_id]*dust_flux(irho, k, j, i));

      dust_flux(ivz, k, j, i) = temp*(fra_roe*df_w_li[v3_id]*dust_flux(irho, k, j, i) +
                                (1.0-fra_roe)*df_w_ri[v3_id]*dust_flux(irho, k, j, i));
    }
  }
  return;
}
