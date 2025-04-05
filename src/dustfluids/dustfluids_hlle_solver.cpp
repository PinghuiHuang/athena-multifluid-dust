//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_hlle_solver.cpp
//! \brief spatially isothermal HLLE Riemann solver for dust fludis
//!
//! Computes 1D fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.  This flux is
//! very diffusive, especially for contacts, and so it is not recommended for use in
//! applications.  However, as shown by Einfeldt et al.(1991), it is positively
//! conservative (cannot return negative densities or pressure), so it is a useful
//! option when other approximate solvers fail and/or when extra dissipation is needed.
//!
//!REFERENCES:
//!- E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!  Springer-Verlag, Berlin, (1999) chpt. 10.
//!- Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
//!- A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//!  schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).

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
//! \fn void DustFluids::HLLE_RiemannSolverDustFluids
//! \brief The HLLE Riemann solver for Dust Fluids (spatially isothermal)

void DustFluids::HLLERiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &prim_df_l,
                          AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {

  Real df_w_li[(NDUSTVARS)], df_w_ri[(NDUSTVARS)], df_w_roe[(NDUSTVARS)];
  Real df_fl[(NDUSTVARS)], df_fr[(NDUSTVARS)], df_flxi[(NDUSTVARS)];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int idust = n;
    int irho  = 4*idust;

    int v1_id = irho + 1;
    int v2_id = irho + 2;
    int v3_id = irho + 3;

    int ivx = irho + (IVX + (index-IVX)%3);
    int ivy = irho + (IVX + ((index-IVX)+1)%3);
    int ivz = irho + (IVX + ((index-IVX)+2)%3);

#pragma omp simd private(df_w_li, df_w_ri, df_w_roe, df_fl, df_fr, df_flxi)
    for (int i=il; i<=iu; ++i) {
      const Real &cs = cs_dustfluids_array(idust, k, j, i);

      //Load L/R states into local variables
      df_w_li[irho]  = prim_df_l(irho, i);
      df_w_li[v1_id] = prim_df_l(ivx,  i);
      df_w_li[v2_id] = prim_df_l(ivy,  i);
      df_w_li[v3_id] = prim_df_l(ivz,  i);

      df_w_ri[irho]  = prim_df_r(irho, i);
      df_w_ri[v1_id] = prim_df_r(ivx,  i);
      df_w_ri[v2_id] = prim_df_r(ivy,  i);
      df_w_ri[v3_id] = prim_df_r(ivz,  i);

      //Compute middle state estimates with PVRS (Toro 10.5.2)
      //Real al, ar, el, er;
      Real sqrtdl  = std::sqrt(df_w_li[irho]);
      Real sqrtdr  = std::sqrt(df_w_ri[irho]);
      Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

      df_w_roe[irho]  = sqrtdl*sqrtdr;
      df_w_roe[v1_id] = (sqrtdl*df_w_li[v1_id] + sqrtdr*df_w_ri[v1_id])*isdlpdr;
      df_w_roe[v2_id] = (sqrtdl*df_w_li[v2_id] + sqrtdr*df_w_ri[v2_id])*isdlpdr;
      df_w_roe[v3_id] = (sqrtdl*df_w_li[v3_id] + sqrtdr*df_w_ri[v3_id])*isdlpdr;

      //Compute the max/min wave speeds based on L/R and Roe-averaged values
      Real al = std::min((df_w_roe[v1_id] - cs), (df_w_li[v1_id] - cs));
      Real ar = std::max((df_w_roe[v1_id] + cs), (df_w_ri[v1_id] + cs));

      Real bp = ar > 0.0 ? ar : 0.0;
      Real bm = al < 0.0 ? al : 0.0;

      //Compute L/R df_fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
      Real vxl = df_w_li[v1_id] - bm;
      Real vxr = df_w_ri[v1_id] - bp;

      df_fl[irho]  = vxl * df_w_li[irho];
      df_fr[irho]  = vxr * df_w_ri[irho];

      df_fl[v1_id] = vxl * df_w_li[irho] * df_w_li[v1_id];
      df_fr[v1_id] = vxr * df_w_ri[irho] * df_w_ri[v1_id];

      df_fl[v2_id] = vxl * df_w_li[irho] * df_w_li[v2_id];
      df_fr[v2_id] = vxr * df_w_ri[irho] * df_w_ri[v2_id];

      df_fl[v3_id] = vxl * df_w_li[irho] * df_w_li[v3_id];
      df_fr[v3_id] = vxr * df_w_ri[irho] * df_w_ri[v3_id];

      df_fl[v1_id] += (cs*cs) * df_w_li[irho];
      df_fr[v1_id] += (cs*cs) * df_w_ri[irho];

      //Compute the HLLE df_flux at interface.
      Real tmp = 0.0;
      if (bp != bm) tmp = 0.5*(bp + bm)/(bp - bm);

      df_flxi[irho]  = 0.5*(df_fl[irho]  + df_fr[irho])  + (df_fl[irho]  - df_fr[irho])*tmp;
      df_flxi[v1_id] = 0.5*(df_fl[v1_id] + df_fr[v1_id]) + (df_fl[v1_id] - df_fr[v1_id])*tmp;
      df_flxi[v2_id] = 0.5*(df_fl[v2_id] + df_fr[v2_id]) + (df_fl[v2_id] - df_fr[v2_id])*tmp;
      df_flxi[v3_id] = 0.5*(df_fl[v3_id] + df_fr[v3_id]) + (df_fl[v3_id] - df_fr[v3_id])*tmp;

      dust_flux(irho, k, j, i) = df_flxi[irho];
      dust_flux(ivx,  k, j, i) = df_flxi[v1_id];
      dust_flux(ivy,  k, j, i) = df_flxi[v2_id];
      dust_flux(ivz,  k, j, i) = df_flxi[v3_id];
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void DustFluids::HLLENoCsRiemannSolver_DustFluids
//! \brief The HLLE Riemann solver for Dust Fluids (No Sound Speed)

void DustFluids::HLLENoCsRiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &prim_df_l,
                          AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {

  Real df_w_li[(NDUSTVARS)], df_w_ri[(NDUSTVARS)], df_w_roe[(NDUSTVARS)];
  Real df_fl[(NDUSTVARS)], df_fr[(NDUSTVARS)], df_flxi[(NDUSTVARS)];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int idust = n;
    int irho  = 4*idust;

    int v1_id = irho + 1;
    int v2_id = irho + 2;
    int v3_id = irho + 3;

    int ivx = irho + (IVX + (index-IVX)%3);
    int ivy = irho + (IVX + ((index-IVX)+1)%3);
    int ivz = irho + (IVX + ((index-IVX)+2)%3);

#pragma omp simd private(df_w_li, df_w_ri, df_w_roe, df_fl, df_fr, df_flxi)
    for (int i=il; i<=iu; ++i) {
      //Load L/R states into local variables
      df_w_li[irho]  = prim_df_l(irho, i);
      df_w_li[v1_id] = prim_df_l(ivx,  i);
      df_w_li[v2_id] = prim_df_l(ivy,  i);
      df_w_li[v3_id] = prim_df_l(ivz,  i);

      df_w_ri[irho]  = prim_df_r(irho, i);
      df_w_ri[v1_id] = prim_df_r(ivx,  i);
      df_w_ri[v2_id] = prim_df_r(ivy,  i);
      df_w_ri[v3_id] = prim_df_r(ivz,  i);

      //Compute middle state estimates with PVRS (Toro 10.5.2)
      //Real al, ar, el, er;
      Real sqrtdl  = std::sqrt(df_w_li[irho]);
      Real sqrtdr  = std::sqrt(df_w_ri[irho]);
      Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

      df_w_roe[irho]  = sqrtdl*sqrtdr;
      df_w_roe[v1_id] = (sqrtdl*df_w_li[v1_id] + sqrtdr*df_w_ri[v1_id])*isdlpdr;
      df_w_roe[v2_id] = (sqrtdl*df_w_li[v2_id] + sqrtdr*df_w_ri[v2_id])*isdlpdr;
      df_w_roe[v3_id] = (sqrtdl*df_w_li[v3_id] + sqrtdr*df_w_ri[v3_id])*isdlpdr;

      //Compute the max/min wave speeds based on L/R and Roe-averaged values
      Real al = std::min(df_w_roe[v1_id], df_w_li[v1_id]);
      Real ar = std::max(df_w_roe[v1_id], df_w_ri[v1_id]);

      Real bp = ar > 0.0 ? ar : 0.0;
      Real bm = al < 0.0 ? al : 0.0;

      //Compute L/R df_fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
      Real vxl = df_w_li[v1_id] - bm;
      Real vxr = df_w_ri[v1_id] - bp;

      df_fl[irho] = vxl * df_w_li[irho];
      df_fr[irho] = vxr * df_w_ri[irho];

      df_fl[v1_id] = vxl * df_w_li[irho] * df_w_li[v1_id];
      df_fr[v1_id] = vxr * df_w_ri[irho] * df_w_ri[v1_id];

      df_fl[v2_id] = vxl * df_w_li[irho] * df_w_li[v2_id];
      df_fr[v2_id] = vxr * df_w_ri[irho] * df_w_ri[v2_id];

      df_fl[v3_id] = vxl * df_w_li[irho] * df_w_li[v3_id];
      df_fr[v3_id] = vxr * df_w_ri[irho] * df_w_ri[v3_id];

      //Compute the HLLE df_flux at interface.
      Real tmp = 0.0;
      if (bp != bm) tmp = 0.5*(bp + bm)/(bp - bm);

      df_flxi[irho]  = 0.5*(df_fl[irho] + df_fr[irho]) + (df_fl[irho] - df_fr[irho])*tmp;
      df_flxi[v1_id] = 0.5*(df_fl[v1_id]  + df_fr[v1_id])  + (df_fl[v1_id]  - df_fr[v1_id])*tmp;
      df_flxi[v2_id] = 0.5*(df_fl[v2_id]  + df_fr[v2_id])  + (df_fl[v2_id]  - df_fr[v2_id])*tmp;
      df_flxi[v3_id] = 0.5*(df_fl[v3_id]  + df_fr[v3_id])  + (df_fl[v3_id]  - df_fr[v3_id])*tmp;

      dust_flux(irho, k, j, i) = df_flxi[irho];
      dust_flux(ivx,  k, j, i) = df_flxi[v1_id];
      dust_flux(ivy,  k, j, i) = df_flxi[v2_id];
      dust_flux(ivz,  k, j, i) = df_flxi[v3_id];
    }
  }
  return;
}
