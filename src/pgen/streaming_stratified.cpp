//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  PRIVATE FUNCTION PROTOTYPES:
//  - ran2() - random number generator from NR
//
//  REFERENCE: Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992).*/
//
//======================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp" // ran2()

#if NON_BAROTROPIC_EOS
#error "This problem generator requires isothermal equation of state!"
#endif

#if (!NDUSTFLUIDS)
#error "This problem generator requires NDUSTFLUIDS >= 1!"
#endif

namespace {
Real amp, nwx, nwy, nwz, sigma_g0; // amplitude, Wavenumbers
Real etaVk; // The amplitude of pressure gradient force
int ShBoxCoord, ipert, ifield; // initial pattern
Real gm1, iso_cs;
Real x1size, x2size, x3size;
Real Omega_0, qshear;
Real den_percent_floor;
Real pslope;
Real user_dt;
Real initial_D2G[NDUSTFLUIDS];
Real Stokes_number[NDUSTFLUIDS];
Real kappap, kappap2, AN(0.0), BN(0.0), Psi(0.0), Kai0;
Real Hg, Hdust[NDUSTFLUIDS], Hratio[NDUSTFLUIDS];
//Real Kpar, kx, ky, kz, omg_osi, s_grow;

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void VerticalGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
Real UserTimeStep(MeshBlock *pmb);
Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout);

void LowerZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void UpperZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DustDensityPercentFloor(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
      AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
} // namespace

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties //======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  x1size = mesh_size.x1max - mesh_size.x1min;
  x2size = mesh_size.x2max - mesh_size.x2min;
  x3size = mesh_size.x3max - mesh_size.x3min;

  // initialize global variables
  amp      = pin->GetReal("problem",         "amp");
  sigma_g0 = pin->GetOrAddReal("problem",    "sigma_g0", 1.0);
  nwx      = pin->GetOrAddInteger("problem", "nwx",      1);
  nwy      = pin->GetOrAddInteger("problem", "nwy",      1);
  nwz      = pin->GetOrAddInteger("problem", "nwz",      1);
  ipert    = pin->GetOrAddInteger("problem", "ipert",    1);
  etaVk    = pin->GetReal("problem",         "etaVk");
  iso_cs   = pin->GetReal("hydro",           "iso_sound_speed");

  user_dt = pin->GetOrAddReal("time", "user_dt", 0.0);

  den_percent_floor = pin->GetOrAddReal("dust", "den_percent_floor", 0.0);

  ShBoxCoord = pin->GetOrAddInteger("orbital_advection", "shboxcoord", 1);
  Omega_0    = pin->GetOrAddReal("orbital_advection",    "Omega0",     0.0);
  qshear     = pin->GetOrAddReal("orbital_advection",    "qshear",     0.0);


  kappap  = 2.0*(2.0 - qshear);
  kappap2 = SQR(kappap);
  Kai0    = 2.0*etaVk*iso_cs;

  Hg      = iso_cs/Omega_0;
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      // Dust to gas ratio && dust stokes numbers
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      Hdust[n]         = Hg*Hratio[n];
    }
  }

  if (NDUSTFLUIDS > 0) {
    EnrollUserDustStoppingTime(MyStoppingTime);

    for (int n=0; n<NDUSTFLUIDS; ++n) {
      AN += (initial_D2G[n] * Stokes_number[n])/(1.0 + kappap2 * SQR(Stokes_number[n]));
      BN += (initial_D2G[n])/(1.0 + kappap2 * SQR(Stokes_number[n]));
    }
    AN *= kappap2;
    BN += 1.0;
    Psi = 1.0/(SQR(AN) + kappap2*SQR(BN));
  }


  EnrollUserExplicitSourceFunction(MySource);

  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(UserTimeStep);

  if (ShBoxCoord == 1) {
    if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
      EnrollUserBoundaryFunction(BoundaryFace::inner_x3, LowerZ);
    }
    if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
      EnrollUserBoundaryFunction(BoundaryFace::outer_x3, UpperZ);
    }
  } else if (ShBoxCoord == 2) {
    if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
      EnrollUserBoundaryFunction(BoundaryFace::inner_x2, LowerZ);
    }
    if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
      EnrollUserBoundaryFunction(BoundaryFace::outer_x2, UpperZ);
    }
  }

  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, DustFluidsRatioMaximum, "RatioMax", UserHistoryOperation::max);

  return;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  AllocateUserOutputVariables(16);
  SetUserOutputVariableName(0,  "dust_ratio");
  SetUserOutputVariableName(1,  "Mach_x1");
  SetUserOutputVariableName(2,  "Mach_x2");
  SetUserOutputVariableName(3,  "Mach_x3");
  SetUserOutputVariableName(4,  "Alpha_x1x3");
  SetUserOutputVariableName(5,  "Alpha_x1x2");
  SetUserOutputVariableName(6,  "Alpha_x2x3");
  SetUserOutputVariableName(7,  "Dust_x1_flux");
  SetUserOutputVariableName(8,  "Dust_x2_flux");
  SetUserOutputVariableName(9,  "Dust_x3_flux");
  SetUserOutputVariableName(10, "Dust_x1_kinerg");
  SetUserOutputVariableName(11, "Dust_x2_kinerg");
  SetUserOutputVariableName(12, "Dust_x3_kinerg");
  SetUserOutputVariableName(13, "Vorticity_x1");
  SetUserOutputVariableName(14, "Vorticity_x2");
  SetUserOutputVariableName(15, "Vorticity_x3");
}


//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::int64_t iseed = -1 - gid;
  Real no_orb_adv;
  (!porb->orbital_advection_defined) ? no_orb_adv = 1.0 : no_orb_adv = 0.0;

  if (ShBoxCoord == 1) {  // ShBoxCoord == 1, x-y-z
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {

          Real x_dis = pcoord->x1v(i);
          Real y_dis = pcoord->x2v(j);
          Real z_dis = pcoord->x3v(k);

          Real K_vel    = qshear*Omega_0*x_dis;
          Real gas_vel1 = AN*Kai0*Psi;
          Real gas_vel2 = 0.0;
          gas_vel2      = -no_orb_adv*K_vel - 0.5*kappap2*BN*Kai0*Psi;
          Real gas_vel3 = 0.0;

          Real delta_gas_vel1 = amp*iso_cs*(ran2(&iseed) - 0.5);
          Real delta_gas_vel2 = amp*iso_cs*(ran2(&iseed) - 0.5);
          Real delta_gas_vel3 = amp*iso_cs*(ran2(&iseed) - 0.5);

          Real &gas_dens = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_dens = sigma_g0/(std::sqrt(TWO_PI)*Hg)*std::exp(-SQR(z_dis)/(2.0*SQR(Hg)));
          gas_mom1 = gas_dens * (gas_vel1 + delta_gas_vel1);
          gas_mom2 = gas_dens * (gas_vel2 + delta_gas_vel2);
          gas_mom3 = gas_dens * (gas_vel3 + delta_gas_vel3);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real sigma_d0  = initial_D2G[dust_id]*sigma_g0;
              Real dust_vel1 = 0.0;
              Real dust_vel2 = 0.0;
              Real dust_vel3 = 0.0;

              dust_vel1 = (gas_vel1 + 2.0*Stokes_number[dust_id]*(gas_vel2 + no_orb_adv*K_vel))/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
              dust_vel2 = -no_orb_adv*K_vel + ((gas_vel2 + no_orb_adv*K_vel) - (2.0 - qshear)*Stokes_number[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
              dust_vel3 = 0.0;

              Real delta_dust_vel1 = amp*iso_cs*(ran2(&iseed) - 0.5);
              Real delta_dust_vel2 = amp*iso_cs*(ran2(&iseed) - 0.5);
              Real delta_dust_vel3 = amp*iso_cs*(ran2(&iseed) - 0.5);

              Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

              dust_dens = sigma_d0/(std::sqrt(TWO_PI)*Hdust[n])*std::exp(-SQR(z_dis)/(2.0*SQR(Hdust[n])));
              dust_mom1 = dust_dens * (dust_vel1 + delta_dust_vel1);
              dust_mom2 = dust_dens * (dust_vel2 + delta_dust_vel2);
              dust_mom3 = dust_dens * (dust_vel3 + delta_dust_vel3);
            }
          }
        }
      }
    }
  } else { // ShBoxCoord == 2, x-z plane
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real x_dis = pcoord->x1v(i);
          Real z_dis = pcoord->x2v(j);
          Real y_dis = pcoord->x3v(k);

          Real K_vel    = qshear*Omega_0*x_dis;
          Real gas_vel1 = AN*Kai0*Psi;
          Real gas_vel2 = 0.0;
          Real gas_vel3 = 0.0;
          gas_vel3      = -no_orb_adv*K_vel - 0.5*kappap2*BN*Kai0*Psi;

          Real delta_gas_vel1 = amp*iso_cs*(ran2(&iseed) - 0.5);
          Real delta_gas_vel2 = amp*iso_cs*(ran2(&iseed) - 0.5);
          Real delta_gas_vel3 = amp*iso_cs*(ran2(&iseed) - 0.5);

          Real &gas_dens = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_dens = sigma_g0/(std::sqrt(TWO_PI)*Hg)*std::exp(-SQR(z_dis)/(2.0*SQR(Hg)));
          gas_mom1 = gas_dens * (gas_vel1 + delta_gas_vel1);
          gas_mom2 = gas_dens * (gas_vel2 + delta_gas_vel2);
          gas_mom3 = gas_dens * (gas_vel3 + delta_gas_vel3);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real sigma_d0  = initial_D2G[dust_id]*sigma_g0;
              Real dust_vel1 = 0.0;
              Real dust_vel2 = 0.0;
              Real dust_vel3 = 0.0;

              dust_vel1 = (gas_vel1 + 2.0*Stokes_number[dust_id]*(gas_vel3 + no_orb_adv*K_vel))/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
              dust_vel2 = 0.0;
              dust_vel3 = -no_orb_adv*K_vel + ((gas_vel3 + no_orb_adv*K_vel) - (2.0 - qshear)*Stokes_number[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));

              Real delta_dust_vel1 = amp*iso_cs*(ran2(&iseed)-0.5);
              Real delta_dust_vel2 = amp*iso_cs*(ran2(&iseed)-0.5);
              Real delta_dust_vel3 = amp*iso_cs*(ran2(&iseed)-0.5);

              Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

              dust_dens = sigma_d0/(std::sqrt(TWO_PI)*Hdust[n])*std::exp(-SQR(z_dis)/(2.0*SQR(Hdust[n])));
              dust_mom1 = dust_dens * (dust_vel1 + delta_dust_vel1);
              dust_mom2 = dust_dens * (dust_vel2 + delta_dust_vel2);
              dust_mom3 = dust_dens * (dust_vel3 + delta_dust_vel3);
            }
          }
        }
      }
    }
  }
  return;
}


namespace {
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  PressureGradient(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  //if (pmb->block_size.nx3 > 1)
  VerticalGravity(pmb,  time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &gas_rho  = prim(IDN, k, j, i);
        Real press_gra       = gas_rho*Kai0*Omega_0*dt;
        Real &gas_mom1       = cons(IM1, k, j, i);
        gas_mom1            += press_gra;
      }
    }
  }
  return;
}


void VerticalGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int nc1 = pmb->ncells1;

  if (ShBoxCoord == 1) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        AthenaArray<Real> vert_gravity(nc1);
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real &vertical_dis = pmb->pcoord->x3v(k);
          vert_gravity(i)    = -SQR(Omega_0)*vertical_dis;

          const Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_mom2       = cons(IM2, k, j, i);
          Real &gas_mom3       = cons(IM3, k, j, i);
          gas_mom3            += gas_rho*vert_gravity(i)*dt;
        }

        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            const Real &dust_rho  = prim_df(rho_id, k, j, i);
            Real &dust_mom2       = cons_df(v2_id,  k, j, i);
            Real &dust_mom3       = cons_df(v3_id,  k, j, i);
            dust_mom3            += dust_rho*vert_gravity(i)*dt;
          }
        }
      }
    }
  } else if (ShBoxCoord == 2) {
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        AthenaArray<Real> vert_gravity(nc1);
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real &vertical_dis = pmb->pcoord->x2v(j);
          vert_gravity(i)    = -SQR(Omega_0)*vertical_dis;

          const Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_mom2       = cons(IM2, k, j, i);
          Real &gas_mom3       = cons(IM3, k, j, i);
          gas_mom2            += gas_rho*vert_gravity(i)*dt;
        }

        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            const Real &dust_rho  = prim_df(rho_id, k, j, i);
            Real &dust_mom2       = cons_df(v2_id,  k, j, i);
            Real &dust_mom3       = cons_df(v3_id,  k, j, i);
            dust_mom2            += dust_rho*vert_gravity(i)*dt;
          }
        }
      }
    }
  }
  return;
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

    Real inv_Omega = 1.0/Omega_0;

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &st_time = stopping_time(dust_id, k, j, i);
          st_time       = Stokes_number[dust_id]*inv_Omega;
        }
      }
    }
  }
  return;
}


Real UserTimeStep(MeshBlock *pmb) {

  Real min_user_dt = user_dt;
  return min_user_dt;
}


void LowerZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  Real inv_sqrt_2PIHg = 1.0/(std::sqrt(TWO_PI)*Hg);
  Real inv_2square_Hg = 1.0/(2.0*SQR(Hg));

  Real inv_sqrt_2PIHd[NDUSTFLUIDS];
  Real inv_2square_Hd[NDUSTFLUIDS];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    inv_sqrt_2PIHd[dust_id] = 1.0/(std::sqrt(TWO_PI)*Hdust[dust_id]);
    inv_2square_Hd[dust_id] = 1.0/(2.0*SQR(Hdust[dust_id]));
  }

  if ( ShBoxCoord == 1 ) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real x_dis = pco->x1v(i);
          Real y_dis = pco->x2v(j);
          Real z_dis = pco->x3v(kl-k);

          Real &gas_rho_ac  = prim(IDN, kl, j, i);
          Real &gas_vel1_ac = prim(IM1, kl, j, i);
          Real &gas_vel2_ac = prim(IM2, kl, j, i);
          Real &gas_vel3_ac = prim(IM3, kl, j, i);

          Real &gas_rho_gh  = prim(IDN, kl-k, j, i);
          Real &gas_vel1_gh = prim(IM1, kl-k, j, i);
          Real &gas_vel2_gh = prim(IM2, kl-k, j, i);
          Real &gas_vel3_gh = prim(IM3, kl-k, j, i);

          gas_rho_gh  = sigma_g0*inv_sqrt_2PIHg*std::exp(-SQR(z_dis)*inv_2square_Hg);
          gas_vel1_gh = gas_vel1_ac;
          gas_vel2_gh = gas_vel2_ac;
          gas_vel3_gh = (gas_vel3_ac < 0.0) ? gas_vel3_ac : 0.0;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real sigma_d0  = initial_D2G[dust_id]*sigma_g0;

              Real &dust_rho_ac  = prim_df(rho_id, kl, j, i);
              Real &dust_vel1_ac = prim_df(v1_id,  kl, j, i);
              Real &dust_vel2_ac = prim_df(v2_id,  kl, j, i);
              Real &dust_vel3_ac = prim_df(v3_id,  kl, j, i);

              Real &dust_rho_gh  = prim_df(rho_id, kl-k, j, i);
              Real &dust_vel1_gh = prim_df(v1_id,  kl-k, j, i);
              Real &dust_vel2_gh = prim_df(v2_id,  kl-k, j, i);
              Real &dust_vel3_gh = prim_df(v3_id,  kl-k, j, i);

              dust_rho_gh  = sigma_d0*inv_sqrt_2PIHd[dust_id]*std::exp(-SQR(z_dis)*inv_2square_Hd[dust_id]);
              dust_vel1_gh = dust_vel1_ac;
              dust_vel2_gh = dust_vel2_ac;
              dust_vel3_gh = (dust_vel3_ac < 0.0) ? dust_vel3_ac : 0.0;
            }
          }
        }
      }
    }
  } else if ( ShBoxCoord == 2 ) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real x_dis = pco->x1v(i);
          Real z_dis = pco->x2v(jl-j);
          Real y_dis = pco->x3v(k);

          Real &gas_rho_ac  = prim(IDN, k, jl, i);
          Real &gas_vel1_ac = prim(IM1, k, jl, i);
          Real &gas_vel2_ac = prim(IM2, k, jl, i);
          Real &gas_vel3_ac = prim(IM3, k, jl, i);

          Real &gas_rho_gh  = prim(IDN, k, jl-j, i);
          Real &gas_vel1_gh = prim(IM1, k, jl-j, i);
          Real &gas_vel2_gh = prim(IM2, k, jl-j, i);
          Real &gas_vel3_gh = prim(IM3, k, jl-j, i);

          gas_rho_gh  = sigma_g0*inv_sqrt_2PIHg*std::exp(-SQR(z_dis)*inv_2square_Hg);
          gas_vel1_gh = gas_vel1_ac;
          gas_vel3_gh = gas_vel3_ac;
          gas_vel2_gh = (gas_vel2_ac < 0.0) ? gas_vel2_ac : 0.0;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real sigma_d0  = initial_D2G[dust_id]*sigma_g0;

              Real &dust_rho_ac  = prim_df(rho_id, k, jl, i);
              Real &dust_vel1_ac = prim_df(v1_id,  k, jl, i);
              Real &dust_vel2_ac = prim_df(v2_id,  k, jl, i);
              Real &dust_vel3_ac = prim_df(v3_id,  k, jl, i);

              Real &dust_rho_gh  = prim_df(rho_id, k, jl-j, i);
              Real &dust_vel1_gh = prim_df(v1_id,  k, jl-j, i);
              Real &dust_vel2_gh = prim_df(v2_id,  k, jl-j, i);
              Real &dust_vel3_gh = prim_df(v3_id,  k, jl-j, i);

              dust_rho_gh  = sigma_d0*inv_sqrt_2PIHd[dust_id]*std::exp(-SQR(z_dis)*inv_2square_Hd[dust_id]);
              dust_vel1_gh = dust_vel1_ac;
              dust_vel3_gh = dust_vel3_ac;
              dust_vel2_gh = (dust_vel2_ac < 0.0) ? dust_vel2_ac : 0.0;
            }
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void UpperZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  Real inv_sqrt_2PIHg = 1.0/(std::sqrt(TWO_PI)*Hg);
  Real inv_2square_Hg = 1.0/(2.0*SQR(Hg));

  Real inv_sqrt_2PIHd[NDUSTFLUIDS];
  Real inv_2square_Hd[NDUSTFLUIDS];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    inv_sqrt_2PIHd[dust_id] = 1.0/(std::sqrt(TWO_PI)*Hdust[dust_id]);
    inv_2square_Hd[dust_id] = 1.0/(2.0*SQR(Hdust[dust_id]));
  }

  if ( ShBoxCoord == 1 ) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real x_dis = pco->x1v(i);
          Real y_dis = pco->x2v(j);
          Real z_dis = pco->x3v(ku+k);

          Real &gas_rho_ac  = prim(IDN, ku, j, i);
          Real &gas_vel1_ac = prim(IM1, ku, j, i);
          Real &gas_vel2_ac = prim(IM2, ku, j, i);
          Real &gas_vel3_ac = prim(IM3, ku, j, i);

          Real &gas_rho_gh  = prim(IDN, ku+k, j, i);
          Real &gas_vel1_gh = prim(IM1, ku+k, j, i);
          Real &gas_vel2_gh = prim(IM2, ku+k, j, i);
          Real &gas_vel3_gh = prim(IM3, ku+k, j, i);

          gas_rho_gh  = sigma_g0*inv_sqrt_2PIHg*std::exp(-SQR(z_dis)*inv_2square_Hg);
          gas_vel1_gh = gas_vel1_ac;
          gas_vel2_gh = gas_vel2_ac;
          gas_vel3_gh = (gas_vel3_ac > 0.0) ? gas_vel3_ac : 0.0;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real sigma_d0 = initial_D2G[dust_id]*sigma_g0;

              Real &dust_rho_ac  = prim_df(rho_id, ku, j, i);
              Real &dust_vel1_ac = prim_df(v1_id,  ku, j, i);
              Real &dust_vel2_ac = prim_df(v2_id,  ku, j, i);
              Real &dust_vel3_ac = prim_df(v3_id,  ku, j, i);

              Real &dust_rho_gh  = prim_df(rho_id, ku+k, j, i);
              Real &dust_vel1_gh = prim_df(v1_id,  ku+k, j, i);
              Real &dust_vel2_gh = prim_df(v2_id,  ku+k, j, i);
              Real &dust_vel3_gh = prim_df(v3_id,  ku+k, j, i);

              dust_rho_gh  = sigma_d0*inv_sqrt_2PIHd[dust_id]*std::exp(-SQR(z_dis)*inv_2square_Hd[dust_id]);
              dust_vel1_gh = dust_vel1_ac;
              dust_vel2_gh = dust_vel2_ac;
              dust_vel3_gh = (dust_vel3_ac > 0.0) ? dust_vel3_ac : 0.0;
            }
          }
        }
      }
    }
  } else if ( ShBoxCoord == 2 ) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real x_dis = pco->x1v(i);
          Real z_dis = pco->x2v(ju+j);
          Real y_dis = pco->x3v(k);

          Real &gas_rho_ac  = prim(IDN, k, ju, i);
          Real &gas_vel1_ac = prim(IM1, k, ju, i);
          Real &gas_vel2_ac = prim(IM2, k, ju, i);
          Real &gas_vel3_ac = prim(IM3, k, ju, i);

          Real &gas_rho_gh  = prim(IDN, k, ju+j, i);
          Real &gas_vel1_gh = prim(IM1, k, ju+j, i);
          Real &gas_vel2_gh = prim(IM2, k, ju+j, i);
          Real &gas_vel3_gh = prim(IM3, k, ju+j, i);

          gas_rho_gh  = sigma_g0*inv_sqrt_2PIHg*std::exp(-SQR(z_dis)*inv_2square_Hg);
          gas_vel1_gh = gas_vel1_ac;
          gas_vel3_gh = gas_vel3_ac;
          gas_vel2_gh = (gas_vel2_ac > 0.0) ? gas_vel2_ac : 0.0;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real sigma_d0 = initial_D2G[dust_id]*sigma_g0;

              Real &dust_rho_ac  = prim_df(rho_id, k, ju, i);
              Real &dust_vel1_ac = prim_df(v1_id,  k, ju, i);
              Real &dust_vel2_ac = prim_df(v2_id,  k, ju, i);
              Real &dust_vel3_ac = prim_df(v3_id,  k, ju, i);

              Real &dust_rho_gh  = prim_df(rho_id, k, ju+j, i);
              Real &dust_vel1_gh = prim_df(v1_id,  k, ju+j, i);
              Real &dust_vel2_gh = prim_df(v2_id,  k, ju+j, i);
              Real &dust_vel3_gh = prim_df(v3_id,  k, ju+j, i);

              dust_rho_gh  = sigma_d0*inv_sqrt_2PIHd[dust_id]*std::exp(-SQR(z_dis)*inv_2square_Hd[dust_id]);
              dust_vel1_gh = dust_vel1_ac;
              dust_vel3_gh = dust_vel3_ac;
              dust_vel2_gh = (dust_vel2_ac > 0.0) ? dust_vel2_ac : 0.0;
            }
          }
        }
      }
    }
  }
  return;
}

void DustDensityPercentFloor(MeshBlock *pmb, int il, int iu, int jl, int ju,
            int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

  int nc1 = pmb->ncells1;

  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=kl; k<=ku; ++k) {
      Real x3 = pmb->pcoord->x3v(k);
      for (int j=jl; j<=ju; ++j) {
        Real z_dis;
        Real x2 = pmb->pcoord->x2v(j);
        (ShBoxCoord == 1) ? z_dis = x3 : z_dis = x2;
#pragma omp simd
        for (int i=jl; i<=iu; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real &dust_rho  = prim_df(rho_id, k, j, i);
          Real &dust_vel1 = prim_df(v1_id,  k, j, i);
          Real &dust_vel2 = prim_df(v2_id,  k, j, i);
          Real &dust_vel3 = prim_df(v3_id,  k, j, i);

          Real &dust_dens = cons_df(rho_id, k, j, i);
          Real &dust_mom1 = cons_df(v1_id,  k, j, i);
          Real &dust_mom2 = cons_df(v2_id,  k, j, i);
          Real &dust_mom3 = cons_df(v3_id,  k, j, i);

          Real sigma_d0     = initial_D2G[dust_id]*sigma_g0;
          Real density_init = sigma_d0/(std::sqrt(TWO_PI)*Hdust[n])*std::exp(-SQR(z_dis)/(2.0*SQR(Hdust[n])));

          dust_rho  = std::max(dust_rho, den_percent_floor*density_init);
          dust_dens = dust_rho;
          dust_mom1 = dust_dens*dust_vel1;
          dust_mom2 = dust_dens*dust_vel2;
          dust_mom3 = dust_dens*dust_vel3;
        }
      }
    }
	}
  return;
}


Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout) {
  Real ratio_maximum = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
  AthenaArray<Real> &w       = pmb->phydro->w;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real &gas_rho  = w(IDN,k,j,i);
        //for (int n=0; n<NDUSTFLUIDS; n++) {
        int dust_id     = IDN;
        int rho_id      = 4*dust_id;
        Real &dust_rho  = df_w(rho_id, k, j, i);
        ratio_maximum   = std::max(ratio_maximum, dust_rho/gas_rho);
        //}
      }
    }
  }
  return ratio_maximum;
}

}


void MeshBlock::UserWorkInLoop() {

  Real &time = pmy_mesh->time;
  Real &dt   = pmy_mesh->dt;
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;
  int ku = ke + dk;
  int jl = js - NGHOST;
  int ju = je + NGHOST;
  int il = is - NGHOST;
  int iu = ie + NGHOST;

  if (den_percent_floor > 0.0)
    DustDensityPercentFloor(this, il, iu, jl, ju, kl, ku, pdustfluids->df_w, pdustfluids->df_u);

  return;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  Coordinates *pco = pcoord;

  Real no_orb_adv;
  (!porb->orbital_advection_defined) ? no_orb_adv = 1.0: no_orb_adv = 0.0;
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

  Real gamma = peos->GetGamma();
  Real inv_gamma = 1.0/gamma;

	int rho_id = 0;
	int v1_id  = 1;
	int v2_id  = 2;
	int v3_id  = 3;

  // Isothermal Sound Speed
  Real &sound_speed    = iso_cs;
  Real inv_sound_speed = 1./sound_speed;

	for(int k=ks; k<=ke; k++) {
		for(int j=js; j<=je; j++) {
#pragma omp simd
			for(int i=is; i<=ie; i++) {
        Real &dust_rho  = pdustfluids->df_w(rho_id, k, j, i);
        Real &dust_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        Real &dust_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        Real &dust_vel3 = pdustfluids->df_w(v3_id,  k, j, i);

        Real &gas_rho  = phydro->w(IDN, k, j, i);
        Real &gas_vel1 = phydro->w(IM1, k, j, i);
        Real &gas_vel2 = phydro->w(IM2, k, j, i);
        Real &gas_vel3 = phydro->w(IM3, k, j, i);
        //Real &gas_pres = phydro->w(IPR, k, j, i);

        // Dust-Gas Ratio
        Real &ratio = user_out_var(0, k, j, i);
        ratio       = dust_rho/gas_rho;

        // Mach Number X1
        Real &mach_x1 = user_out_var(1, k, j, i);
        mach_x1       = gas_vel1*inv_sound_speed;

        // Mach Number X2
        Real &mach_x2 = user_out_var(2, k, j, i);
        mach_x2       = gas_vel2*inv_sound_speed;

        // Mach Number X3
        Real &mach_x3 = user_out_var(3, k, j, i);
        mach_x3       = gas_vel3*inv_sound_speed;

        // Alapha X1X3
        Real &Alpha_x1x3 = user_out_var(4, k, j, i);
        Alpha_x1x3       = mach_x1*mach_x3;

        // Alapha X1X2
        Real &Alpha_x1x2 = user_out_var(5, k, j, i);
        Alpha_x1x2       = mach_x1*mach_x2;

        // Alapha X2X3
        Real &Alpha_x2x3 = user_out_var(6, k, j, i);
        Alpha_x2x3       = mach_x2*mach_x3;

        // Dust X1 flux
        Real &dust_x1_flux = user_out_var(7, k, j, i);
        dust_x1_flux       = dust_rho*dust_vel1;

        // Dust X2 flux
        Real &dust_x2_flux = user_out_var(8, k, j, i);
        dust_x2_flux       = dust_rho*dust_vel2;

        // Dust X3 flux
        Real &dust_x3_flux = user_out_var(9, k, j, i);
        dust_x3_flux       = dust_rho*dust_vel3;

        // Dust X1 kinerg
        Real &dust_x1_kinerg = user_out_var(10, k, j, i);
        dust_x1_kinerg       = dust_rho*SQR(dust_vel1);

        // Dust X2 kinerg
        Real &dust_x2_kinerg = user_out_var(11, k, j, i);
        dust_x2_kinerg       = dust_rho*SQR(dust_vel2);

        // Dust X3 kinerg
        Real &dust_x3_kinerg = user_out_var(12, k, j, i);
        dust_x3_kinerg       = dust_rho*SQR(dust_vel3);

        // Gas X1 Vorticity
        Real &gas_vort_x1 = user_out_var(13, k, j, i);
        gas_vort_x1  = (phydro->w(IM3, k,   j+1, i) - phydro->w(IM3, k, j, i))/pco->dx2v(j);
        gas_vort_x1 -= (phydro->w(IM2, k+1, j,   i) - phydro->w(IM2, k, j, i))/pco->dx3v(k);

        // Gas X2 Vorticity
        Real &gas_vort_x2 = user_out_var(14, k, j, i);
        gas_vort_x2  = (phydro->w(IM1, k+1, j, i)   - phydro->w(IM1, k, j, i))/pco->dx3v(k);
        gas_vort_x2 -= (phydro->w(IM3, k,   j, i+1) - phydro->w(IM3, k, j, i))/pco->dx1v(i);

        // Gas X3 Vorticity
        Real &gas_vort_x3 = user_out_var(15, k, j, i);
        gas_vort_x3  = (phydro->w(IM2, k, j,   i+1) - phydro->w(IM2, k, j, i))/pco->dx1v(i);
        gas_vort_x3 -= (phydro->w(IM1, k, j+1, i)   - phydro->w(IM1, k, j, i))/pco->dx2v(j);

			}
		}
	}
	return;
}
