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
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp" // ran2()

#if NON_BAROTROPIC_EOS
#error "This problem generator requires isothermal equation of state!"
#endif

namespace {
Real amp, nwx, nwy, nwz, rhog0; // amplitude, Wavenumbers
Real etaVk; // The amplitude of pressure gradient force
int ShBoxCoord, ipert, ifield; // initial pattern
Real gm1, iso_cs;
Real x1size, x2size, x3size;
Real Omega_0, qshear;
Real pslope;
Real user_dt;
Real initial_D2G[NDUSTFLUIDS];
Real Stokes_number[NDUSTFLUIDS];
Real kappap, kappap2, AN(0.0), BN(0.0), Psi(0.0), Kai0;
//Real Kpar, kx, ky, kz, omg_osi, s_grow;

// User Sources
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
Real UserTimeStep(MeshBlock *pmb);
Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout);
Real DustFluidsRatioStd(MeshBlock *pmb, int iout);
Real DustFluidsGasStd(MeshBlock *pmb, int iout);
} // namespace

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties //======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  x1size = mesh_size.x1max - mesh_size.x1min;
  x2size = mesh_size.x2max - mesh_size.x2min;
  x3size = mesh_size.x3max - mesh_size.x3min;

  // initialize global variables
  amp    = pin->GetOrAddReal("problem",    "amp",   1e-5);
  rhog0  = pin->GetOrAddReal("problem",    "rhog0", 1.0);
  nwx    = pin->GetOrAddInteger("problem", "nwx",   1);
  nwy    = pin->GetOrAddInteger("problem", "nwy",   1);
  nwz    = pin->GetOrAddInteger("problem", "nwz",   1);
  ipert  = pin->GetOrAddInteger("problem", "ipert", 1);
  etaVk  = pin->GetOrAddReal("problem",    "etaVk", 0.05);
  iso_cs = pin->GetReal("hydro", "iso_sound_speed");

  user_dt = pin->GetOrAddReal("time", "user_dt", 0.0);

  ShBoxCoord = pin->GetOrAddInteger("orbital_advection", "shboxcoord", 1);
  Omega_0    = pin->GetOrAddReal("orbital_advection",    "Omega0",     0.0);
  qshear     = pin->GetOrAddReal("orbital_advection",    "qshear",     0.0);

  kappap     = 2.0*(2.0 - qshear);
  kappap2    = SQR(kappap);
  Kai0       = 2.0*etaVk*iso_cs;

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      // Dust to gas ratio && dust stokes numbers
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
    }
  }

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      AN += (initial_D2G[n] * Stokes_number[n])/(1.0 + kappap2 * SQR(Stokes_number[n]));
      BN += (initial_D2G[n])/(1.0 + kappap2 * SQR(Stokes_number[n]));
    }
    AN *= kappap2;
    BN += 1.0;
    Psi = 1.0/(SQR(AN) + kappap2*SQR(BN));
  }

  EnrollUserDustStoppingTime(MyStoppingTime);

  EnrollUserExplicitSourceFunction(PressureGradient);

  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(UserTimeStep);

  //AllocateUserHistoryOutput(1);
  //EnrollUserHistoryOutput(0, DustFluidsRatioMaximum, "RatioMax", UserHistoryOperation::max);
  //EnrollUserHistoryOutput(1, DustFluidsRatioStd,     "RatioStd");
  //EnrollUserHistoryOutput(2, DustFluidsGasStd,       "GasStd");

  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::int64_t iseed = -1 - gid;
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
          // NSH-equilibrium
          if(!porb->orbital_advection_defined)
            gas_vel2 = -1.0*K_vel - 0.5*kappap2*BN*Kai0*Psi;
          else
            gas_vel2 = -0.5*kappap2*BN*Kai0*Psi;
          Real gas_vel3 = 0.0;

          Real delta_gas_rho  = 0.0;
          Real delta_gas_vel1 = 0.0;
          Real delta_gas_vel2 = 0.0;
          Real delta_gas_vel3 = 0.0;

          Real &gas_dens = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_dens = rhog0 + delta_gas_rho;
          gas_mom1 = gas_dens* (gas_vel1 + delta_gas_vel1);
          gas_mom2 = gas_dens* (gas_vel2 + delta_gas_vel2);
          gas_mom3 = gas_dens* (gas_vel3 + delta_gas_vel3);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real rhod0     = initial_D2G[dust_id]*rhog0;
              Real dust_vel1 = 0.0;
              Real dust_vel2 = 0.0;
              Real dust_vel3 = 0.0;

              // NSH-equilibrium
              if(!porb->orbital_advection_defined) { // orbital advection turns off
                dust_vel1 = (gas_vel1 + 2.0*Stokes_number[dust_id]*(gas_vel2 + K_vel))/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
                dust_vel2 = -1.0 * K_vel + ((gas_vel2 + K_vel) - (2.0 - qshear)*Stokes_number[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
                dust_vel3 = 0.0;
              } else { // orbital advection truns on
                dust_vel1 = (gas_vel1 + 2.0*Stokes_number[dust_id]*gas_vel2)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
                dust_vel2 = (gas_vel2 - (2.0 - qshear)*Stokes_number[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
                dust_vel3 = 0.0;
              }

              //Real delta_dust_rho  = amp*rhod0*(ran2(&iseed) - 0.5);
              //Real delta_dust_vel1 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
              //Real delta_dust_vel2 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);
              //Real delta_dust_vel3 = amp*etaVk*iso_cs*(ran2(&iseed) - 0.5);

              Real delta_dust_rho  = 0.0;
              Real delta_dust_vel1 = 0.0;
              Real delta_dust_vel2 = 0.0;
              Real delta_dust_vel3 = 0.0;

              Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

              dust_dens = rhod0 + delta_dust_rho;
              dust_mom1 = dust_dens* (dust_vel1 + delta_dust_vel1);
              dust_mom2 = dust_dens* (dust_vel2 + delta_dust_vel2);
              dust_mom3 = dust_dens* (dust_vel3 + delta_dust_vel3);
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

          // NSH-equilibrium
          if(!porb->orbital_advection_defined)
            gas_vel3 = -1.0*K_vel - 0.5*kappap2*BN*Kai0*Psi;
          else
            gas_vel3 = -0.5*kappap2*BN*Kai0*Psi;

          //Real delta_gas_rho  = amp*rhog0* (ran2(&iseed)-0.5);
          //Real delta_gas_vel1 = amp*etaVk*iso_cs*(ran2(&iseed)-0.5);
          //Real delta_gas_vel2 = amp*etaVk*iso_cs*(ran2(&iseed)-0.5);
          //Real delta_gas_vel3 = amp*etaVk*iso_cs*(ran2(&iseed)-0.5);

          Real delta_gas_rho  = 0.0;
          Real delta_gas_vel1 = 0.0;
          Real delta_gas_vel2 = 0.0;
          Real delta_gas_vel3 = 0.0;

          Real &gas_dens = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_dens = rhog0 + delta_gas_rho;
          gas_mom1 = gas_dens* (gas_vel1 + delta_gas_vel1);
          gas_mom2 = gas_dens* (gas_vel2 + delta_gas_vel2);
          gas_mom3 = gas_dens* (gas_vel3 + delta_gas_vel3);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real rhod0     = initial_D2G[dust_id]*rhog0;
              Real dust_vel1 = 0.0;
              Real dust_vel2 = 0.0;
              Real dust_vel3 = 0.0;

              // NSH-equilibrium
              if(!porb->orbital_advection_defined) { // orbital advection turns off
                dust_vel1 = (gas_vel1 + 2.0*Stokes_number[dust_id]*(gas_vel3 + K_vel))/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
                dust_vel2 = 0.0;
                dust_vel3 = -1.0 * K_vel + ((gas_vel3 + K_vel) - (2.0 - qshear)*Stokes_number[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
              } else { // orbital advection truns on
                dust_vel1 = (gas_vel1 + 2.0*Stokes_number[dust_id]*gas_vel3)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
                dust_vel2 = 0.0;
                dust_vel3 = (gas_vel3 - (2.0 - qshear)*Stokes_number[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
              }

              //Real delta_dust_rho  = amp*rhog0* (ran2(&iseed)-0.5);
              //Real delta_dust_vel1 = amp*etaVk*iso_cs*(ran2(&iseed)-0.5);
              //Real delta_dust_vel2 = amp*etaVk*iso_cs*(ran2(&iseed)-0.5);
              //Real delta_dust_vel3 = amp*etaVk*iso_cs*(ran2(&iseed)-0.5);

              Real delta_dust_rho  = 0.0;
              Real delta_dust_vel1 = 0.0;
              Real delta_dust_vel2 = 0.0;
              Real delta_dust_vel3 = 0.0;

              Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

              dust_dens = rhod0 + delta_dust_rho;
              dust_mom1 = dust_dens* (dust_vel1 + delta_dust_vel1);
              dust_mom2 = dust_dens* (dust_vel2 + delta_dust_vel2);
              dust_mom3 = dust_dens* (dust_vel3 + delta_dust_vel3);
            }
          }

        }
      }
    }
  }
  return;
}


namespace {
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &gas_rho  = prim(IDN, k, j, i);
        Real press_gra       = gas_rho*Kai0*Omega_0*dt;
        Real &m1_gas         = cons(IM1, k, j, i);
        m1_gas              += press_gra;
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
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real &st_time = stopping_time(n, k, j, i);
            st_time       = Stokes_number[n]*inv_Omega;
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


Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout) {
  Real ratio_maximum = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
  AthenaArray<Real> &w       = pmb->phydro->w;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real &gas_rho  = w(IDN,k,j,i);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id     = n;
          int rho_id      = 4*dust_id;
          Real &dust_rho  = df_w(rho_id,k,j,i);
          ratio_maximum   = std::max(ratio_maximum, dust_rho/gas_rho);
        }
      }
    }
  }
  return ratio_maximum;
}


Real DustFluidsRatioStd(MeshBlock *pmb, int iout) {
  Real ratio_sum = 0.0;
  Real ratio_std = 0.0;
  Real number    = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
  AthenaArray<Real> &w       = pmb->phydro->w;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        number        += 1.0;
        Real &gas_rho  = w(IDN,k,j,i);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id     = n;
          int rho_id      = 4*dust_id;
          Real &dust_rho  = df_w(rho_id,k,j,i);
          ratio_sum      += (dust_rho/gas_rho);
        }
      }
    }
  }

  Real ratio_average = ratio_sum/number;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real &gas_rho  = w(IDN,k,j,i);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id     = n;
          int rho_id      = 4*dust_id;
          Real &dust_rho  = df_w(rho_id,k,j,i);
          ratio_std      += SQR(dust_rho/gas_rho - ratio_average)/number;
        }
      }
    }
  }

  ratio_std = std::sqrt(ratio_std);
  return ratio_std;
}


Real DustFluidsGasStd(MeshBlock *pmb, int iout) {
  Real gas_sum = 0.0;
  Real gas_std = 0.0;
  Real number  = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
  AthenaArray<Real> &w       = pmb->phydro->w;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        number        += 1.0;
        Real &gas_rho  = w(IDN,k,j,i);
        gas_sum       += gas_rho;
      }
    }
  }

  Real gas_average = gas_sum/number;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real &gas_rho  = w(IDN,k,j,i);
        gas_std       += SQR(gas_rho - gas_average)/number;
      }
    }
  }

  gas_std = std::sqrt(gas_std);
  return gas_std;
}
}

