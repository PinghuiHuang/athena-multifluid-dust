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

#if (NDUSTFLUIDS != 2)
#error "This problem generator requires ndustfluids == 2!"
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
Real Kpar, kx, ky, kz, omg_osi, s_grow;

// Perturbations
Real rho_gas_real,  rho_gas_imag,  velx_gas_real, velx_gas_imag;
Real vely_gas_real, vely_gas_imag, velz_gas_real, velz_gas_imag;
Real rho_dust_real[NDUSTFLUIDS],  rho_dust_imag[NDUSTFLUIDS];
Real velx_dust_real[NDUSTFLUIDS], velx_dust_imag[NDUSTFLUIDS];
Real vely_dust_real[NDUSTFLUIDS], vely_dust_imag[NDUSTFLUIDS];
Real velz_dust_real[NDUSTFLUIDS], velz_dust_imag[NDUSTFLUIDS];

// User Sources
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
Real UserTimeStep(MeshBlock *pmb);
Real PertEven(const Real fR, const Real fI, const Real x, const Real z, const Real t);
Real PertOdd(const  Real fR, const Real fI, const Real x, const Real z, const Real t);
} // namespace

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties //======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  x1size = mesh_size.x1max - mesh_size.x1min;
  x2size = mesh_size.x2max - mesh_size.x2min;
  x3size = mesh_size.x3max - mesh_size.x3min;

  // initialize global variables
  amp    = pin->GetReal("problem",         "amp");
  rhog0  = pin->GetOrAddReal("problem",    "rhog0", 1.0);
  nwx    = pin->GetOrAddInteger("problem", "nwx",   1);
  nwy    = pin->GetOrAddInteger("problem", "nwy",   1);
  nwz    = pin->GetOrAddInteger("problem", "nwz",   1);
  ipert  = pin->GetOrAddInteger("problem", "ipert", 1);
  etaVk  = pin->GetReal("problem", "etaVk");
  Kpar   = pin->GetReal("problem", "Kpar");
  iso_cs = pin->GetReal("hydro", "iso_sound_speed");

  user_dt = pin->GetOrAddReal("time", "user_dt", 0.0);

  ShBoxCoord = pin->GetOrAddInteger("orbital_advection", "shboxcoord", 1);
  Omega_0    = pin->GetOrAddReal("orbital_advection",    "Omega0",     0.0);
  qshear     = pin->GetOrAddReal("orbital_advection",    "qshear",     0.0);

  kappap     = 2.0*(2.0 - qshear);
  kappap2    = SQR(kappap);
  Kai0       = 2.0*etaVk*iso_cs;

  //kx = Kpar * Omega_0/(etaVk * iso_cs);
  //ky = 0.0;
  //kz = Kpar * Omega_0/(etaVk * iso_cs);

  // Eigenvalues, Eigenvectors of gas
  rho_gas_real  = pin->GetReal("problem", "rho_real_gas");
  rho_gas_imag  = pin->GetReal("problem", "rho_imag_gas");
  velx_gas_real = pin->GetReal("problem", "velx_real_gas");
  velx_gas_imag = pin->GetReal("problem", "velx_imag_gas");
  vely_gas_real = pin->GetReal("problem", "vely_real_gas");
  vely_gas_imag = pin->GetReal("problem", "vely_imag_gas");
  velz_gas_real = pin->GetReal("problem", "velz_real_gas");
  velz_gas_imag = pin->GetReal("problem", "velz_imag_gas");

  // Oscillation rate and Growth rate of instabilities
  omg_osi  = pin->GetReal("problem", "oscillation_rate");
  omg_osi *= Omega_0;
  s_grow   = pin->GetReal("problem", "growth_rate");
  s_grow  *= Omega_0;

  kx = TWO_PI/x1size*nwx;
  if (ShBoxCoord == 1)
    kz = TWO_PI/x3size*nwz;
  else
    kz = TWO_PI/x2size*nwz;

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      // Dust to gas ratio && dust stokes numbers
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));

      // Eigenvalues, Eigenvectors of dust
      rho_dust_real[n]  = pin->GetReal("problem", "rho_real_dust_"  + std::to_string(n+1));
      rho_dust_imag[n]  = pin->GetReal("problem", "rho_imag_dust_"  + std::to_string(n+1));
      velx_dust_real[n] = pin->GetReal("problem", "velx_real_dust_" + std::to_string(n+1));
      velx_dust_imag[n] = pin->GetReal("problem", "velx_imag_dust_" + std::to_string(n+1));
      vely_dust_real[n] = pin->GetReal("problem", "vely_real_dust_" + std::to_string(n+1));
      vely_dust_imag[n] = pin->GetReal("problem", "vely_imag_dust_" + std::to_string(n+1));
      velz_dust_real[n] = pin->GetReal("problem", "velz_real_dust_" + std::to_string(n+1));
      velz_dust_imag[n] = pin->GetReal("problem", "velz_imag_dust_" + std::to_string(n+1));
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

  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::int64_t iseed = -1 - gid;
  //Real x_dis, y_dis, z_dis;
  //
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

          //Real cn = std::cos(kx*x_dis + kz*z_dis);
          //Real sn = std::sin(kx*x_dis + kz*z_dis);
          //Real delta_gas_rho  = amp*rhog0*(rho_gas_real*cn - rho_gas_imag*sn);
          //Real delta_gas_vel1 = amp*etaVk*iso_cs*(velx_gas_real*cn - velx_gas_imag*sn);
          //Real delta_gas_vel2 = amp*etaVk*iso_cs*(vely_gas_real*cn - vely_gas_imag*sn);
          //Real delta_gas_vel3 = amp*etaVk*iso_cs*(velz_gas_real*cn - velz_gas_imag*sn);

          Real delta_gas_rho  = amp*rhog0*PertEven(rho_gas_real, rho_gas_imag, x_dis, z_dis, 0);
          Real delta_gas_vel1 = amp*etaVk*iso_cs*PertEven(velx_gas_real, velx_gas_imag, x_dis, z_dis, 0);
          Real delta_gas_vel2 = amp*etaVk*iso_cs*PertEven(vely_gas_real, vely_gas_imag, x_dis, z_dis, 0);
          Real delta_gas_vel3 = amp*etaVk*iso_cs*PertOdd(velz_gas_real,  velz_gas_imag, x_dis, z_dis, 0);

          //Real delta_gas_rho  = 0.0;
          //Real delta_gas_vel1 = 0.0;
          //Real delta_gas_vel2 = 0.0;
          //Real delta_gas_vel3 = 0.0;

          Real &gas_den  = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_den  = rhog0 + delta_gas_rho;
          gas_mom1 = gas_den * (gas_vel1 + delta_gas_vel1);
          gas_mom2 = gas_den * (gas_vel2 + delta_gas_vel2);
          gas_mom3 = gas_den * (gas_vel3 + delta_gas_vel3);

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

              //Real delta_dust_rho  = amp*rhod0*(rho_dust_real[dust_id]*cn - rho_dust_imag[dust_id]*sn);
              //Real delta_dust_vel1 = amp*etaVk*iso_cs*(velx_dust_real[dust_id]*cn - velx_dust_imag[dust_id]*sn);
              //Real delta_dust_vel2 = amp*etaVk*iso_cs*(vely_dust_real[dust_id]*cn - vely_dust_imag[dust_id]*sn);
              //Real delta_dust_vel3 = amp*etaVk*iso_cs*(velz_dust_real[dust_id]*cn - velz_dust_imag[dust_id]*sn);

              Real delta_dust_rho  = amp*rhod0*PertEven(rho_dust_real[dust_id], rho_dust_imag[dust_id], x_dis, z_dis, 0);
              Real delta_dust_vel1 = amp*etaVk*iso_cs*PertEven(velx_dust_real[dust_id], velx_dust_imag[dust_id], x_dis, z_dis, 0);
              Real delta_dust_vel2 = amp*etaVk*iso_cs*PertEven(vely_dust_real[dust_id], vely_dust_imag[dust_id], x_dis, z_dis, 0);
              Real delta_dust_vel3 = amp*etaVk*iso_cs*PertOdd(velz_dust_real[dust_id],  velz_dust_imag[dust_id], x_dis, z_dis, 0);

              //Real delta_dust_rho  = 0.0;
              //Real delta_dust_vel1 = 0.0;
              //Real delta_dust_vel2 = 0.0;
              //Real delta_dust_vel3 = 0.0;

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

          //Real cn = std::cos(kx*x_dis + kz*z_dis);
          //Real sn = std::sin(kx*x_dis + kz*z_dis);

          //Real delta_gas_rho  = amp*rhog0*(rho_gas_real*cn - rho_gas_imag*sn);
          //Real delta_gas_vel1 = amp*iso_cs*(velx_gas_real*cn - velx_gas_imag*sn);
          //Real delta_gas_vel2 = amp*iso_cs*(velz_gas_real*cn - velz_gas_imag*sn);
          //Real delta_gas_vel3 = amp*iso_cs*(vely_gas_real*cn - vely_gas_imag*sn);

          Real delta_gas_rho  = amp*rhog0*PertEven(rho_gas_real, rho_gas_imag, x_dis, z_dis, 0.0);
          Real delta_gas_vel1 = amp*etaVk*iso_cs*PertEven(velx_gas_real, velx_gas_imag, x_dis, z_dis, 0.0);
          Real delta_gas_vel2 = amp*etaVk*iso_cs*PertOdd(velz_gas_real,  velz_gas_imag, x_dis, z_dis, 0.0);
          Real delta_gas_vel3 = amp*etaVk*iso_cs*PertEven(vely_gas_real, vely_gas_imag, x_dis, z_dis, 0.0);

          //Real delta_gas_rho  = 0.0;
          //Real delta_gas_vel1 = 0.0;
          //Real delta_gas_vel2 = 0.0;
          //Real delta_gas_vel3 = 0.0;

          Real &gas_den  = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_den  = rhog0 + delta_gas_rho;
          gas_mom1 = gas_den * (gas_vel1 + delta_gas_vel1);
          gas_mom2 = gas_den * (gas_vel2 + delta_gas_vel2);
          gas_mom3 = gas_den * (gas_vel3 + delta_gas_vel3);

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

              //Real delta_dust_rho  = amp*rhog0*(rho_dust_real[dust_id]*cn - rho_dust_imag[dust_id]*sn);
              //Real delta_dust_vel1 = amp*iso_cs*(velx_dust_real[dust_id]*cn - velx_dust_imag[dust_id]*sn);
              //Real delta_dust_vel2 = amp*iso_cs*(velz_dust_real[dust_id]*cn - velz_dust_imag[dust_id]*sn);
              //Real delta_dust_vel3 = amp*iso_cs*(vely_dust_real[dust_id]*cn - vely_dust_imag[dust_id]*sn);

              Real delta_dust_rho  = amp*rhod0*PertEven(rho_dust_real[dust_id], rho_dust_imag[dust_id], x_dis, z_dis, 0.0);
              Real delta_dust_vel1 = amp*etaVk*iso_cs*PertEven(velx_dust_real[dust_id], velx_dust_imag[dust_id], x_dis, z_dis, 0.0);
              Real delta_dust_vel2 = amp*etaVk*iso_cs*PertOdd(velz_dust_real[dust_id],  velz_dust_imag[dust_id], x_dis, z_dis, 0.0);
              Real delta_dust_vel3 = amp*etaVk*iso_cs*PertEven(vely_dust_real[dust_id], vely_dust_imag[dust_id], x_dis, z_dis, 0.0);

              //Real delta_dust_rho  = 0.0;
              //Real delta_dust_vel1 = 0.0;
              //Real delta_dust_vel2 = 0.0;
              //Real delta_dust_vel3 = 0.0;

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


Real PertEven(const Real fR, const Real fI, const Real x, const Real z, const Real t) {
  return (fR*cos(kx*x-omg_osi*t)-fI*sin(kx*x-omg_osi*t))*cos(kz*z)*exp(s_grow*t);
}


Real PertOdd(const Real fR, const Real fI, const Real x, const Real z, const Real t) {
  return -(fR*sin(kx*x-omg_osi*t)+fI*cos(kx*x-omg_osi*t))*sin(kz*z)*exp(s_grow*t);
}

}
