//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in cylindrical
//  coordinate. Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../parameter_input.hpp"

namespace {
void GetCylCoord(Coordinates *pco, Real &rad, Real &phi, Real &z, int i, int j, int k);
Real DenProfileCyl(const Real rad, const Real phi, const Real z);
Real Keplerian_velocity(const Real rad);
Real CsSquare(const Real rad, const Real phi, const Real z);
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void DustVelProfileCyl_NSH(const Real Ts, const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
Real Delta_gas_vr(const Real vk,   const Real SN, const Real QN, const Real Psi);
Real Delta_gas_vphi(const Real vk, const Real SN, const Real QN, const Real Psi);
Real Delta_dust_vr(const Real ts,   const Real vk, const Real d_vgr, const Real d_vgphi);
Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
// User defined time step
Real MyTimeStep(MeshBlock *pmb);
Real Out4massBC(MeshBlock *pmb, int iout);

void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
void DustDensityPercentFloor(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
      AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
// User-defined boundary conditions for disk simulations
void InnerX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

Real gm0, r0, rho_0, dslope, tslope, gamma_gas, dfloor, user_dt, iso_cs2_r0,
x1min, x1max, damping_rate, radius_inner_damping, radius_outer_damping,
inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping,
Omega0, SN_const(0.0), QN_const(0.0), Psi_const(0.0),
initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], eta_gas, beta_gas, ks_gas;
bool Damping_Flag, Isothermal_Flag;
} // namespace

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "GM", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);

  // Get parameters for initial density and velocity
  rho_0  = pin->GetReal("problem",      "rho0");
  dslope = pin->GetOrAddReal("problem", "dslope", 0.0);

  // The parameters of the amplitude of random perturbation on the radial velocity
  Damping_Flag    = pin->GetOrAddBoolean("problem", "Damping_Flag", 1);
  Isothermal_Flag = pin->GetOrAddBoolean("problem", "Isothermal_Flag", 1);

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 2.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, 2./3.);
  radius_outer_damping = x1max*pow(outer_ratio_region, -2./3.);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_inner_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);
  Omega0       = pin->GetOrAddReal("orbital_advection","Omega0",0.0);
  user_dt      = pin->GetOrAddReal("problem", "user_dt",    0.0);

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
    }
  }

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    iso_cs2_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
    tslope     = pin->GetReal("problem", "tslope");
    gamma_gas  = pin->GetReal("hydro", "gamma");
  } else {
    iso_cs2_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor         = pin->GetOrAddReal("hydro", "dfloor", (1024*(float_min)));

  eta_gas  = 0.5*iso_cs2_r0*(tslope + dslope);
  beta_gas = std::sqrt(1.0 + 2.0*eta_gas);
  ks_gas   = 0.5 * beta_gas;

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      SN_const += (initial_D2G[n])/(1.0 + SQR(Stokes_number[n]));
      QN_const += (initial_D2G[n]*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
    }
    Psi_const = 1.0/((SN_const + beta_gas)*(SN_const + 2.0*ks_gas) + SQR(QN_const));
  }

  // Enroll damping zone and local isothermal equation of state
  //EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1_NSH);

  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1_NSH);

  // Enroll user-defined time step
  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);

  EnrollUserDustStoppingTime(MyStoppingTime);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0) {
    std::stringstream msg;
    msg << "This problem file must be setup in the cylindrical coordinate!" << std::endl;
    ATHENA_ERROR(msg);
  }

  Real rad(0.0),   phi(0.0),   z(0.0);
  Real g_v1(0.0),  g_v2(0.0),  g_v3(0.0);
  Real df_v1(0.0), df_v2(0.0), df_v3(0.0);
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  Real orb_defined;
  (porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  // Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        Real &gas_den = phydro->u(IDN, k, j, i);
        Real &gas_m1  = phydro->u(IM1, k, j, i);
        Real &gas_m2  = phydro->u(IM2, k, j, i);
        Real &gas_m3  = phydro->u(IM3, k, j, i);

        // convert to cylindrical coordinates
        GetCylCoord(pcoord, rad, phi, z, i, j, k);

        // compute initial conditions in cylindrical coordinates
        gas_den          = DenProfileCyl(rad, phi, z);
        Real inv_gas_den = 1.0/gas_den;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id    = n;
          int rho_id     = 4*dust_id;
          Real &dust_den = pdustfluids->df_u(rho_id, k, j, i);
          dust_den       = initial_D2G[dust_id] * gas_den;

          SN += (dust_den*inv_gas_den)/(1.0 + SQR(Stokes_number[n]));
          QN += (dust_den*inv_gas_den*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, g_v1, g_v2, g_v3);
        g_v2 -= orb_defined*vK(porb, x1, x2, x3);

        gas_m1 = gas_den * g_v1;
        gas_m2 = gas_den * g_v2;
        gas_m3 = gas_den * g_v3;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_erg  = phydro->u(IEN, k, j, i);
          Real p_over_r  = CsSquare(rad, phi, z);
          gas_erg        = p_over_r * gas_den * igm1;
          gas_erg       += 0.5 * (SQR(gas_m1) + SQR(gas_m2) + SQR(gas_m3))/gas_den;
        }

        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real &dust_den = pdustfluids->df_u(rho_id, k, j, i);
          Real &dust_m1  = pdustfluids->df_u(v1_id,  k, j, i);
          Real &dust_m2  = pdustfluids->df_u(v2_id,  k, j, i);
          Real &dust_m3  = pdustfluids->df_u(v3_id,  k, j, i);

          DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad, phi, z, df_v1, df_v2, df_v3);
          df_v2 -= orb_defined*vK(porb, x1, x2, x3);

          dust_m1 = dust_den * df_v1;
          dust_m2 = dust_den * df_v2;
          dust_m3 = dust_den * df_v3;
        }

      }
    }
  }
  return;
}

namespace {
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &rad = pmb->pcoord->x1v(i);
          Real inv_omega = std::sqrt(rad*rad*rad)*inv_sqrt_gm0;

          Real &st_time = stopping_time(dust_id, k, j, i);
          //Constant Stokes number in disk problems
          st_time = Stokes_number[dust_id]*inv_omega;
        }
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s)
{
  return;
}

Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}


void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        Real &gas_pres = prim(IPR, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real inv_gas_dens = 1.0/gas_dens;
        gas_pres = CsSquare(rad, phi, z)*gas_dens;
        gas_erg  = gas_pres*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}


void InnerWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;
  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        Real vel_gas_R, vel_gas_phi, vel_gas_z;
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

        if (rad_arr(i) <= radius_inner_damping) {
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_init   = DenProfileCyl(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_init = CsSquare(rad_arr(i), phi_arr(i), z_arr(i));

          GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_arr(i), phi_arr(i), z_arr(i), vel_gas_R, vel_gas_phi, vel_gas_z);
          vel_gas_phi -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real &gas_vel1_init = vel_gas_R;
          Real &gas_vel2_init = vel_gas_phi;
          Real &gas_vel3_init = vel_gas_z;
          Real gas_pre_init   = cs_square_init*gas_rho_init;

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);
          Real &gas_pre  = prim(IPR, k, j, i);

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real delta_gas_rho  = (gas_rho_init  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_init - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_init - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_init - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_init  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;
          gas_pre  += delta_gas_pre;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
          gas_erg = gas_pre*igm1 + Ek/gas_dens;
        }
      }
    }
  }
  return;
}


void OuterWaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  int nc1 = pmb->ncells1;

  AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;
  AthenaArray<Real> rad_arr, phi_arr, z_arr;

  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);
  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        Real vel_gas_R, vel_gas_phi, vel_gas_z;
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

        if (rad_arr(i) >= radius_outer_damping) {
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real gas_rho_init   = DenProfileCyl(rad_arr(i), phi_arr(i), z_arr(i));
          Real cs_square_init = CsSquare(rad_arr(i), phi_arr(i), z_arr(i));

          GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_arr(i), phi_arr(i), z_arr(i), vel_gas_R, vel_gas_phi, vel_gas_z);
          vel_gas_phi -= orb_defined*vK(pmb->porb, x1, x2, x3);

          Real &gas_vel1_init = vel_gas_R;
          Real &gas_vel2_init = vel_gas_phi;
          Real &gas_vel3_init = vel_gas_z;
          Real gas_pre_init   = cs_square_init*gas_rho_init;

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real &gas_erg  = cons(IEN, k, j, i);

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);
          Real &gas_pre  = prim(IPR, k, j, i);

          Real delta_gas_rho  = (gas_rho_init  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_init - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_init - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_init - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_init  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;
          gas_pre  += delta_gas_pre;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          Real Ek = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
          gas_erg = gas_pre*igm1 + Ek/gas_dens;
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
void GetCylCoord(Coordinates *pco, Real &rad, Real &phi, Real &z, int i, int j, int k) {
  rad = pco->x1v(i);
  phi = pco->x2v(j);
  z   = pco->x3v(k);
  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates
Real DenProfileCyl(const Real rad, const Real phi, const Real z) {
  Real rho_mid  = rho_0*std::pow(rad/r0, dslope); // 2D
  return std::max(rho_mid, dfloor);
}

//! \f  computes pressure/density in cylindrical coordinates
Real CsSquare(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = iso_cs2_r0*std::pow(rad/r0, tslope);
  return poverr;
}


//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {

  Real vel_Keplerian  = Keplerian_velocity(rad);
  Real vel            = beta_gas*vel_Keplerian;

  Real delta_gas_vr   = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  v1 = delta_gas_vr;
  v2 = vel + delta_gas_vphi;
  v3 = 0.0;
  return;
}


void DustVelProfileCyl_NSH(const Real ts, const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {

  Real vel_Keplerian   = Keplerian_velocity(rad);
  Real delta_gas_vr    = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi  = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  Real delta_dust_vr   = Delta_dust_vr(ts,   vel_Keplerian, delta_gas_vr, delta_gas_vphi);
  Real delta_dust_vphi = Delta_dust_vphi(ts, vel_Keplerian, delta_gas_vr, delta_gas_vphi);

  v1 = delta_dust_vr;
  v2 = vel_Keplerian + delta_dust_vphi;
  v3 = 0.0;
  return;
}

Real Out4massBC(MeshBlock *pmb, int iout) {
  return pmb->ruser_meshblock_data[0](iout);
}

Real Keplerian_velocity(const Real rad) {
  Real vk = std::sqrt(gm0/rad);
  return vk;
}


Real Delta_gas_vr(const Real vk, const Real SN, const Real QN, const Real Psi) {
  Real d_g_vr = -2.0*beta_gas*QN*Psi*(beta_gas - 1.0)*vk;
  return d_g_vr;
}


Real Delta_gas_vphi(const Real vk, const Real SN, const Real QN, const Real Psi) {
  Real d_g_vphi = -1.0*((SN + 2.0*ks_gas)*SN + SQR(QN))*Psi*(beta_gas - 1.0)*vk;
  return d_g_vphi;
}


Real Delta_dust_vr(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi) {
  Real d_d_vr = (2.0*ts*(beta_gas - 1.0)*vk)/(1.0+SQR(ts)) + ((d_vgr + 2.0*ts*d_vgphi)/(1.0+SQR(ts)));
  return d_d_vr;
}


Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi) {
  Real d_d_vphi = ((beta_gas - 1.0)*vk)/(1.0+SQR(ts)) + ((2.0*d_vgphi - ts*d_vgr)/(2.0+2.0*SQR(ts)));
  return d_d_vphi;
}


void InnerX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_gh, phi_gh, z_gh,  rad_ac,   phi_ac,   z_ac;
  Real v1_gh,  v2_gh,  v3_gh, df_v1_gh, df_v2_gh, df_v3_gh;
  Real v1_ac,  v2_ac,  v3_ac, df_v1_ac, df_v2_ac, df_v3_ac;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        x1 = pco->x1v(i);
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, il-i, j, k);

        Real &gas_rho_gh    = prim(IDN, k, j, il-i);
        Real &gas_v1_gh     = prim(IM1, k, j, il-i);
        Real &gas_v2_gh     = prim(IM2, k, j, il-i);
        Real &gas_v3_gh     = prim(IM3, k, j, il-i);
        gas_rho_gh          = DenProfileCyl(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, v1_gh, v2_gh, v3_gh);
        v2_gh -= orb_defined*vK(pmb->porb, x1, x2, x3);
        gas_v1_gh = v1_gh;
        gas_v2_gh = v2_gh;
        gas_v3_gh = v3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, j, il-i);
          gas_pre_gh       = CsSquare(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh = prim_df(rho_id, k, j, il-i);
            Real &dust_v1_gh  = prim_df(v1_id,  k, j, il-i);
            Real &dust_v2_gh  = prim_df(v2_id,  k, j, il-i);
            Real &dust_v3_gh  = prim_df(v3_id,  k, j, il-i);

            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_v1_gh, df_v2_gh, df_v3_gh);
            df_v2_gh -= orb_defined*vK(pmb->porb, x1, x2, x3);

            dust_v1_gh = df_v1_gh;
            dust_v2_gh = df_v2_gh;
            dust_v3_gh = df_v3_gh;
          }
        }

      }
    }
  }
  return;
}


void OuterX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_gh, phi_gh, z_gh,  rad_ac,   phi_ac,   z_ac;
  Real v1_gh,  v2_gh,  v3_gh, df_v1_gh, df_v2_gh, df_v3_gh;
  Real v1_ac,  v2_ac,  v3_ac, df_v1_ac, df_v2_ac, df_v3_ac;
  Real x1, x2, x3;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  for (int k=kl; k<=ku; ++k) {
    x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        x1 = pco->x1v(i);
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, iu+i, j, k);

        Real &gas_rho_gh    = prim(IDN, k, j, iu+i);
        Real &gas_v1_gh     = prim(IM1, k, j, iu+i);
        Real &gas_v2_gh     = prim(IM2, k, j, iu+i);
        Real &gas_v3_gh     = prim(IM3, k, j, iu+i);
        gas_rho_gh          = DenProfileCyl(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, v1_gh, v2_gh, v3_gh);
        v2_gh -= orb_defined*vK(pmb->porb, x1, x2, x3);

        gas_v1_gh = v1_gh;
        gas_v2_gh = v2_gh;
        gas_v3_gh = v3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, j, iu+i);
          gas_pre_gh       = CsSquare(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh = prim_df(rho_id, k, j, iu+i);
            Real &dust_v1_gh  = prim_df(v1_id,  k, j, iu+i);
            Real &dust_v2_gh  = prim_df(v2_id,  k, j, iu+i);
            Real &dust_v3_gh  = prim_df(v3_id,  k, j, iu+i);

            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_v1_gh, df_v2_gh, df_v3_gh);
            df_v2_gh -= orb_defined*vK(pmb->porb, x1, x2, x3);

            dust_v1_gh = df_v1_gh;
            dust_v2_gh = df_v2_gh;
            dust_v3_gh = df_v3_gh;
          }
        }
      }
    }
  }
  return;
}
} // namespace


void MeshBlock::UserWorkInLoop() {

  Real &time = pmy_mesh->time;
  Real &dt   = pmy_mesh->dt;
  Real mygam = peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - NGHOST; int ju = je + NGHOST;
  int il = is - NGHOST; int iu = ie + NGHOST;

  if (Damping_Flag) {
    InnerWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
    OuterWaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);
  }

  if (Isothermal_Flag)
    LocalIsothermalEOS(this, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  return;
}
