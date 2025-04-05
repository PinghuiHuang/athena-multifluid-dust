//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

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
#include "../utils/utils.hpp" // ran2()

#if (!NDUSTFLUIDS)
#error "This problem generator requires NDUSTFLUIDS > 0!"
#endif

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_bump(const Real rad, const Real phi, const Real z, const Real diff);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
      const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void DustVelProfileCyl_NSH(const Real Ts, const Real SN, const Real QN, const Real Psi,
      const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void Keplerian_interpolate(const Real r_active, const Real r_ghost, const Real vphi_active,
      Real &vphi_ghost);
void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
      const Real slope, Real &rho_ghost);
// problem parameters which are useful to make global to this file
//
Real Keplerian_velocity(const Real rad);
Real Delta_gas_vr(const Real vk,    const Real SN, const Real QN,    const Real Psi);
Real Delta_gas_vphi(const Real vk,  const Real SN, const Real QN,    const Real Psi);
Real Delta_dust_vr(const Real ts,   const Real vk, const Real d_vgr, const Real d_vgphi);
Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi);

// problem parameters which are useful to make global to this file
Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, beta, alpha_vis;
Real tau_relax, rs, gmstar, gmp, rad_planet, phi_planet_0, z_planet, t0_planet, t_end_planet, vk_planet, omega_planet, inv_omega_planet;
Real dfloor, dffloor, Omega0, user_dt;

Real initial_D2G[NDUSTFLUIDS], ring_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS],
weight_dust[NDUSTFLUIDS], Dd[NDUSTFLUIDS];
bool Damping_Flag, Isothermal_Flag, Bump_Flag;

Real x1min, x1max, tau_damping, damping_rate;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region,
    inner_width_damping, outer_width_damping;
Real A_bump, sigma_bump, r0_bump;
Real eta_gas, beta_gas, ks_gas;
Real SN_const(0.0), QN_const(0.0), Psi_const(0.0);

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// User Stopping time
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
      int is, int ie, int js, int je, int ks, int ke);

// User-defined orbital velocity
Real UserOrbitalVelocity(OrbitalAdvection * porb, Real x1, Real x2, Real x3);
// x1 direction
Real UserOrbitalVelocity_r(OrbitalAdvection * porb, Real x1, Real x2, Real x3);
// x3 direction in Cartesian and cylindrical, x2 direction in spherical polar
Real UserOrbitalVelocity_z(OrbitalAdvection * porb, Real x1, Real x2, Real x3);
int RefinementCondition(MeshBlock *pmb);
void Vr_interpolate_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "GM", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);
  Damping_Flag    = pin->GetBoolean("problem", "Damping_Flag");
  Isothermal_Flag = pin->GetBoolean("problem", "Isothermal_Flag");
  Bump_Flag       = pin->GetBoolean("problem", "Bump_Flag");

  // Get parameters for initial density and velocity
  rho0 = pin->GetReal("problem", "rho0");
  dslope = pin->GetOrAddReal("problem", "dslope", -1.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
    pslope     = pin->GetOrAddReal("problem", "pslope",     -0.5);
    gamma_gas  = pin->GetReal("hydro", "gamma");
    beta       = pin->GetOrAddReal("problem", "beta", 0.0);
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor         = pin->GetOrAddReal("hydro", "dfloor",  (1024*(float_min)));
  dffloor        = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));
  Omega0         = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
  alpha_vis       = pin->GetOrAddReal("problem", "alpha_vis",  0.0);

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      ring_D2G[n]      = pin->GetReal("dust", "ring_D2G_" + std::to_string(n+1));
      Dd[n]            = pin->GetReal("dust", "nu_dust_" + std::to_string(n+1));
      weight_dust[n]   = 2.0/(Stokes_number[n] + SQR(1.0+initial_D2G[n])/Stokes_number[n]);
    }
  }

  eta_gas  = 0.5*p0_over_r0*(pslope + dslope);
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

  // The parameters of one planet
  tau_relax        = pin->GetOrAddReal("hydro",    "tau_relax",    0.01);
  rad_planet       = pin->GetOrAddReal("problem",  "rad_planet",   1.0); // radial position of the planet
  phi_planet_0     = pin->GetOrAddReal("problem",  "phi_planet",   0.0); // azimuthal position of the planet
  z_planet         = pin->GetOrAddReal("problem",  "z_planet",     0.0); // vertical position of the planet
  t0_planet        = (pin->GetOrAddReal("problem", "t0_planet",    0.0))*TWO_PI; // time to put in the planet
  t_end_planet     = (pin->GetOrAddReal("problem", "t_end_planet", HUGE_NUMBER))*TWO_PI; // time to disapear the planet
  gmp              = pin->GetOrAddReal("problem",  "GMp",          0.0); // GM of the planet
  user_dt          = pin->GetOrAddReal("problem",  "user_dt",      0.0);
  vk_planet        = std::sqrt(gm0/rad_planet);
  omega_planet     = vk_planet/rad_planet;
  inv_omega_planet = 1.0/omega_planet;

  if (t_end_planet < t0_planet)
    t_end_planet = t0_planet;

  rs  = pin->GetOrAddReal("problem", "rs", 0.6); // softening length of the gravitational potential of planets
  rs *= std::sqrt(p0_over_r0);   // softening length of the gravitational potential of planets

  A_bump     = pin->GetOrAddReal("problem", "A_bump",     0.0);
  sigma_bump = pin->GetOrAddReal("problem", "sigma_bump", 0.0);
  r0_bump    = pin->GetOrAddReal("problem", "r0_bump",    0.0);

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.2);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, 2./3.);
  radius_outer_damping = x1max*pow(outer_ratio_region, -2./3.);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  }
  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiskInnerX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiskOuterX3);
  }

  EnrollUserDustStoppingTime(MyStoppingTime);

  //EnrollDustDiffusivity(MyDustDiffusivity);

  // Enroll damping zone and local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined AMR criterion
  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);

  // Enroll user orbital velocity
  //EnrollOrbitalVelocity(UserOrbitalVelocity);
  // x1 direction
  //EnrollOrbitalVelocityDerivative(0, UserOrbitalVelocity_r);
  // x3 direction in Cartesian and cylindrical, x2 direction in spherical polar
  //EnrollOrbitalVelocityDerivative(1, UserOrbitalVelocity_z);


  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  bool diffusion_corretion = (pdustfluids->dfdif.dustfluids_diffusion_defined);
  diffusion_corretion = (diffusion_corretion && (pdustfluids->dfdif.Momentum_Diffusion_Flag));

  Real rad(0.0), phi(0.0), z(0.0);
  Real x1, x2, x3;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
  //  Initialize density and momenta
  Real inv_2sigma2 = 1./(2.0*SQR(sigma_bump));
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        x1 = pcoord->x1v(i);
        GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
        // compute initial conditions in cylindrical coordinates
        Real den_gas     = DenProfileCyl_gas(rad, phi, z);
        Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
        if (porb->orbital_advection_defined)
          vel_gas_phi -= vK(porb, x1, x2, x3);

        phydro->u(IDN, k, j, i) = den_gas;
        phydro->u(IM1, k, j, i) = 0.0;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM2, k, j, i) = den_gas*vel_gas_phi;
          phydro->u(IM3, k, j, i) = 0.0;
        } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = den_gas*vel_gas_phi;
        }

        if (NON_BAROTROPIC_EOS) {
          Real p_over_r = PoverRho(rad, phi, z);
          phydro->u(IEN, k, j, i)  = p_over_r*phydro->u(IDN, k, j, i)*igm1;
          phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                        + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            // compute initial conditions in cylindrical coordinates
            Real vel_dust_phi = VelProfileCyl_dust(rad, phi, z);
            if (porb->orbital_advection_defined)
              vel_dust_phi -= vK(porb, x1, x2, x3);

            Real den_dust_1 = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
            Real den_dust_2 = initial_D2G[dust_id]*A_bump*den_gas*std::exp(-SQR(rad - r0_bump)*inv_2sigma2);
            Real den_dust_total = den_dust_1 + den_dust_2;

            pdustfluids->df_u(rho_id, k, j, i) = den_dust_total;
            if (diffusion_corretion) {
              //Real diff_mom = A_bump*Dd[dust_id]*std::exp(-SQR(rad - r0_bump)*inv_2sigma2)*
                                //initial_D2G[dust_id]*(SQR(rad) - rad*r0_bump + dslope*SQR(sigma_bump))/(rad*SQR(sigma_bump));
              Real diff_mom = den_dust_2*(rad - r0_bump)*Dd[dust_id]*2.0*inv_2sigma2;
              pdustfluids->df_u(v1_id, k, j, i) = diff_mom;
              pdustfluids->dfccdif.diff_mom_cc(v1_id, k, j, i) = diff_mom;
            } else
              pdustfluids->df_u(v1_id,  k, j, i) = 0.0;

            if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
              pdustfluids->df_u(v2_id, k, j, i) = den_dust_total*vel_dust_phi;
              pdustfluids->df_u(v3_id, k, j, i) = 0.0;
            } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
              pdustfluids->df_u(v2_id, k, j, i) = 0.0;
              pdustfluids->df_u(v3_id, k, j, i) = den_dust_total*vel_dust_phi;
            }
          }
        }
      }
    }
  }
  return;
}


namespace {
//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  //if (gmp > 0.0)
    //PlanetaryGravity(pmb, time, dt, prim, prim_df, bcc, cons, cons_df);

  if (Damping_Flag) {
    InnerWavedamping(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
    OuterWavedamping(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  }

  if (Isothermal_Flag && NON_BAROTROPIC_EOS)
    LocalIsothermalEOS(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  else if (beta > 0.0)
    ThermalRelaxation(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


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
          Real rad, phi, z;
          //const Real &gas_rho = prim(IDN, k, j, i);
          Real &st_time = stopping_time(dust_id, k, j, i);

          //Constant Stokes number in disk problems
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          st_time = Stokes_number[dust_id]*std::pow(rad, 1.5)*inv_sqrt_gm0;
          //st_time = Stokes_number[dust_id]*std::pow(rad, 1.5)*inv_sqrt_gm0/gas_rho;
        }
      }
    }
  }
  return;
}


void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

    Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);
    for (int n=0; n<NDUSTFLUIDS; n++) {
      int dust_id = n;
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

            Real inv_Omega_K = std::pow(rad, 1.5)*inv_sqrt_gm0;

            const Real &gas_pre = w(IPR, k, j, i);
            const Real &gas_rho = w(IDN, k, j, i);
            Real inv_gas_rho    = 1.0/gas_rho;
            Real nu_gas         = alpha_vis*inv_Omega_K*gas_pre*inv_gas_rho;

            Real &diffusivity = nu_dust(dust_id, k, j, i);
            diffusivity       = nu_gas/(1.0 + SQR(Stokes_number[dust_id]*inv_gas_rho));

            Real &soundspeed  = cs_dust(dust_id, k, j, i);
            soundspeed        = std::sqrt(diffusivity*inv_Omega_K);
          }
        }
      }
    }
  return;
}
//
//! \f  computes rotational velocity in cylindrical coordinates
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
  Real vel_Keplerian  = Keplerian_velocity(rad);
  Real vel            = beta_gas*vel_Keplerian;

  Real delta_gas_vr   = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = delta_gas_vr;
    v2 = vel + delta_gas_vphi;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = delta_gas_vr;
    v2 = 0.0;
    v3 = vel + delta_gas_vphi;
  }
  return;
}


void DustVelProfileCyl_NSH(const Real ts, const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {

  Real vel_Keplerian   = Keplerian_velocity(rad);
  Real delta_gas_vr    = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi  = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  Real delta_dust_vr   = Delta_dust_vr(ts,   vel_Keplerian, delta_gas_vr, delta_gas_vphi);
  Real delta_dust_vphi = Delta_dust_vphi(ts, vel_Keplerian, delta_gas_vr, delta_gas_vphi);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = delta_dust_vr;
    v2 = vel_Keplerian+delta_dust_vphi;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = delta_dust_vr;
    v2 = 0.0;
    v3 = vel_Keplerian+delta_dust_vphi;
  }
  return;
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



// Add planet
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real phi_planet_move = omega_planet*time + phi_planet_0;
  phi_planet_move -= Omega0*time;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        if (time >= t0_planet) {
          Real cs_planet  = std::sqrt(PoverRho(rad_planet, phi, z));
          // Thermal Mass gMth
          Real gMth       = gm0*SQR(cs_planet)*cs_planet/(SQR(vk_planet)*vk_planet);
          Real t_growth   = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
          Real t_disapear = 50.*TWO_PI*inv_omega_planet*(gmp/gMth);
          //Real planet_gm;
          //if (time <= t_end_planet)
            //planet_gm = gmp*std::sin(0.5*PI*std::min(time/t_growth, 1.0));
          //else
            //planet_gm = gmp*(1.0 - std::sin(0.5*PI*std::min((time-t_end_planet)/t_disapear, 1.0)));
          Real planet_gm = gmp;

          Real x_dis   = rad*std::cos(phi) - rad_planet*std::cos(phi_planet_move);
          Real y_dis   = rad*std::sin(phi) - rad_planet*std::sin(phi_planet_move);
          Real z_dis   = z - z_planet;

          Real r_dis   = rad - rad_planet*cos(phi-phi_planet_move);
          Real phi_dis = rad_planet*sin(phi-phi_planet_move);

          Real distance_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);
          //Real distance        = std::sqrt(distance_square);

          //second order gravity
          Real sec_g   = planet_gm/pow(distance_square+SQR(rs),1.5);
          Real acc_r   = sec_g*r_dis;   // radial acceleration
          Real acc_phi = sec_g*phi_dis; // asimuthal acceleration
          Real acc_z   = sec_g*z_dis;   // vertical acceleartion

          //fourth order gravity
          //Real forth_g = planet_gm*(5.0*SQR(rs)+2.0*dis2)/(2.0*pow(SQR(rs)+SQR(dis), 2.5));
          //Real acc_r   = forth_g*r_dis;   // radial acceleration
          //Real acc_phi = forth_g*phi_dis; // asimuthal acceleration
          //Real acc_z   = forth_g*z_dis;   // vertical acceleartion

          //sixth order gravity
          //Real sixth_g = planet_gm*(35.0*SQR(SQR(rs))+28.0*SQR(rs)*dis2+8.0*SQR(dis2))/(8.0*pow(SQR(rs)+dis2, 3.5));
          //Real acc_r   = sixth_g*r_dis;   // radial acceleration
          //Real acc_phi = sixth_g*phi_dis; // asimuthal acceleration
          //Real acc_z   = sixth_g*z_dis;   // vertical acceleartion

          const Real &gas_rho  = prim(IDN, k, j, i);
          const Real &gas_vel1 = prim(IM1, k, j, i);
          const Real &gas_vel2 = prim(IM2, k, j, i);
          const Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_dens  = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_mom1 = -dt*gas_rho*acc_r;
          Real delta_mom2 = -dt*gas_rho*acc_phi;
          Real delta_mom3 = -dt*gas_rho*acc_z;

          gas_mom1 += delta_mom1;
          gas_mom2 += delta_mom2;
          gas_mom3 += delta_mom3;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_erg  = cons(IEN, k, j, i);
            gas_erg       += (delta_mom1*gas_vel1 + delta_mom2*gas_vel2 + delta_mom3*gas_vel3);
          }

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              const Real &dust_rho  = prim_df(rho_id, k, j, i);
              const Real &dust_vel1 = prim_df(v1_id,  k, j, i);
              const Real &dust_vel2 = prim_df(v2_id,  k, j, i);
              const Real &dust_vel3 = prim_df(v3_id,  k, j, i);

              Real &dust_dens  = cons_df(rho_id, k, j, i);
              Real &dust_mom1 = cons_df(v1_id,  k, j, i);
              Real &dust_mom2 = cons_df(v2_id,  k, j, i);
              Real &dust_mom3 = cons_df(v3_id,  k, j, i);

              Real delta_mom_dust_1 = -dt*dust_rho*acc_r;
              Real delta_mom_dust_2 = -dt*dust_rho*acc_phi;
              Real delta_mom_dust_3 = -dt*dust_rho*acc_z;

              dust_mom1 += delta_mom_dust_1;
              dust_mom2 += delta_mom_dust_2;
              dust_mom3 += delta_mom_dust_3;
            }
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
// Wavedamping function
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_inner_damp = 1.0/inner_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  AthenaArray<Real> Vel_K, omega_dyn, R_func, inv_damping_tau;
  Vel_K.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          // compute initial conditions in cylindrical coordinates
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad < radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad*rad*rad));
            R_func(i)          = SQR((rad - radius_inner_damping)*inv_inner_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(alpha_vis*cs_square/rad/omega_dyn(i));

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = vel_gas_phi;
            Real gas_vel3_0 = 0.0;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*inv_damping_tau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                 + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              Real x1 = pmb->pcoord->x1v(i);
              Real rad, phi, z;
              // compute initial conditions in cylindrical coordinates
              GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
              if (rad >= x1min && rad < radius_inner_damping) {
                // See de Val-Borro et al. 2006 & 2007
                Real cs_square = PoverRho(rad, phi, z);
                Real pre_diff  = (pslope + dslope)*cs_square;

                Real dust_rho_0    = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
                Real vel_dust_phi  = VelProfileCyl_dust(rad, phi, z);
                Real vel_K         = vK(pmb->porb, x1, x2, x3);
                Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
                vel_dust_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

                Real dust_vel1_0  = vel_dust_r;
                Real dust_vel2_0  = vel_dust_phi;
                Real dust_vel3_0  = 0.0;

                Real &dust_dens    = cons_df(rho_id, k, j, i);
                Real &dust_mom1    = cons_df(v1_id,  k, j, i);
                Real &dust_mom2    = cons_df(v2_id,  k, j, i);
                Real &dust_mom3    = cons_df(v3_id,  k, j, i);
                Real inv_dens_dust = 1.0/dust_dens;

                Real dust_vel1 = dust_mom1*inv_dens_dust;
                Real dust_vel2 = dust_mom2*inv_dens_dust;
                Real dust_vel3 = dust_mom3*inv_dens_dust;

                Real delta_dust_dens = (dust_rho_0  - dust_dens)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

                dust_dens += delta_dust_dens;
                dust_vel1 += delta_dust_vel1;
                dust_vel2 += delta_dust_vel2;
                dust_vel3 += delta_dust_vel3;

                dust_mom1 = dust_dens*dust_vel1;
                dust_mom2 = dust_dens*dust_vel2;
                dust_mom3 = dust_dens*dust_vel3;
              }
            }
          }
        }
      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          // compute initial conditions in cylindrical coordinates
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad >= x1min && rad < radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad*rad*rad));
            R_func(i)          = SQR((rad - radius_inner_damping)*inv_inner_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(alpha_vis*cs_square/rad/omega_dyn(i));

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = 0.0;
            Real gas_vel3_0 = vel_gas_phi;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*inv_damping_tau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                 + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              Real x1 = pmb->pcoord->x1v(i);
              Real rad, phi, z;
              // compute initial conditions in cylindrical coordinates
              GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
              if (rad >= x1min && rad < radius_inner_damping) {
                // See de Val-Borro et al. 2006 & 2007
                Real cs_square = PoverRho(rad, phi, z);
                Real pre_diff  = (pslope + dslope)*cs_square;

                Real dust_rho_0    = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
                Real vel_dust_phi  = VelProfileCyl_dust(rad, phi, z);
                Real vel_K         = vK(pmb->porb, x1, x2, x3);
                Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
                vel_dust_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

                Real dust_vel1_0  = vel_dust_r;
                Real dust_vel2_0  = 0.0;
                Real dust_vel3_0  = vel_dust_phi;

                Real &dust_dens    = cons_df(rho_id, k, j, i);
                Real &dust_mom1    = cons_df(v1_id, k, j, i);
                Real &dust_mom2    = cons_df(v2_id, k, j, i);
                Real &dust_mom3    = cons_df(v3_id, k, j, i);
                Real inv_dens_dust = 1.0/dust_dens;

                Real dust_vel1 = dust_mom1*inv_dens_dust;
                Real dust_vel2 = dust_mom2*inv_dens_dust;
                Real dust_vel3 = dust_mom3*inv_dens_dust;

                Real delta_dust_dens = (dust_rho_0  - dust_dens)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

                dust_dens += delta_dust_dens;
                dust_vel1 += delta_dust_vel1;
                dust_vel2 += delta_dust_vel2;
                dust_vel3 += delta_dust_vel3;

                dust_mom1 = dust_dens*dust_vel1;
                dust_mom2 = dust_dens*dust_vel2;
                dust_mom3 = dust_dens*dust_vel3;
              }
            }
          }
        }
      }
    }
  }
  return;
}


void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  Real inv_outer_damp = 1.0/outer_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  if (pmb->porb->orbital_advection_defined)
    orb_defined = 1.0;
  else
    orb_defined = 0.0;

  AthenaArray<Real> Vel_K, omega_dyn, R_func, inv_damping_tau;
  Vel_K.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad >= radius_outer_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad*rad*rad));
            R_func(i)          = SQR((rad - radius_outer_damping)*inv_outer_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(alpha_vis*cs_square/rad/omega_dyn(i));

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = vel_gas_phi;
            Real gas_vel3_0 = 0.0;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*inv_damping_tau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                      + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              Real x1 = pmb->pcoord->x1v(i);
              Real rad, phi, z;
              GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
              if (rad <= x1max && rad >= radius_outer_damping) {
                Real cs_square = PoverRho(rad, phi, z);
                Real pre_diff  = (pslope + dslope)*cs_square;

                Real dust_rho_0    = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
                Real vel_dust_phi  = VelProfileCyl_dust(rad, phi, z);
                Real vel_K         = vK(pmb->porb, x1, x2, x3);
                Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
                vel_dust_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

                Real dust_vel1_0  = vel_dust_r;
                Real dust_vel2_0  = vel_dust_phi;
                Real dust_vel3_0  = 0.0;

                Real &dust_dens    = cons_df(rho_id, k, j, i);
                Real &dust_mom1    = cons_df(v1_id,  k, j, i);
                Real &dust_mom2    = cons_df(v2_id,  k, j, i);
                Real &dust_mom3    = cons_df(v3_id,  k, j, i);
                Real inv_dens_dust = 1.0/dust_dens;

                Real dust_vel1 = dust_mom1*inv_dens_dust;
                Real dust_vel2 = dust_mom2*inv_dens_dust;
                Real dust_vel3 = dust_mom3*inv_dens_dust;

                Real delta_dust_dens = (dust_rho_0  - dust_dens)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

                dust_dens += delta_dust_dens;
                dust_vel1 += delta_dust_vel1;
                dust_vel2 += delta_dust_vel2;
                dust_vel3 += delta_dust_vel3;

                dust_mom1 = dust_dens*dust_vel1;
                dust_mom2 = dust_dens*dust_vel2;
                dust_mom3 = dust_dens*dust_vel3;
              }
            }
          }
        }
      } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          Real rad, phi, z;
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          if (rad <= x1max && rad >= radius_outer_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad*rad*rad));
            R_func(i)          = SQR((rad - radius_outer_damping)*inv_outer_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs_square = PoverRho(rad, phi, z);
            Real vis_vel_r = -1.5*(alpha_vis*cs_square/rad/omega_dyn(i));

            Real gas_rho_0    = DenProfileCyl_gas(rad, phi, z);
            Real vel_gas_phi  = VelProfileCyl_gas(rad, phi, z);
            vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

            Real gas_vel1_0 = vis_vel_r;
            Real gas_vel2_0 = 0.0;
            Real gas_vel3_0 = vel_gas_phi;

            Real &gas_dens    = cons(IDN, k, j, i);
            Real &gas_mom1    = cons(IM1, k, j, i);
            Real &gas_mom2    = cons(IM2, k, j, i);
            Real &gas_mom3    = cons(IM3, k, j, i);
            Real inv_dens_gas = 1.0/gas_dens;
            Real gas_pre      = 0.0;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg     = cons(IEN, k, j, i);
              Real internal_erg = gas_erg - 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_dens_gas;
              gas_pre           = internal_erg*(gamma_gas - 1.0);
            }

            Real gas_vel1 = gas_mom1*inv_dens_gas;
            Real gas_vel2 = gas_mom2*inv_dens_gas;
            Real gas_vel3 = gas_mom3*inv_dens_gas;

            Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            gas_dens += delta_gas_dens;
            gas_vel1 += delta_gas_vel1;
            gas_vel2 += delta_gas_vel2;
            gas_vel3 += delta_gas_vel3;

            gas_mom1 = gas_dens*gas_vel1;
            gas_mom2 = gas_dens*gas_vel2;
            gas_mom3 = gas_dens*gas_vel3;

            if (NON_BAROTROPIC_EOS && (!Isothermal_Flag)) {
              Real &gas_erg       = cons(IEN, k, j, i);
              Real gas_pre_0      = PoverRho(rad, phi, z)*gas_rho_0;
              Real delta_gas_pre  = (gas_pre_0 - gas_pre)*R_func(i)*inv_damping_tau(i)*dt;
              gas_pre            += delta_gas_pre;
              gas_erg             = gas_pre*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2)
                                                      + SQR(gas_mom3))*inv_dens_gas;
            }
          }
        }

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;
#pragma omp simd
            for (int i=is; i<=ie; ++i) {
              Real x1 = pmb->pcoord->x1v(i);
              Real rad, phi, z;
              GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
              if (rad <= x1max && rad >= radius_outer_damping) {
                Real cs_square = PoverRho(rad, phi, z);
                Real pre_diff  = (pslope + dslope)*cs_square;

                Real dust_rho_0    = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
                Real vel_dust_phi  = VelProfileCyl_dust(rad, phi, z);
                Real vel_K         = vK(pmb->porb, x1, x2, x3);
                Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
                vel_dust_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);

                Real dust_vel1_0  = vel_dust_r;
                Real dust_vel2_0  = 0.0;
                Real dust_vel3_0  = vel_dust_phi;

                Real &dust_dens    = cons_df(rho_id, k, j, i);
                Real &dust_mom1    = cons_df(v1_id,  k, j, i);
                Real &dust_mom2    = cons_df(v2_id,  k, j, i);
                Real &dust_mom3    = cons_df(v3_id,  k, j, i);
                Real inv_dens_dust = 1.0/dust_dens;

                Real dust_vel1 = dust_mom1*inv_dens_dust;
                Real dust_vel2 = dust_mom2*inv_dens_dust;
                Real dust_vel3 = dust_mom3*inv_dens_dust;

                Real delta_dust_dens = (dust_rho_0  - dust_dens)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
                Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

                dust_dens += delta_dust_dens;
                dust_vel1 += delta_dust_vel1;
                dust_vel2 += delta_dust_vel2;
                dust_vel3 += delta_dust_vel3;

                dust_mom1 = dust_dens*dust_vel1;
                dust_mom2 = dust_dens*dust_vel2;
                dust_mom3 = dust_dens*dust_vel3;
              }
            }
          }
        }
      }
    }
  }
  return;
}


void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  // Local Isothermal equation of state
  Real rad, phi, z;
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        const Real &gas_rho  = prim(IDN, k, j, i);
        const Real &gas_vel1 = prim(IM1, k, j, i);
        const Real &gas_vel2 = prim(IM2, k, j, i);
        const Real &gas_vel3 = prim(IM3, k, j, i);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real inv_gas_dens = 1.0/gas_dens;
        Real press        = PoverRho(rad, phi, z)*gas_dens;
        gas_erg           = press*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  // Local Isothermal equation of state
  Real rad, phi, z;
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real inv_beta  = 1.0/beta;
  Real igm1      = 1.0/(gamma_gas - 1.0);
  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &gas_pre = prim(IEN, k, j, i);
        Real &gas_dens      = cons(IDN, k, j, i);
        Real &gas_erg       = cons(IEN, k, j, i);

        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        Real omega_dyn  = std::sqrt(gm0/(rad*rad*rad));
        Real inv_t_cool = omega_dyn*inv_beta;
        Real cs2_init   = PoverRho(rad, phi, z);

        Real delta_erg  = (gas_pre*igm1 - gas_rho*cs2_init*igm1)*inv_t_cool*dt;
        gas_erg        -= delta_erg;
      }
    }
  }
  return;
}


void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad = pco->x1v(i);
    phi = pco->x2v(j);
    z   = pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi = pco->x3v(k);
    z   = pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! computes density in cylindrical coordinates

Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den = dentem;
  return std::max(den,dfloor);
}

Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  den         = dentem;
  return std::max(den,dffloor);
}

//----------------------------------------------------------------------------------------
//! computes pressure/density in cylindrical coordinates

Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}

//----------------------------------------------------------------------------------------
//! computes rotational velocity in cylindrical coordinates

Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_bump(const Real rad, const Real phi, const Real z, const Real diff) {
  Real vel = std::sqrt(gm0/rad + diff) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}

Real UserOrbitalVelocity(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return std::sqrt(porb->gm/x1)-porb->Omega0*x1;
}

Real UserOrbitalVelocity_r(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return -0.5*std::sqrt(porb->gm/x1)/x1-porb->Omega0;
}

Real UserOrbitalVelocity_z(OrbitalAdvection *porb, Real x1, Real x2, Real x3) {
  return 0.0;
}

int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
  Real maxeps = 0.0;
  Real rad(0.0), phi(0.0), z(0.0);
  Real max_rad = 0.0;
  Real min_rad = 3.0;
  int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; j++) {
    for (int i=pmb->is; i<=pmb->ie; i++) {
      GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
      Real epsr_g = (std::abs(w(IDN,k,j,i+1) - 2.0*w(IDN,k,j,i) + w(IDN,k,j,i-1))
                  + std::abs(w(IDN,k,j+1,i) - 2.0*w(IDN,k,j,i) + w(IDN,k,j-1,i)))
                  /w(IDN,k,j,i);

      Real epsp = (std::abs(w(IPR,k,j,i+1) - 2.0*w(IPR,k,j,i) + w(IPR,k,j,i-1))
                  + std::abs(w(IPR,k,j+1,i) - 2.0*w(IPR,k,j,i) + w(IPR,k,j-1,i)))
                  /w(IPR,k,j,i);

      Real epsr_d = 0;
      if (NDUSTFLUIDS > 0) {
        for (int n = 0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real epsr_temp = (std::abs(df_w(rho_id,k,j,i+1) - 2.0*df_w(rho_id,k,j,i) + df_w(rho_id,k,j,i-1))
                      + std::abs(df_w(rho_id,k,j+1,i) - 2.0*df_w(rho_id,k,j,i) + df_w(rho_id,k,j-1,i)))
                      /df_w(rho_id,k,j,i);

          epsr_d = std::max(epsr_d, epsr_temp);
        }
      }

      Real eps = std::max(std::max(epsr_g, epsr_d), epsp);
      maxeps   = std::max(maxeps, eps);
      max_rad  = std::max(max_rad, rad);
      min_rad  = std::min(min_rad, rad);
    }
  }
  // refine : curvature > 0.01
  if ((max_rad >= 0.5) && ( min_rad <= 2.8 ) && (maxeps > 0.01)) return 1;
  // derefinement: curvature < 0.005
  if ((max_rad < 0.5) || ( min_rad > 2.8 ) || (maxeps < 0.005)) return -1;
  // otherwise, stay
  return 0;
}

void Vr_interpolate_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
  //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}

} // namespace

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          //GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          Real &gas_vel3_ghost = prim(IM3, k, j, il-i);

          Real vis_vel_r = -1.5*(alpha_vis*cs_square/rad_ghost/omega_dyn);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = vel_gas_phi;
          gas_vel3_ghost = 0.0;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, il-i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

          //Real &gas_rho_active  = prim(IDN, k, j, il);
          //Real &gas_vel1_active = prim(IM1, k, j, il);
          //Real &gas_vel2_active = prim(IM2, k, j, il);
          //Real &gas_vel3_active = prim(IM3, k, j, il);
          //Real &gas_pres_active = prim(IEN, k, j, il);

          //Vr_interpolate_outer_nomatter(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
              //gas_vel1_active, gas_vel1_ghost);
          //gas_vel1_ghost = gas_vel1_active;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real &dust_rho_ghost  = prim_df(rho_id, k, j, il-i);
              Real &dust_vel1_ghost = prim_df(v1_id,  k, j, il-i);
              Real &dust_vel2_ghost = prim_df(v2_id,  k, j, il-i);
              Real &dust_vel3_ghost = prim_df(v3_id,  k, j, il-i);

              //Real &dust_rho_active  = prim_df(rho_id, k, j, il);
              //Real &dust_vel1_active = prim_df(v1_id,  k, j, il);
              //Real &dust_vel2_active = prim_df(v2_id,  k, j, il);
              //Real &dust_vel3_active = prim_df(v3_id,  k, j, il);

              //Vr_interpolate_outer_nomatter(rad_active, rad_ghost, dust_rho_active, dust_rho_ghost,
                  //dust_vel1_active, dust_vel1_ghost);
              //dust_vel1_ghost = dust_vel1_active;

              dust_rho_ghost    = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_r   = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
              Real vel_dust_phi = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vel_K;

              dust_vel1_ghost = vel_dust_r;
              dust_vel2_ghost = vel_dust_phi;
              dust_vel3_ghost = 0.0;
            }
          }
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vis_vel_r = -1.5*(alpha_vis*cs_square/rad_ghost/omega_dyn);
          Real vel_K     = vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          Real &gas_vel3_ghost = prim(IM3, k, j, il-i);

          //GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          //Real &gas_rho_active  = prim(IDN, k, j, il);
          //Real &gas_vel1_active = prim(IM1, k, j, il);
          //Real &gas_vel2_active = prim(IM2, k, j, il);
          //Real &gas_vel3_active = prim(IM3, k, j, il);
          //Real &gas_pres_active = prim(IEN, k, j, il);

          gas_rho_ghost    = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          //gas_vel1_ghost = gas_vel1_active;
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas_phi;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, il-i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real &dust_rho_ghost  = prim_df(rho_id, k, j, il-i);
              Real &dust_vel1_ghost = prim_df(v1_id,  k, j, il-i);
              Real &dust_vel2_ghost = prim_df(v2_id,  k, j, il-i);
              Real &dust_vel3_ghost = prim_df(v3_id,  k, j, il-i);

              //Real &dust_rho_active  = prim_df(rho_id, k, j, il);
              //Real &dust_vel1_active = prim_df(v1_id,  k, j, il);
              //Real &dust_vel2_active = prim_df(v2_id,  k, j, il);
              //Real &dust_vel3_active = prim_df(v3_id,  k, j, il);

              dust_rho_ghost    = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_r   = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
              Real vel_dust_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vel_K;

              dust_vel1_ghost = vel_dust_r;
              dust_vel2_ghost = 0.0;
              dust_vel3_ghost = vel_dust_phi;
            }
          }
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);

          Real vis_vel_r   = -1.5*(alpha_vis*cs_square/rad_ghost/omega_dyn);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = vel_gas_phi;
          gas_vel3_ghost = 0.0;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real &dust_rho_ghost  = prim_df(rho_id, k, j, iu+i);
              Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
              Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
              Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);

              dust_rho_ghost    = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_r   = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
              Real vel_dust_phi = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vel_K;

              dust_vel1_ghost = vel_dust_r;
              dust_vel2_ghost = vel_dust_phi;
              dust_vel3_ghost = 0.0;
            }
          }
        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          Real rad_ghost, phi_ghost, z_ghost;
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          Real cs_square = PoverRho(rad_ghost, phi_ghost, z_ghost);
          Real omega_dyn = std::sqrt(gm0/(rad_ghost*rad_ghost*rad_ghost));
          Real vis_vel_r = -1.5*(alpha_vis*cs_square/rad_ghost/omega_dyn);
          Real vel_K     = vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));
          Real pre_diff  = (pslope + dslope)*cs_square;

          Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);

          gas_rho_ghost    = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          gas_vel1_ghost = vis_vel_r;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas_phi;
          if (NON_BAROTROPIC_EOS) {
            Real &gas_pres_ghost = prim(IEN, k, j, iu+i);
            gas_pres_ghost       = cs_square*gas_rho_ghost;
          }

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real &dust_rho_ghost  = prim_df(rho_id, k, j, iu+i);
              Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
              Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
              Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);

              dust_rho_ghost    = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_r   = weight_dust[dust_id]*pre_diff/(2.0*vel_K);
              Real vel_dust_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vel_K;

              dust_vel1_ghost = vel_dust_r;
              dust_vel2_ghost = 0.0;
              dust_vel3_ghost = vel_dust_phi;
            }
          }
        }
      }
    }
  }
}


//void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  //FaceField &b, Real time, Real dt,
                  //int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  //Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  //Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  //OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    //for (int k=kl; k<=ku; ++k) {
      //for (int j=jl; j<=ju; ++j) {
        //for (int i=1; i<=ngh; ++i) {
          //GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          //GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          //Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          //Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          //Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          //Real &gas_vel3_ghost = prim(IM3, k, j, il-i);
          //Real &gas_pres_ghost = prim(IEN, k, j, il-i);

          //Real &gas_rho_active  = prim(IDN, k, j, il);
          //Real &gas_vel1_active = prim(IM1, k, j, il);
          //Real &gas_vel2_active = prim(IM2, k, j, il);
          //Real &gas_vel3_active = prim(IM3, k, j, il);
          //Real &gas_pres_active = prim(IEN, k, j, il);

          //gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          //Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          //if (pmb->porb->orbital_advection_defined)
            //vel_gas_phi -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

          //Vr_interpolate_nomatter(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
              //gas_vel1_active, gas_vel1_ghost);
          ////gas_vel1_ghost = gas_vel1_active;
          //gas_vel2_ghost = vel_gas_phi;
          //gas_vel3_ghost = 0.0;
          //if (NON_BAROTROPIC_EOS)
            //gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

          //if (NDUSTFLUIDS > 0) {
            //for (int n=0; n<NDUSTFLUIDS; ++n) {
              //int dust_id = n;
              //int rho_id  = 4*dust_id;
              //int v1_id   = rho_id + 1;
              //int v2_id   = rho_id + 2;
              //int v3_id   = rho_id + 3;

              //Real &dust_rho_ghost  = prim_df(rho_id, k, j, il-i);
              //Real &dust_vel1_ghost = prim_df(v1_id,  k, j, il-i);
              //Real &dust_vel2_ghost = prim_df(v2_id,  k, j, il-i);
              //Real &dust_vel3_ghost = prim_df(v3_id,  k, j, il-i);

              //Real &dust_rho_active  = prim_df(rho_id, k, j, il);
              //Real &dust_vel1_active = prim_df(v1_id,  k, j, il);
              //Real &dust_vel2_active = prim_df(v2_id,  k, j, il);
              //Real &dust_vel3_active = prim_df(v3_id,  k, j, il);

              //dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              //Real vel_dust_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              //if (pmb->porb->orbital_advection_defined)
                //vel_dust_phi -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

              //Vr_interpolate_nomatter(rad_active, rad_ghost, dust_rho_active, dust_rho_ghost,
                  //dust_vel1_active, dust_vel1_ghost);
              ////dust_vel1_ghost = dust_vel1_active;
              //dust_vel2_ghost = vel_dust_phi;
              //dust_vel3_ghost = 0.0;
            //}
          //}
        //}
      //}
    //}
  //} else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    //for (int k=kl; k<=ku; ++k) {
      //for (int j=jl; j<=ju; ++j) {
        //for (int i=1; i<=ngh; ++i) {
          //GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          //GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          //Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          //Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          //Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          //Real &gas_vel3_ghost = prim(IM3, k, j, il-i);
          //Real &gas_pres_ghost = prim(IEN, k, j, il-i);

          //Real &gas_rho_active  = prim(IDN, k, j, il);
          //Real &gas_vel1_active = prim(IM1, k, j, il);
          //Real &gas_vel2_active = prim(IM2, k, j, il);
          //Real &gas_vel3_active = prim(IM3, k, j, il);
          //Real &gas_pres_active = prim(IEN, k, j, il);

          //gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          //Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          //if (pmb->porb->orbital_advection_defined)
            //vel_gas_phi -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

          //gas_vel1_ghost = gas_vel1_active;
          //gas_vel2_ghost = 0.0;
          //gas_vel3_ghost = vel_gas_phi;
          //if (NON_BAROTROPIC_EOS)
            //gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

          //if (NDUSTFLUIDS > 0) {
            //for (int n=0; n<NDUSTFLUIDS; ++n) {
              //int dust_id = n;
              //int rho_id  = 4*dust_id;
              //int v1_id   = rho_id + 1;
              //int v2_id   = rho_id + 2;
              //int v3_id   = rho_id + 3;

              //Real &dust_rho_ghost  = prim_df(rho_id, k, j, il-i);
              //Real &dust_vel1_ghost = prim_df(v1_id,  k, j, il-i);
              //Real &dust_vel2_ghost = prim_df(v2_id,  k, j, il-i);
              //Real &dust_vel3_ghost = prim_df(v3_id,  k, j, il-i);

              //Real &dust_rho_active  = prim_df(rho_id, k, j, il);
              //Real &dust_vel1_active = prim_df(v1_id,  k, j, il);
              //Real &dust_vel2_active = prim_df(v2_id,  k, j, il);
              //Real &dust_vel3_active = prim_df(v3_id,  k, j, il);

              //dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              //Real vel_dust_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              //if (pmb->porb->orbital_advection_defined)
                //vel_dust_phi -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

              //dust_vel1_ghost = dust_vel1_active;
              //dust_vel2_ghost = 0.0;
              //dust_vel3_ghost = vel_dust_phi;
            //}
          //}
        //}
      //}
    //}
  //}
//}

////----------------------------------------------------------------------------------------
////! User-defined boundary Conditions: sets solution in ghost zones to initial values

//void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  //FaceField &b, Real time, Real dt,
                  //int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  //Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  //Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  //OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  //if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    //for (int k=kl; k<=ku; ++k) {
      //for (int j=jl; j<=ju; ++j) {
        //for (int i=1; i<=ngh; ++i) {
          //GetCylCoord(pco, rad_active, phi_active, z_active, iu,   j, k);
          //GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          //Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          //Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          //Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          //Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);
          //Real &gas_pres_ghost = prim(IEN, k, j, iu+i);

          //Real &gas_rho_active  = prim(IDN, k, j, iu);
          //Real &gas_vel1_active = prim(IM1, k, j, iu);
          //Real &gas_vel2_active = prim(IM2, k, j, iu);
          //Real &gas_vel3_active = prim(IM3, k, j, iu);
          //Real &gas_pres_active = prim(IEN, k, j, iu);

          //gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          //Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          //if (pmb->porb->orbital_advection_defined)
            //vel_gas_phi -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

          //Vr_interpolate_nomatter(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
              //gas_vel1_active, gas_vel1_ghost);
          ////gas_vel1_ghost = gas_vel1_active;
          //gas_vel2_ghost = vel_gas_phi;
          //gas_vel3_ghost = 0.0;
          //if (NON_BAROTROPIC_EOS)
            //gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

          //if (NDUSTFLUIDS > 0) {
            //for (int n=0; n<NDUSTFLUIDS; ++n) {
              //int dust_id = n;
              //int rho_id  = 4*dust_id;
              //int v1_id   = rho_id + 1;
              //int v2_id   = rho_id + 2;
              //int v3_id   = rho_id + 3;

              //Real &dust_rho_ghost  = prim_df(rho_id, k, j, iu+i);
              //Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
              //Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
              //Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);

              //Real &dust_rho_active  = prim_df(rho_id, k, j, iu);
              //Real &dust_vel1_active = prim_df(v1_id,  k, j, iu);
              //Real &dust_vel2_active = prim_df(v2_id,  k, j, iu);
              //Real &dust_vel3_active = prim_df(v3_id,  k, j, iu);

              //dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              //Real vel_dust_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              //if (pmb->porb->orbital_advection_defined)
                //vel_dust_phi -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

              //Vr_interpolate_nomatter(rad_active, rad_ghost, dust_rho_active, dust_rho_ghost,
                  //dust_vel1_active, dust_vel1_ghost);
              ////dust_vel1_ghost = dust_vel1_active;
              //dust_vel2_ghost = vel_dust_phi;
              //dust_vel3_ghost = 0.0;
            //}
          //}
        //}
      //}
    //}
  //} else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    //for (int k=kl; k<=ku; ++k) {
      //for (int j=jl; j<=ju; ++j) {
        //for (int i=1; i<=ngh; ++i) {
          //GetCylCoord(pco, rad_active, phi_active, z_active, iu,   j, k);
          //GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          //Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          //Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          //Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          //Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);
          //Real &gas_pres_ghost = prim(IEN, k, j, iu+i);

          //Real &gas_rho_active  = prim(IDN, k, j, iu);
          //Real &gas_vel1_active = prim(IM1, k, j, iu);
          //Real &gas_vel2_active = prim(IM2, k, j, iu);
          //Real &gas_vel3_active = prim(IM3, k, j, iu);
          //Real &gas_pres_active = prim(IEN, k, j, iu);

          //gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          //Real vel_gas_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          //if (pmb->porb->orbital_advection_defined)
            //vel_gas_phi -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

          //gas_vel1_ghost = gas_vel1_active;
          //gas_vel2_ghost = 0.0;
          //gas_vel3_ghost = vel_gas_phi;
          //if (NON_BAROTROPIC_EOS)
            //gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

          //if (NDUSTFLUIDS > 0) {
            //for (int n=0; n<NDUSTFLUIDS; ++n) {
              //int dust_id = n;
              //int rho_id  = 4*dust_id;
              //int v1_id   = rho_id + 1;
              //int v2_id   = rho_id + 2;
              //int v3_id   = rho_id + 3;

              //Real &dust_rho_ghost  = prim_df(rho_id, k, j, iu+i);
              //Real &dust_vel1_ghost = prim_df(v1_id,  k, j, iu+i);
              //Real &dust_vel2_ghost = prim_df(v2_id,  k, j, iu+i);
              //Real &dust_vel3_ghost = prim_df(v3_id,  k, j, iu+i);

              //Real &dust_rho_active  = prim_df(rho_id, k, j, iu);
              //Real &dust_vel1_active = prim_df(v1_id,  k, j, iu);
              //Real &dust_vel2_active = prim_df(v2_id,  k, j, iu);
              //Real &dust_vel3_active = prim_df(v3_id,  k, j, iu);

              //dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              //Real vel_dust_phi = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              //if (pmb->porb->orbital_advection_defined)
                //vel_dust_phi -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

              //dust_vel1_ghost = dust_vel1_active;
              //dust_vel2_ghost = 0.0;
              //dust_vel3_ghost = vel_dust_phi;
            //}
          //}
        //}
      //}
    //}
  //}
//}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, jl-j, k);
          prim(IDN, k, jl-j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1, k, jl-j, i) = 0.0;
          prim(IM2, k, jl-j, i) = vel_gas_phi;
          prim(IM3, k, jl-j, i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, k, jl-j, i) = PoverRho(rad, phi, z)*prim(IDN, k, jl-j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, jl-j, k);
              prim_df(rho_id, k, jl-j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(k));
              prim_df(v1_id, k, jl-j, i) = 0.0;
              prim_df(v2_id, k, jl-j, i) = vel_dust_phi;
              prim_df(v3_id, k, jl-j, i) = 0.0;
            }
          }

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, jl-j, k);
          prim(IDN, k, jl-j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
          prim(IM1, k, jl-j, i) = 0.0;
          prim(IM2, k, jl-j, i) = 0.0;
          prim(IM3, k, jl-j, i) = vel_gas_phi;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, k, jl-j, i) = PoverRho(rad, phi, z)*prim(IDN, k, jl-j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, jl-j, k);
              prim_df(rho_id, k, jl-j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(jl-j), pco->x3v(k));
              prim_df(v1_id, k, jl-j, i) = 0.0;
              prim_df(v2_id, k, jl-j, i) = 0.0;
              prim_df(v3_id, k, jl-j, i) = vel_dust_phi;
            }
          }

        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, ju+j, k);
          prim(IDN, k, ju+j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1, k, ju+j, i) = 0.0;
          prim(IM2, k, ju+j, i) = vel_gas_phi;
          prim(IM3, k, ju+j, i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, k, ju+j, i) = PoverRho(rad, phi, z)*prim(IDN, k, ju+j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, ju+j, k);
              prim_df(rho_id, k, ju+j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
              prim_df(v1_id, k, ju+j, i) = 0.0;
              prim_df(v2_id, k, ju+j, i) = vel_dust_phi;
              prim_df(v3_id, k, ju+j, i) = 0.0;
            }
          }

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, ju+j, k);
          prim(IDN, k, ju+j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
          prim(IM1, k, ju+j, i) = 0.0;
          prim(IM2, k, ju+j, i) = 0.0;
          prim(IM3, k, ju+j, i) = vel_gas_phi;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, k, ju+j, i) = PoverRho(rad, phi, z)*prim(IDN, k, ju+j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, ju+j, k);
              prim_df(rho_id, k, ju+j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(ju+j), pco->x3v(k));
              prim_df(v1_id, k, ju+j, i) = 0.0;
              prim_df(v2_id, k, ju+j, i) = 0.0;
              prim_df(v3_id, k, ju+j, i) = vel_dust_phi;
            }
          }

        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, kl-k);
          prim(IDN, kl-k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1, kl-k, j, i) = 0.0;
          prim(IM2, kl-k, j, i) = vel_gas_phi;
          prim(IM3, kl-k, j, i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, kl-k, j, i) = PoverRho(rad, phi, z)*prim(IDN, kl-k, j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, j, kl-k);
              prim_df(rho_id, kl-k, j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
              prim_df(v1_id, kl-k, j, i) = 0.0;
              prim_df(v2_id, kl-k, j, i) = vel_dust_phi;
              prim_df(v3_id, kl-k, j, i) = 0.0;
            }
          }

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, kl-k);
          prim(IDN, kl-k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
          prim(IM1, kl-k, j, i) = 0.0;
          prim(IM2, kl-k, j, i) = 0.0;
          prim(IM3, kl-k, j, i) = vel_gas_phi;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, kl-k, j, i) = PoverRho(rad, phi, z)*prim(IDN, kl-k, j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, j, kl-k);
              prim_df(rho_id, kl-k, j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(kl-k));
              prim_df(v1_id, kl-k, j, i) = 0.0;
              prim_df(v2_id, kl-k, j, i) = 0.0;
              prim_df(v3_id, kl-k, j, i) = vel_dust_phi;
            }
          }

        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad(0.0), phi(0.0), z(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, ku+k);
          prim(IDN, ku+k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1, ku+k, j, i) = 0.0;
          prim(IM2, ku+k, j, i) = vel_gas_phi;
          prim(IM3, ku+k, j, i) = 0.0;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, ku+k, j, i) = PoverRho(rad, phi, z)*prim(IDN, ku+k, j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, j, ku+k);
              prim_df(rho_id, ku+k, j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
              prim_df(v1_id, ku+k, j, i) = 0.0;
              prim_df(v2_id, ku+k, j, i) = vel_dust_phi;
              prim_df(v3_id, ku+k, j, i) = 0.0;
            }
          }

        }
      }
    }
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          GetCylCoord(pco, rad, phi, z, i, j, ku+k);
          prim(IDN, ku+k, j, i) = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
          prim(IM1, ku+k, j, i) = 0.0;
          prim(IM2, ku+k, j, i) = 0.0;
          prim(IM3, ku+k, j, i) = vel_gas_phi;
          if (NON_BAROTROPIC_EOS)
            prim(IEN, ku+k, j, i) = PoverRho(rad, phi, z)*prim(IDN, ku+k, j, i);

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              GetCylCoord(pco, rad, phi, z, i, j, ku+k);
              prim_df(rho_id, ku+k, j, i) = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_gas(rad, phi, z);
              if (pmb->porb->orbital_advection_defined)
                vel_dust_phi -= vK(pmb->porb, pco->x1v(i), pco->x2v(j), pco->x3v(ku+k));
              prim_df(v1_id, ku+k, j, i) = 0.0;
              prim_df(v2_id, ku+k, j, i) = 0.0;
              prim_df(v3_id, ku+k, j, i) = vel_dust_phi;
            }
          }

        }
      }
    }
  }
}
