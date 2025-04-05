//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file visc.cpp
//  iprob = 0 - test viscous shear flow density column in various coordinate systems
//  iprob = 1 - test viscous spreading of Keplerain ring

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt()
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
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

//#if NON_BAROTROPIC_EOS
//#error "This problem generator requires isothermal equation of state!"
//#endif

// problem parameters which are useful to make global to this file
namespace {
Real v0, t0, x0, d0, rho0, v1, v2, v3;
Real nuiso, iso_cs, amp;
Real A0, sig_x1, sig_x2, cen1, cen2, offset;
int iprob;
Real gm0, r0, dslope, p0_over_r0, pslope, gamma_gas, dfloor, dffloor, user_dt, iso_cs2_r0;
Real tau_relax, rs, gmp, rad_planet, phi_planet, t0pot, omega_p, Bump_flag, dwidth, rn, rand_amp, dust_dens_slope;
Real x1min, x1max, tau_damping, damping_rate;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region,
     inner_width_damping, outer_width_damping;
Real Omega0;
Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], nu_dust[NDUSTFLUIDS], Hratio[NDUSTFLUIDS];
bool Damping_Flag, Isothermal_Flag;

// User Sources
Real TotalMomentum1(MeshBlock *pmb, int iout);
Real TotalMomentum2(MeshBlock *pmb, int iout);
Real TotalMomentum3(MeshBlock *pmb, int iout);
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope);
void GasVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
void DustVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3);
void MySource(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void Linearinterpolate(const Real x_ac0, const Real x_ac1, const Real y_ac0, const Real y_ac1,
    const Real &x_gh, Real &y_gh);
Real MyTimeStep(MeshBlock *pmb);
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust_arr,
      AthenaArray<Real> &cs_dust_arr, int is, int ie, int js, int je, int ks, int ke);
} // namespace

void CartInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void CartOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void CartInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void CartOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

// User-defined boundary conditions for disk simulations
//void InnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
             //Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
//void OuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
             //Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  // Get parameters for gravitatonal potential of central point mass
  amp     = pin->GetOrAddReal("problem",    "amp",             0.01);
  iso_cs  = pin->GetOrAddReal("hydro",      "iso_sound_speed", 1e-1);
  iprob   = pin->GetOrAddInteger("problem", "iprob",           0);
  v1      = pin->GetOrAddReal("problem",    "v1",              0.001);
  v2      = pin->GetOrAddReal("problem",    "v2",              0.001);
  v3      = pin->GetOrAddReal("problem",    "v3",              0.001);
  x0      = pin->GetOrAddReal("problem",    "x0",              0.0);
  d0      = pin->GetOrAddReal("problem",    "d0",              1.0);
  t0      = pin->GetOrAddReal("problem",    "t0",              0.5);
  nuiso   = pin->GetOrAddReal("problem",    "nu_iso",          0.0);
  A0      = pin->GetOrAddReal("problem",    "A0",              1.0);
  rho0    = pin->GetOrAddReal("problem",    "rho0",            1.0);
  sig_x1  = pin->GetOrAddReal("problem",    "sigma_x1",        0.2);
  sig_x2  = pin->GetOrAddReal("problem",    "sigma_x2",        0.2);
  offset  = pin->GetOrAddReal("problem",    "offset",          0.0);
  user_dt = pin->GetOrAddReal("problem",    "user_dt",         0.0);

  cen1   = pin->GetOrAddReal("problem", "cen1", 0.5*(x1min + x1max));
  cen2   = pin->GetOrAddReal("problem", "cen2", PI);

  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "GM", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);

  // Get parameters for initial density and velocity
  dslope          = pin->GetOrAddReal("problem", "dslope", 0.0);
  dust_dens_slope = pin->GetOrAddReal("problem", "dust_dens_slope", 0.0);

  // The parameters of the amplitude of random perturbation on the radial velocity
  rand_amp        = pin->GetOrAddReal("problem", "random_vel_r_amp", 0.0);
  Damping_Flag    = pin->GetOrAddBoolean("problem", "Damping_Flag", true);
  Isothermal_Flag = pin->GetOrAddBoolean("problem", "Isothermal_Flag", true);

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 2.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, 2./3.);
  radius_outer_damping = x1max*pow(outer_ratio_region, -2./3.);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_inner_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  Omega0 = pin->GetOrAddReal("orbital_advection","Omega0",0.0);

  // Dust to gas ratio && dust stopping time
  for (int n=0; n<NDUSTFLUIDS; n++) {
    nu_dust[n] = pin->GetReal("dust", "nu_dust_" + std::to_string(n+1));
  }

  // Dust to gas ratio && dust stopping time
  if ((NDUSTFLUIDS > 0) && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0)) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "internal_density_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
    }
  }


  if ((std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) ||
      (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)) {
    // Get parameters of initial pressure and cooling parameters
    if (NON_BAROTROPIC_EOS) {
      p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
      pslope     = pin->GetReal("problem", "pslope");
      gamma_gas  = pin->GetReal("hydro", "gamma");
    } else {
      iso_cs2_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
    }
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor         = pin->GetOrAddReal("hydro", "dfloor", (1024*(float_min)));
  dffloor        = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));

  //EnrollDustDiffusivity(MyDustDiffusivity);

  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    // Enroll user-defined boundary condition
    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user"))
      EnrollUserBoundaryFunction(BoundaryFace::inner_x1, CartInnerX1);

    if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user"))
      EnrollUserBoundaryFunction(BoundaryFace::outer_x1, CartOuterX1);

    if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user"))
      EnrollUserBoundaryFunction(BoundaryFace::inner_x2, CartInnerX2);

    if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user"))
      EnrollUserBoundaryFunction(BoundaryFace::outer_x2, CartOuterX2);
  }

  if ((std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) ||
      (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)) {
    // Enroll damping zone and local isothermal equation of state
    EnrollUserExplicitSourceFunction(MySource);

    // Enroll user-defined boundary condition
    if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user"))
      EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);

    if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user"))
      EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }

  // Enroll user-defined time step
  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);

  AllocateUserHistoryOutput(3);
  EnrollUserHistoryOutput(0, TotalMomentum1, "tot-mom1");
  EnrollUserHistoryOutput(1, TotalMomentum2, "tot-mom2");
  EnrollUserHistoryOutput(2, TotalMomentum3, "tot-mom3");
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes viscous shear flow.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::int64_t iseed = -1 - gid;
  Real x1, x2, x3; // x2 and x3 are set but unused
  Real rad, z, phi, theta;

  const bool f2 = pmy_mesh->f2;
  const bool f3 = pmy_mesh->f3;

  if (iprob == 0) { //visc column
  //  Initialize density and momenta in Cartesian grids
    if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in dust_diffusion.cpp ProblemGenerator" << std::endl
          << "The test with iporb == 0 must be in cartesian coordinate!" << std::endl;
      ATHENA_ERROR(msg);
    }

    bool diffusion_corretion = (pdustfluids->dfdif.dustfluids_diffusion_defined);
    diffusion_corretion = (diffusion_corretion && (pdustfluids->dfdif.Momentum_Diffusion_Flag));

    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          x3 = pcoord->x3v(k);

          Real &gas_dens = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_dens = d0;
          gas_mom1 = gas_dens*v1;
          gas_mom2 = gas_dens*v2;
          gas_mom3 = gas_dens*v3;

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; n++) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

              if ((sig_x1 > 0.0) && (sig_x2 > 0.0)) { // 2D Gaussian
                Real gaussian_den   = A0*std::exp(-SQR(x1-x0)/(2.0*SQR(sig_x1))-SQR(x2-x0)/(2.0*SQR(sig_x2)));
                Real gaussian_vel_x = 0.0;
                Real gaussian_vel_y = 0.0;
                Real concentration  = gaussian_den + offset;
                if (diffusion_corretion) {
                  gaussian_vel_x = nu_dust[dust_id]*(x1-x0)/SQR(sig_x1)*gaussian_den;
                  gaussian_vel_y = nu_dust[dust_id]*(x2-x0)/SQR(sig_x2)*gaussian_den;
                  pdustfluids->dfccdif.diff_mom_cc(v1_id, k, j, i) = concentration*gas_dens*gaussian_vel_x;
                  pdustfluids->dfccdif.diff_mom_cc(v2_id, k, j, i) = concentration*gas_dens*gaussian_vel_y;
                }
                dust_dens = concentration*gas_dens;
                dust_mom1 = dust_dens*(v1 + gaussian_vel_x);
                dust_mom2 = dust_dens*(v2 + gaussian_vel_y);
                dust_mom3 = dust_dens*v3;
              } else if (sig_x1 > 0.0 ) {
                Real concentration = offset + A0*std::exp(-SQR(x1-x0)/(2.0*SQR(sig_x1)));
                Real gaussian_vel  = 0.0;
                if (diffusion_corretion) {
                  gaussian_vel = -A0*nu_dust[dust_id]*(x1-x0)/(SQR(sig_x1)*(A0 + offset*std::exp(SQR(x1-x0)/(2.0*SQR(sig_x1)))));
                  pdustfluids->dfccdif.diff_mom_cc(v1_id, k, j, i) = concentration*gas_dens*gaussian_vel;
                }
                dust_dens = concentration*gas_dens;
                dust_mom1 = dust_dens*(v1 + gaussian_vel);
                dust_mom2 = dust_dens*v2;
                dust_mom3 = dust_dens*v3;
              } else if (sig_x2 > 0.0) {
                Real concentration = offset + A0*std::exp(-SQR(x2-x0)/(2.0*SQR(sig_x2)));
                Real gaussian_vel  = 0.0;
                if (diffusion_corretion) {
                  gaussian_vel = A0*nu_dust[dust_id]*(x2-x0)/(SQR(sig_x2)*(A0 + offset*std::exp(SQR(x2-x0)/(2.0*SQR(sig_x2)))));
                  pdustfluids->dfccdif.diff_mom_cc(v2_id, k, j, i) = concentration*gas_dens*gaussian_vel;
                }
                dust_dens = concentration*gas_dens;
                dust_mom1 = dust_dens*v1;
                dust_mom2 = dust_dens*(v2 + gaussian_vel);
                dust_mom3 = dust_dens*v3;
              } else {
                Real concentration = offset + 1.0;
                dust_dens = concentration*gas_dens;
                dust_mom1 = dust_dens*v1;
                dust_mom2 = dust_dens*v2;
                dust_mom3 = dust_dens*v3;
              }

            }
          }
        }
      }
    }
  } else if (iprob == 1) { // dusty gaussian ring in cylindrical coordinate
    if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0 && gm0 == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in visc.cpp ProblemGenerator" << std::endl
          << "viscous ring test only compatible with cylindrical coord"
          << std::endl << "with point mass in center" << std::endl;
      ATHENA_ERROR(msg);
    }

    Real igm1 = 1.0/(gamma_gas - 1.0);
    Real rad(0.0), phi(0.0), z(0.0);
    Real dust_vel1(0.0), dust_vel2(0.0), dust_vel3(0.0);
    OrbitalVelocityFunc &vK = porb->OrbitalVelocity;
    for (int k=ks; k<=ke; ++k) {
      x3 = pcoord->x3v(k);
      for (int j=js; j<=je; ++j) {
        x2 = pcoord->x2v(j);
        for (int i=is; i<=ie; ++i) {
          x1 = pcoord->x1v(i);
          GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
          // compute initial conditions in cylindrical coordinates
          Real den_gas = DenProfileCyl_gas(rad, phi, z);

          Real cs2     = PoverRho(rad, phi, z);
          Real vel_gas = VelProfileCyl_gas(rad, phi, z);

          if (porb->orbital_advection_defined)
            vel_gas -= vK(porb, x1, x2, x3);

          Real &gas_dens = phydro->u(IDN, k, j, i);
          Real &gas_mom1 = phydro->u(IM1, k, j, i);
          Real &gas_mom2 = phydro->u(IM2, k, j, i);
          Real &gas_mom3 = phydro->u(IM3, k, j, i);

          gas_dens = den_gas;
          gas_mom1 = 0.0;

          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            gas_mom2 = den_gas*vel_gas;
            gas_mom3 = 0.0;
          }

          if (NON_BAROTROPIC_EOS) {
            Real p_over_r = PoverRho(rad, phi, z);
            phydro->u(IEN, k, j, i)  = p_over_r*phydro->u(IDN, k, j, i)*igm1;
            phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                          + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
          }

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              Real concentration = 0.0;
              if (sig_x2 > 0)
                concentration = A0*std::exp(-SQR(rad-cen1)/(2.0*SQR(sig_x1))-SQR(rad*(phi-cen2))/(2.0*SQR(sig_x2)));
              else
                concentration = A0*std::exp(-SQR(rad-cen1)/(2.0*SQR(sig_x1)));

              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
              Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
              Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
              Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

              dust_dens = (initial_D2G[dust_id] + concentration)*gas_dens;

              DustVelProfileCyl(rad, phi, z, dust_vel1, dust_vel2, dust_vel3);
              if (porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
                dust_vel2 -= vK(porb, x1, x2, x3);

              dust_mom1 = dust_dens * dust_vel1;
              dust_mom2 = dust_dens * dust_vel2;
              dust_mom3 = dust_dens * dust_vel3;
            }
          }

        }
      }
    }
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in dust_diffusion.cpp ProblemGenerator" << std::endl
        << "dust diffusion iprob has to be either 0 or 1" << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}


namespace {
void MySource(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  if (Damping_Flag) {
    InnerWavedamping(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
    OuterWavedamping(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  }

  if (Isothermal_Flag && NON_BAROTROPIC_EOS)
    LocalIsothermalEOS(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}

Real MyTimeStep(MeshBlock *pmb) {
  Real min_user_dt = user_dt;
  return min_user_dt;
}

void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust_arr,
      AthenaArray<Real> &cs_dust_arr, int is, int ie, int js, int je, int ks, int ke) {

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real &diffusivity = nu_dust_arr(dust_id, k, j, i);
          diffusivity       = nu_dust[dust_id];

          Real &soundspeed  = cs_dust_arr(dust_id, k, j, i);
          soundspeed        = std::sqrt(diffusivity);
        }
      }
    }
  }
  return;
}

void Linearinterpolate(const Real x_ac0, const Real x_ac1, const Real y_ac0, const Real y_ac1,
    const Real &x_gh, Real &y_gh) {
  y_gh = (x_gh-x_ac1)*y_ac0/(x_ac0-x_ac1) + (x_gh-x_ac0)*y_ac1/(x_ac1-x_ac0);
  return;
}


//----------------------------------------------------------------------------------------
// Wavedamping function
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real igm1           = 1.0/(gamma_gas - 1.0);
  Real inv_inner_damp = 1.0/inner_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        Real rad, phi, z;
        // compute initial conditions in cylindrical coordinates
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        if (rad >= x1min && rad < radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
          Real R_func      = SQR((rad - radius_inner_damping)*inv_inner_damp);
          Real inv_damping_tau = 1.0/(damping_rate*omega_dyn);

          Real gas_rho_0 = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_0 = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_0 -= vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0, gas_vel2_0, gas_vel3_0;
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            gas_vel1_0 = 0.0;
            gas_vel2_0 = vel_gas_0;
            gas_vel3_0 = 0.0;
          } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            gas_vel1_0 = 0.0;
            gas_vel2_0 = 0.0;
            gas_vel3_0 = vel_gas_0;
          }

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real inv_den_gas_ori = 1.0/gas_dens;

          Real gas_vel1 = gas_mom1*inv_den_gas_ori;
          Real gas_vel2 = gas_mom2*inv_den_gas_ori;
          Real gas_vel3 = gas_mom3*inv_den_gas_ori;

          Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func*inv_damping_tau*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func*inv_damping_tau*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func*inv_damping_tau*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func*inv_damping_tau*dt;

          gas_dens += delta_gas_dens;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;

          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          //if (NON_BAROTROPIC_EOS) {
            //Real &gas_erg       = cons(IEN, k, j, i);
            //Real gas_erg_0      = PoverRho(rad, phi, z)*gas_rho_0;
            //gas_erg_0          += 0.5*gas_rho_0*(SQR(gas_vel1_0) + SQR(gas_vel2_0) + SQR(gas_vel3_0));
            //Real delta_gas_erg  = (gas_erg_0 - gas_erg)*R_func*inv_damping_tau*dt;
            //gas_erg            += delta_gas_erg;
          //}

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real dust_rho_0 = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_0 = VelProfileCyl_dust(rad, phi, z);
              Real dust_vel1_0, dust_vel2_0, dust_vel3_0;
              if (pmb->porb->orbital_advection_defined)
                vel_dust_0 -= vK(pmb->porb, x1, x2, x3);

              if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
                dust_vel1_0  = 0.0;
                dust_vel2_0  = vel_dust_0;
                dust_vel3_0  = 0.0;
              } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
                dust_vel1_0  = 0.0;
                dust_vel2_0  = 0.0;
                dust_vel3_0  = vel_dust_0;
              }

              Real &dust_dens = cons_df(rho_id, k, j, i);
              Real &dust_mom1 = cons_df(v1_id, k, j, i);
              Real &dust_mom2 = cons_df(v2_id, k, j, i);
              Real &dust_mom3 = cons_df(v3_id, k, j, i);
              Real inv_den_dust_ori = 1.0/dust_dens;

              Real dust_vel1 = dust_mom1*inv_den_dust_ori;
              Real dust_vel2 = dust_mom2*inv_den_dust_ori;
              Real dust_vel3 = dust_mom3*inv_den_dust_ori;

              Real delta_dust_dens = (dust_rho_0  - dust_dens)*R_func*inv_damping_tau*dt;
              Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func*inv_damping_tau*dt;
              Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func*inv_damping_tau*dt;
              Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func*inv_damping_tau*dt;

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
  return;
}


void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {


  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real igm1           = 1.0/(gamma_gas - 1.0);
  Real inv_outer_damp = 1.0/outer_width_damping;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  for (int k=ks; k<=ke; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
    Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        if (rad <= x1max && rad >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
          Real R_func      = SQR((rad - radius_outer_damping)*inv_outer_damp);
          Real inv_damping_tau = 1.0/(damping_rate*omega_dyn);

          Real gas_rho_0 = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_0 = VelProfileCyl_gas(rad, phi, z);
          if (pmb->porb->orbital_advection_defined)
            vel_gas_0 -= vK(pmb->porb, x1, x2, x3);

          Real gas_vel1_0, gas_vel2_0, gas_vel3_0;
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            gas_vel1_0 = 0.0;
            gas_vel2_0 = vel_gas_0;
            gas_vel3_0 = 0.0;
          } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            gas_vel1_0 = 0.0;
            gas_vel2_0 = 0.0;
            gas_vel3_0 = vel_gas_0;
          }

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);
          Real inv_den_gas_ori = 1.0/gas_dens;

          Real gas_vel1 = gas_mom1*inv_den_gas_ori;
          Real gas_vel2 = gas_mom2*inv_den_gas_ori;
          Real gas_vel3 = gas_mom3*inv_den_gas_ori;

          Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func*inv_damping_tau*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func*inv_damping_tau*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func*inv_damping_tau*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func*inv_damping_tau*dt;

          gas_dens += delta_gas_dens;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;

          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          //if (NON_BAROTROPIC_EOS) {
            //Real &gas_erg       = cons(IEN, k, j, i);
            //Real gas_erg_0      = PoverRho(rad, phi, z)*gas_rho_0;
            //gas_erg_0          += 0.5*gas_rho_0*(SQR(gas_vel1_0) + SQR(gas_vel2_0) + SQR(gas_vel3_0));
            //Real delta_gas_erg  = (gas_erg_0 - gas_erg)*R_func*inv_damping_tau*dt;
            //gas_erg            += delta_gas_erg;
          //}

          if (NDUSTFLUIDS > 0) {
            for (int n=0; n<NDUSTFLUIDS; ++n) {
              int dust_id = n;
              int rho_id  = 4*dust_id;
              int v1_id   = rho_id + 1;
              int v2_id   = rho_id + 2;
              int v3_id   = rho_id + 3;

              Real dust_rho_0 = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_0 = VelProfileCyl_dust(rad, phi, z);
              Real dust_vel1_0, dust_vel2_0, dust_vel3_0;
              if (pmb->porb->orbital_advection_defined)
                vel_dust_0 -= vK(pmb->porb, x1, x2, x3);

              if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
                dust_vel1_0  = 0.0;
                dust_vel2_0  = vel_dust_0;
                dust_vel3_0  = 0.0;
              } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
                dust_vel1_0  = 0.0;
                dust_vel2_0  = 0.0;
                dust_vel3_0  = vel_dust_0;
              }

              Real &dust_dens = cons_df(rho_id, k, j, i);
              Real &dust_mom1 = cons_df(v1_id, k, j, i);
              Real &dust_mom2 = cons_df(v2_id, k, j, i);
              Real &dust_mom3 = cons_df(v3_id, k, j, i);
              Real inv_den_dust_ori = 1.0/dust_dens;

              Real dust_vel1 = dust_mom1*inv_den_dust_ori;
              Real dust_vel2 = dust_mom2*inv_den_dust_ori;
              Real dust_vel3 = dust_mom3*inv_den_dust_ori;

              Real delta_dust_dens = (dust_rho_0  - dust_dens) *R_func*inv_damping_tau*dt;
              Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func*inv_damping_tau*dt;
              Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func*inv_damping_tau*dt;
              Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func*inv_damping_tau*dt;

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
  return;
}


void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt,
    const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
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


//----------------------------------------------------------------------------------------
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad = pco->x1v(i);
    phi = pco->x2v(j);
    z   = pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi = pco->x3v(i);
    z   = pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \f  computes density in cylindrical coordinates
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope) {
  Real dust2gas = initial_dust2gas*std::pow(rad/r0, slope);
  return dust2gas;
}


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


Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}


Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad+z*z);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}

Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z) {
  Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}

void Vr_constMdot(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
  //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}


//----------------------------------------------------------------------------------------
//! \f  computes rotational velocity in cylindrical coordinates
void GasVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real iso_cs2 = PoverRho(rad, phi, z);
  Real vel     = (dslope+pslope)*iso_cs2/(gm0/rad) + 1.0;
  vel          = std::sqrt(gm0/rad)*std::sqrt(vel);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = 0.0;
    v2 = vel;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = 0.0;
    v2 = 0.0;
    v3 = vel;
  }
  return;
}

void DustVelProfileCyl(const Real rad, const Real phi, const Real z,
                   Real &v1, Real &v2, Real &v3) {
  Real vel = std::sqrt(gm0/rad);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = 0.0;
    v2 = vel;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = 0.0;
    v2 = 0.0;
    v3 = vel;
  }
  return;
}


Real TotalMomentum1(MeshBlock *pmb, int iout) {
  Real total_mom = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &s = pmb->pdustfluids->df_u;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i=is; i<=ie; i++) {
        total_mom += volume(i)*u(IM1,k,j,i);
      }
    }
  }

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i=is; i<=ie; i++) {
          total_mom += volume(i)*s(v1_id,k,j,i);
        }
      }
    }
  }

  return total_mom;
}


Real TotalMomentum2(MeshBlock *pmb, int iout) {
  Real total_mom = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &s = pmb->pdustfluids->df_u;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i=is; i<=ie; i++) {
        total_mom += volume(i)*u(IM2,k,j,i);
      }
    }
  }

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i=is; i<=ie; i++) {
          total_mom += volume(i)*s(v2_id,k,j,i);
        }
      }
    }
  }

  return total_mom;
}


Real TotalMomentum3(MeshBlock *pmb, int iout) {
  Real total_mom = 0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &s = pmb->pdustfluids->df_u;
  AthenaArray<Real> &u = pmb->phydro->u;
  AthenaArray<Real> volume; // 1D array of volumes
  // allocate 1D array for cell volume used in usr def history
  volume.NewAthenaArray(pmb->ncells1);

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i=is; i<=ie; i++) {
        total_mom += volume(i)*u(IM3,k,j,i);
      }
    }
  }

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
        for (int i=is; i<=ie; i++) {
          total_mom += volume(i)*s(v3_id,k,j,i);
        }
      }
    }
  }

  return total_mom;
}
} // namespace


void CartInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {

        Real &x1_gh  = pco->x1v(il-i);
        Real &x1_ac0 = pco->x1v(il);
        Real &x1_ac1 = pco->x1v(il+1);

        Real &gas_rho_gh  = prim(IDN, k, j, il-i);
        Real &gas_vel1_gh = prim(IM1, k, j, il-i);
        Real &gas_vel2_gh = prim(IM2, k, j, il-i);
        Real &gas_vel3_gh = prim(IM3, k, j, il-i);

        Real &gas_rho_ac0  = prim(IDN, k, j, il);
        Real &gas_vel1_ac0 = prim(IM1, k, j, il);
        Real &gas_vel2_ac0 = prim(IM2, k, j, il);
        Real &gas_vel3_ac0 = prim(IM3, k, j, il);

        Real &gas_rho_ac1  = prim(IDN, k, j, il+1);
        Real &gas_vel1_ac1 = prim(IM1, k, j, il+1);
        Real &gas_vel2_ac1 = prim(IM2, k, j, il+1);
        Real &gas_vel3_ac1 = prim(IM3, k, j, il+1);

        Linearinterpolate(x1_ac0, x1_ac1, gas_rho_ac0,  gas_rho_ac1,  x1_gh, gas_rho_gh);
        Linearinterpolate(x1_ac0, x1_ac1, gas_vel1_ac0, gas_vel1_ac1, x1_gh, gas_vel1_gh);
        Linearinterpolate(x1_ac0, x1_ac1, gas_vel2_ac0, gas_vel2_ac1, x1_gh, gas_vel2_gh);
        Linearinterpolate(x1_ac0, x1_ac1, gas_vel3_ac0, gas_vel3_ac1, x1_gh, gas_vel3_gh);

        if (gas_rho_gh < 0.0) gas_rho_gh = 0.0;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, j, il-i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, j, il-i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, j, il-i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, j, il-i);

            Real &dust_rho_ac0  = prim_df(rho_id, k, j, il);
            Real &dust_vel1_ac0 = prim_df(v1_id,  k, j, il);
            Real &dust_vel2_ac0 = prim_df(v2_id,  k, j, il);
            Real &dust_vel3_ac0 = prim_df(v3_id,  k, j, il);

            Real &dust_rho_ac1  = prim_df(rho_id, k, j, il+1);
            Real &dust_vel1_ac1 = prim_df(v1_id,  k, j, il+1);
            Real &dust_vel2_ac1 = prim_df(v2_id,  k, j, il+1);
            Real &dust_vel3_ac1 = prim_df(v3_id,  k, j, il+1);

            Linearinterpolate(x1_ac0, x1_ac1, dust_rho_ac0,  dust_rho_ac1,  x1_gh, dust_rho_gh);
            Linearinterpolate(x1_ac0, x1_ac1, dust_vel1_ac0, dust_vel1_ac1, x1_gh, dust_vel1_gh);
            Linearinterpolate(x1_ac0, x1_ac1, dust_vel2_ac0, dust_vel2_ac1, x1_gh, dust_vel2_gh);
            Linearinterpolate(x1_ac0, x1_ac1, dust_vel3_ac0, dust_vel3_ac1, x1_gh, dust_vel3_gh);

            if (dust_rho_gh < 0.0) dust_rho_gh = dffloor;
          }
        }
      }
    }
  }
  return;
}


void CartOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {

        Real &x1_gh  = pco->x1v(iu+i);
        Real &x1_ac0 = pco->x1v(iu);
        Real &x1_ac1 = pco->x1v(iu-1);

        Real &gas_rho_gh  = prim(IDN, k, j, iu+i);
        Real &gas_vel1_gh = prim(IM1, k, j, iu+i);
        Real &gas_vel2_gh = prim(IM2, k, j, iu+i);
        Real &gas_vel3_gh = prim(IM3, k, j, iu+i);

        Real &gas_rho_ac0  = prim(IDN, k, j, iu);
        Real &gas_vel1_ac0 = prim(IM1, k, j, iu);
        Real &gas_vel2_ac0 = prim(IM2, k, j, iu);
        Real &gas_vel3_ac0 = prim(IM3, k, j, iu);

        Real &gas_rho_ac1  = prim(IDN, k, j, iu-1);
        Real &gas_vel1_ac1 = prim(IM1, k, j, iu-1);
        Real &gas_vel2_ac1 = prim(IM2, k, j, iu-1);
        Real &gas_vel3_ac1 = prim(IM3, k, j, iu-1);

        Linearinterpolate(x1_ac0, x1_ac1, gas_rho_ac0,  gas_rho_ac1,  x1_gh, gas_rho_gh);
        Linearinterpolate(x1_ac0, x1_ac1, gas_vel1_ac0, gas_vel1_ac1, x1_gh, gas_vel1_gh);
        Linearinterpolate(x1_ac0, x1_ac1, gas_vel2_ac0, gas_vel2_ac1, x1_gh, gas_vel2_gh);
        Linearinterpolate(x1_ac0, x1_ac1, gas_vel3_ac0, gas_vel3_ac1, x1_gh, gas_vel3_gh);

        if (gas_rho_gh < 0.0) gas_rho_gh = 0.0;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, j, iu+i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, j, iu+i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, j, iu+i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, j, iu+i);

            Real &dust_rho_ac0  = prim_df(rho_id, k, j, iu);
            Real &dust_vel1_ac0 = prim_df(v1_id,  k, j, iu);
            Real &dust_vel2_ac0 = prim_df(v2_id,  k, j, iu);
            Real &dust_vel3_ac0 = prim_df(v3_id,  k, j, iu);

            Real &dust_rho_ac1  = prim_df(rho_id, k, j, iu-1);
            Real &dust_vel1_ac1 = prim_df(v1_id,  k, j, iu-1);
            Real &dust_vel2_ac1 = prim_df(v2_id,  k, j, iu-1);
            Real &dust_vel3_ac1 = prim_df(v3_id,  k, j, iu-1);

            Linearinterpolate(x1_ac0, x1_ac1, dust_rho_ac0,  dust_rho_ac1,  x1_gh, dust_rho_gh);
            Linearinterpolate(x1_ac0, x1_ac1, dust_vel1_ac0, dust_vel1_ac1, x1_gh, dust_vel1_gh);
            Linearinterpolate(x1_ac0, x1_ac1, dust_vel2_ac0, dust_vel2_ac1, x1_gh, dust_vel2_gh);
            Linearinterpolate(x1_ac0, x1_ac1, dust_vel3_ac0, dust_vel3_ac1, x1_gh, dust_vel3_gh);

            if (dust_rho_gh < 0.0) dust_rho_gh = dffloor;
          }
        }
      }
    }
  }
  return;
}


void CartInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {

        Real &x2_gh  = pco->x2v(jl-j);
        Real &x2_ac0 = pco->x2v(jl);
        Real &x2_ac1 = pco->x2v(jl+1);

        Real &gas_rho_gh  = prim(IDN, k, jl-j, i);
        Real &gas_vel1_gh = prim(IM1, k, jl-j, i);
        Real &gas_vel2_gh = prim(IM2, k, jl-j, i);
        Real &gas_vel3_gh = prim(IM3, k, jl-j, i);

        Real &gas_rho_ac0  = prim(IDN, k, jl, i);
        Real &gas_vel1_ac0 = prim(IM1, k, jl, i);
        Real &gas_vel2_ac0 = prim(IM2, k, jl, i);
        Real &gas_vel3_ac0 = prim(IM3, k, jl, i);

        Real &gas_rho_ac1  = prim(IDN, k, jl+1, i);
        Real &gas_vel1_ac1 = prim(IM1, k, jl+1, i);
        Real &gas_vel2_ac1 = prim(IM2, k, jl+1, i);
        Real &gas_vel3_ac1 = prim(IM3, k, jl+1, i);

        Linearinterpolate(x2_ac0, x2_ac1, gas_rho_ac0,  gas_rho_ac1,  x2_gh, gas_rho_gh);
        Linearinterpolate(x2_ac0, x2_ac1, gas_vel1_ac0, gas_vel1_ac1, x2_gh, gas_vel1_gh);
        Linearinterpolate(x2_ac0, x2_ac1, gas_vel2_ac0, gas_vel2_ac1, x2_gh, gas_vel2_gh);
        Linearinterpolate(x2_ac0, x2_ac1, gas_vel3_ac0, gas_vel3_ac1, x2_gh, gas_vel3_gh);

        if (gas_rho_gh < 0.0) gas_rho_gh = 0.0;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, jl-j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, jl-j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, jl-j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, jl-j, i);

            Real &dust_rho_ac0  = prim_df(rho_id, k, jl, i);
            Real &dust_vel1_ac0 = prim_df(v1_id,  k, jl, i);
            Real &dust_vel2_ac0 = prim_df(v2_id,  k, jl, i);
            Real &dust_vel3_ac0 = prim_df(v3_id,  k, jl, i);

            Real &dust_rho_ac1  = prim_df(rho_id, k, jl+1, i);
            Real &dust_vel1_ac1 = prim_df(v1_id,  k, jl+1, i);
            Real &dust_vel2_ac1 = prim_df(v2_id,  k, jl+1, i);
            Real &dust_vel3_ac1 = prim_df(v3_id,  k, jl+1, i);

            Linearinterpolate(x2_ac0, x2_ac1, dust_rho_ac0,  dust_rho_ac1,  x2_gh, dust_rho_gh);
            Linearinterpolate(x2_ac0, x2_ac1, dust_vel1_ac0, dust_vel1_ac1, x2_gh, dust_vel1_gh);
            Linearinterpolate(x2_ac0, x2_ac1, dust_vel2_ac0, dust_vel2_ac1, x2_gh, dust_vel2_gh);
            Linearinterpolate(x2_ac0, x2_ac1, dust_vel3_ac0, dust_vel3_ac1, x2_gh, dust_vel3_gh);

            if (dust_rho_gh < 0.0) dust_rho_gh = dffloor;
          }
        }
      }
    }
  }
  return;
}


void CartOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=il; i<=iu; ++i) {

        Real &x2_gh  = pco->x2v(ju+j);
        Real &x2_ac0 = pco->x2v(ju);
        Real &x2_ac1 = pco->x2v(ju-1);

        Real &gas_rho_gh  = prim(IDN, k, ju+j, i);
        Real &gas_vel1_gh = prim(IM1, k, ju+j, i);
        Real &gas_vel2_gh = prim(IM2, k, ju+j, i);
        Real &gas_vel3_gh = prim(IM3, k, ju+j, i);

        Real &gas_rho_ac0  = prim(IDN, k, ju, i);
        Real &gas_vel1_ac0 = prim(IM1, k, ju, i);
        Real &gas_vel2_ac0 = prim(IM2, k, ju, i);
        Real &gas_vel3_ac0 = prim(IM3, k, ju, i);

        Real &gas_rho_ac1  = prim(IDN, k, ju-1, i);
        Real &gas_vel1_ac1 = prim(IM1, k, ju-1, i);
        Real &gas_vel2_ac1 = prim(IM2, k, ju-1, i);
        Real &gas_vel3_ac1 = prim(IM3, k, ju-1, i);

        Linearinterpolate(x2_ac0, x2_ac1, gas_rho_ac0,  gas_rho_ac1,  x2_gh, gas_rho_gh);
        Linearinterpolate(x2_ac0, x2_ac1, gas_vel1_ac0, gas_vel1_ac1, x2_gh, gas_vel1_gh);
        Linearinterpolate(x2_ac0, x2_ac1, gas_vel2_ac0, gas_vel2_ac1, x2_gh, gas_vel2_gh);
        Linearinterpolate(x2_ac0, x2_ac1, gas_vel3_ac0, gas_vel3_ac1, x2_gh, gas_vel3_gh);

        if (gas_rho_gh < 0.0) gas_rho_gh = 0.0;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_rho_gh  = prim_df(rho_id, k, ju+j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, ju+j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, ju+j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, ju+j, i);

            Real &dust_rho_ac0  = prim_df(rho_id, k, ju, i);
            Real &dust_vel1_ac0 = prim_df(v1_id,  k, ju, i);
            Real &dust_vel2_ac0 = prim_df(v2_id,  k, ju, i);
            Real &dust_vel3_ac0 = prim_df(v3_id,  k, ju, i);

            Real &dust_rho_ac1  = prim_df(rho_id, k, ju-1, i);
            Real &dust_vel1_ac1 = prim_df(v1_id,  k, ju-1, i);
            Real &dust_vel2_ac1 = prim_df(v2_id,  k, ju-1, i);
            Real &dust_vel3_ac1 = prim_df(v3_id,  k, ju-1, i);

            Linearinterpolate(x2_ac0, x2_ac1, dust_rho_ac0,  dust_rho_ac1,  x2_gh, dust_rho_gh);
            Linearinterpolate(x2_ac0, x2_ac1, dust_vel1_ac0, dust_vel1_ac1, x2_gh, dust_vel1_gh);
            Linearinterpolate(x2_ac0, x2_ac1, dust_vel2_ac0, dust_vel2_ac1, x2_gh, dust_vel2_gh);
            Linearinterpolate(x2_ac0, x2_ac1, dust_vel3_ac0, dust_vel3_ac1, x2_gh, dust_vel3_gh);

            if (dust_rho_gh < 0.0) dust_rho_gh = dffloor;
          }
        }
      }
    }
  }
  return;
}


//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          Real &gas_vel3_ghost = prim(IM3, k, j, il-i);
          Real &gas_pres_ghost = prim(IEN, k, j, il-i);

          Real &gas_rho_active  = prim(IDN, k, j, il);
          Real &gas_vel1_active = prim(IM1, k, j, il);
          Real &gas_vel2_active = prim(IM2, k, j, il);
          Real &gas_vel3_active = prim(IM3, k, j, il);
          Real &gas_pres_active = prim(IEN, k, j, il);

          gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          Real vel_gas = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

          Vr_constMdot(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
              gas_vel1_active, gas_vel1_ghost);
          //gas_vel1_ghost = gas_vel1_active;
          gas_vel2_ghost = vel_gas;
          gas_vel3_ghost = 0.0;
          if (NON_BAROTROPIC_EOS)
            gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

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

              Real &dust_rho_active  = prim_df(rho_id, k, j, il);
              Real &dust_vel1_active = prim_df(v1_id,  k, j, il);
              Real &dust_vel2_active = prim_df(v2_id,  k, j, il);
              Real &dust_vel3_active = prim_df(v3_id,  k, j, il);

              dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              Real vel_dust = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

              Vr_constMdot(rad_active, rad_ghost, dust_rho_active, dust_rho_ghost,
                  dust_vel1_active, dust_vel1_ghost);
              //dust_vel1_ghost = dust_vel1_active;
              dust_vel2_ghost = vel_dust;
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
          GetCylCoord(pco, rad_active, phi_active, z_active, il,   j, k);
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  il-i, j, k);

          Real &gas_rho_ghost  = prim(IDN, k, j, il-i);
          Real &gas_vel1_ghost = prim(IM1, k, j, il-i);
          Real &gas_vel2_ghost = prim(IM2, k, j, il-i);
          Real &gas_vel3_ghost = prim(IM3, k, j, il-i);
          Real &gas_pres_ghost = prim(IEN, k, j, il-i);

          Real &gas_rho_active  = prim(IDN, k, j, il);
          Real &gas_vel1_active = prim(IM1, k, j, il);
          Real &gas_vel2_active = prim(IM2, k, j, il);
          Real &gas_vel3_active = prim(IM3, k, j, il);
          Real &gas_pres_active = prim(IEN, k, j, il);

          gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          Real vel_gas = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

          gas_vel1_ghost = gas_vel1_active;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas;
          if (NON_BAROTROPIC_EOS)
            gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

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

              Real &dust_rho_active  = prim_df(rho_id, k, j, il);
              Real &dust_vel1_active = prim_df(v1_id,  k, j, il);
              Real &dust_vel2_active = prim_df(v2_id,  k, j, il);
              Real &dust_vel3_active = prim_df(v3_id,  k, j, il);

              dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              Real vel_dust = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust -= vK(pmb->porb, pco->x1v(il-i), pco->x2v(j), pco->x3v(k));

              dust_vel1_ghost = dust_vel1_active;
              dust_vel2_ghost = 0.0;
              dust_vel3_ghost = vel_dust;
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
  Real rad_active(0.0), phi_active(0.0), z_active(0.0);
  Real rad_ghost(0.0),  phi_ghost(0.0),  z_ghost(0.0);
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=1; i<=ngh; ++i) {
          GetCylCoord(pco, rad_active, phi_active, z_active, iu,   j, k);
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);
          Real &gas_pres_ghost = prim(IEN, k, j, iu+i);

          Real &gas_rho_active  = prim(IDN, k, j, iu);
          Real &gas_vel1_active = prim(IM1, k, j, iu);
          Real &gas_vel2_active = prim(IM2, k, j, iu);
          Real &gas_vel3_active = prim(IM3, k, j, iu);
          Real &gas_pres_active = prim(IEN, k, j, iu);

          gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          Real vel_gas = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

          Vr_constMdot(rad_active, rad_ghost, gas_rho_active, gas_rho_ghost,
              gas_vel1_active, gas_vel1_ghost);
          //gas_vel1_ghost = gas_vel1_active;
          gas_vel2_ghost = vel_gas;
          gas_vel3_ghost = 0.0;
          if (NON_BAROTROPIC_EOS)
            gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

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

              Real &dust_rho_active  = prim_df(rho_id, k, j, iu);
              Real &dust_vel1_active = prim_df(v1_id,  k, j, iu);
              Real &dust_vel2_active = prim_df(v2_id,  k, j, iu);
              Real &dust_vel3_active = prim_df(v3_id,  k, j, iu);

              dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              Real vel_dust = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

              Vr_constMdot(rad_active, rad_ghost, dust_rho_active, dust_rho_ghost,
                  dust_vel1_active, dust_vel1_ghost);
              //dust_vel1_ghost = dust_vel1_active;
              dust_vel2_ghost = vel_dust;
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
          GetCylCoord(pco, rad_active, phi_active, z_active, iu,   j, k);
          GetCylCoord(pco, rad_ghost,  phi_ghost,  z_ghost,  iu+i, j, k);

          Real &gas_rho_ghost  = prim(IDN, k, j, iu+i);
          Real &gas_vel1_ghost = prim(IM1, k, j, iu+i);
          Real &gas_vel2_ghost = prim(IM2, k, j, iu+i);
          Real &gas_vel3_ghost = prim(IM3, k, j, iu+i);
          Real &gas_pres_ghost = prim(IEN, k, j, iu+i);

          Real &gas_rho_active  = prim(IDN, k, j, iu);
          Real &gas_vel1_active = prim(IM1, k, j, iu);
          Real &gas_vel2_active = prim(IM2, k, j, iu);
          Real &gas_vel3_active = prim(IM3, k, j, iu);
          Real &gas_pres_active = prim(IEN, k, j, iu);

          gas_rho_ghost = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);

          Real vel_gas = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
          if (pmb->porb->orbital_advection_defined)
            vel_gas -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

          gas_vel1_ghost = gas_vel1_active;
          gas_vel2_ghost = 0.0;
          gas_vel3_ghost = vel_gas;
          if (NON_BAROTROPIC_EOS)
            gas_pres_ghost = PoverRho(rad_ghost, phi_ghost, z_ghost)*gas_rho_ghost;

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

              Real &dust_rho_active  = prim_df(rho_id, k, j, iu);
              Real &dust_vel1_active = prim_df(v1_id,  k, j, iu);
              Real &dust_vel2_active = prim_df(v2_id,  k, j, iu);
              Real &dust_vel3_active = prim_df(v3_id,  k, j, iu);

              dust_rho_ghost = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);

              Real vel_dust = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
              if (pmb->porb->orbital_advection_defined)
                vel_dust -= vK(pmb->porb, pco->x1v(iu+i), pco->x2v(j), pco->x3v(k));

              dust_vel1_ghost = dust_vel1_active;
              dust_vel2_ghost = 0.0;
              dust_vel3_ghost = vel_dust;
            }
          }
        }
      }
    }
  }
}
