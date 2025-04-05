//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in cylindrical
//! coordinate.  Initial conditions are in vertical hydrostatic eqm.

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

//#if (NDUSTFLUIDS!=1)
//#error "This problem generator requires NDUSTFLUIDS == 1!"
//#endif

namespace {
Real PoverRho(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_bump(const Real rad, const Real phi, const Real z, const Real diff);
Real DenProfileCyl_dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_dust(const Real rad, const Real phi, const Real z);

void GetCylCoord(Coordinates *pco, Real &rad, Real &phi, Real &z, int i, int j, int k);
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
      const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void DustVelProfileCyl_NSH(const Real Ts, const Real SN, const Real QN, const Real Psi,
      const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void Keplerian_interpolate(const Real r_active, const Real r_ghost, const Real vphi_active,
      Real &vphi_ghost);
void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
      const Real slope, Real &rho_ghost);

Real Keplerian_velocity(const Real rad);
Real Delta_gas_vr(const Real vk,    const Real SN, const Real QN,    const Real Psi);
Real Delta_gas_vphi(const Real vk,  const Real SN, const Real QN,    const Real Psi);
Real Delta_dust_vr(const Real ts,   const Real vk, const Real d_vgr, const Real d_vgphi);
Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi);

// problem parameters which are useful to make global to this file
bool Damping_Flag, Isothermal_Flag, Bump_Flag;

Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, beta, alpha_vis,
dfloor, dffloor, Omega0, user_dt, phi_vortex,
vel_vortex, r0_vortex, x0_vortex, y0_vortex, amp, time_drag, x1min, x1max,
tau_damping, damping_rate, radius_inner_damping, radius_outer_damping,
inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping,
A_bump, sigma_bump, r0_bump, width_vortex, eta_gas, beta_gas, ks_gas,
SN_const(0.0), QN_const(0.0), Psi_const(0.0),
refine_factor, derefine_factor, refine_rad_min, refine_rad_max;

Real initial_D2G[NDUSTFLUIDS], ring_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS], weight_dust[NDUSTFLUIDS];


// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
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

int RefinementCondition(MeshBlock *pmb);
void Vr_interpolate_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost);
} // namespace

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void LocalIsothermalEOS(MeshBlock *pmb, int il, int iu, int jl,
    int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void WaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &cons);
void WaveDampingDust(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0             = pin->GetOrAddReal("problem", "GM", 0.0);
  r0              = pin->GetOrAddReal("problem", "r0", 1.0);
  amp             = pin->GetOrAddReal("problem", "amp", 0.0);
  Damping_Flag    = pin->GetBoolean("problem", "Damping_Flag");
  Isothermal_Flag = pin->GetBoolean("problem", "Isothermal_Flag");
  Bump_Flag       = pin->GetBoolean("problem", "Bump_Flag");

  // Get parameters for initial density and velocity
  rho0      = pin->GetReal("problem", "rho0");
  dslope    = pin->GetOrAddReal("problem", "dslope", -1.0);
  time_drag = pin->GetOrAddReal("dust", "time_drag", 0.0);

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
    pslope     = pin->GetOrAddReal("problem", "pslope", -0.5);
    gamma_gas  = pin->GetReal("hydro", "gamma");
    beta       = pin->GetOrAddReal("problem", "beta", 0.0);
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  // parameters of refinement
  refine_factor   = pin->GetOrAddReal("problem", "refine_factor",   0.01);
  derefine_factor = pin->GetOrAddReal("problem", "derefine_factor", 0.005);
  refine_rad_min  = pin->GetOrAddReal("problem", "refine_rad_min",  0.2);
  refine_rad_max  = pin->GetOrAddReal("problem", "refine_rad_max",  2.0);

  Real float_min = std::numeric_limits<float>::min();
  dfloor   = pin->GetOrAddReal("hydro", "dfloor",  (1024*(float_min)));
  dffloor  = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));
  Omega0   = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
  alpha_vis = pin->GetOrAddReal("problem", "alpha_vis", 0.0);

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_"   + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_"        + std::to_string(n+1));
      ring_D2G[n]      = pin->GetReal("dust", "ring_D2G_"      + std::to_string(n+1));
      weight_dust[n]   = 1.0/(Stokes_number[n] + (1.0+SQR(initial_D2G[n]))/Stokes_number[n]);
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

  A_bump     = pin->GetOrAddReal("problem", "A_bump",     0.0);
  sigma_bump = pin->GetOrAddReal("problem", "sigma_bump", 0.0);
  r0_bump    = pin->GetOrAddReal("problem", "r0_bump",    0.0);

  r0_vortex  = pin->GetOrAddReal("problem",  "r0_vortex",  r0_bump);
  phi_vortex = pin->GetOrAddReal("problem",  "phi_vortex", PI);
  vel_vortex = (pin->GetOrAddReal("problem", "vel_vortex", -0.5));

  width_vortex = std::sqrt(p0_over_r0*std::pow(r0_vortex/r0, pslope + 1.0))*r0_vortex;

  x0_vortex = r0_vortex*std::cos(phi_vortex);
  y0_vortex = r0_vortex*std::sin(phi_vortex);

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.2);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, TWO_3RD);
  radius_outer_damping = x1max*pow(outer_ratio_region, -TWO_3RD);

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

  if (NDUSTFLUIDS > 0) {
    EnrollUserDustStoppingTime(MyStoppingTime);
    EnrollDustDiffusivity(MyDustDiffusivity);
  }

  // Enroll damping zone and local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined AMR criterion
  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);

  return;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{

  AllocateUserOutputVariables(4);
  SetUserOutputVariableName(0, "ratio_1");
  SetUserOutputVariableName(1, "sound_speed");
  SetUserOutputVariableName(2, "vortensity_z");
  SetUserOutputVariableName(3, "dust_flux");

  AllocateRealUserMeshBlockDataField(2);
// List of User field:
// 0: initial temperature profile (1d array)

  ruser_meshblock_data[0].NewAthenaArray(block_size.nx1+2*NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx1+2*NGHOST);

  for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
    //Real rad, phi, z;
    //GetCylCoord(pcoord, rad, phi, z, i, 0, 0);
    Real &rad = pcoord->x1v(i);
    ruser_meshblock_data[0](i) = PoverRho(rad, 0, 0);
    ruser_meshblock_data[1](i) = VelProfileCyl_gas(rad, 0, 0);
  }
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0) {
    std::stringstream msg;
    msg << "This problem file must be setup in the cylindrical coordinate!" << std::endl;
    ATHENA_ERROR(msg);
  }

  if ((block_size.nx2 == 1) || (block_size.nx3 > 1)) {
    std::stringstream msg;
    msg << "This problem file must be setup in 2D!" << std::endl;
    ATHENA_ERROR(msg);
  }

  std::int64_t iseed = -1 - gid;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

  if (Bump_Flag) {
    Real inv_2sigma2 = 1./(2.0*SQR(sigma_bump));
    for (int k=ks; k<=ke; ++k) {
      Real x3 = pcoord->x3v(k);
      for (int j=js; j<=je; ++j) {
        Real x2 = pcoord->x2v(j);
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real rad, phi, z;
          GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates

          // compute initial conditions in cylindrical coordinates
          Real den_gas_1     = DenProfileCyl_gas(rad, phi, z);
          Real den_gas_2     = A_bump * std::exp(-SQR(rad - r0_bump)*inv_2sigma2);
          Real den_gas_total = den_gas_1 *(1.0 + den_gas_2);
          Real dev_den_gas_2 = 2.0*(r0_bump - rad)*inv_2sigma2*den_gas_2;

          Real cs2       = PoverRho(rad, phi, z);
          Real omega_dyn = std::sqrt(gm0/(rad*rad*rad));
          //Real pre_diff  = pslope*cs2 + cs2/den_gas_total*(dslope*den_gas_1 +
                                        //2.0*rad*(r0_bump - rad)*inv_2sigma2*den_gas_2);

          Real pre_diff  = (pslope + dslope)*cs2 + rad*cs2*den_gas_1*dev_den_gas_2/den_gas_total;
          Real vis_vel_r = -1.5*(alpha_vis*cs2/rad/omega_dyn);
          Real vel_K     = vK(porb, x1, x2, x3);

          Real vel_gas_phi = VelProfileCyl_bump(rad, phi, z, pre_diff);
          if (porb->orbital_advection_defined)
            vel_gas_phi -= vel_K;

          Real delta_gas_vel1 = amp*std::sqrt(cs2)*(ran2(&iseed) - 0.5);
          Real delta_gas_vel2 = amp*std::sqrt(cs2)*(ran2(&iseed) - 0.5);
          Real delta_gas_vel3 = amp*std::sqrt(cs2)*(ran2(&iseed) - 0.5);

          phydro->u(IDN, k, j, i) = den_gas_total;
          phydro->u(IM1, k, j, i) = den_gas_total*(vis_vel_r + delta_gas_vel1);
          phydro->u(IM2, k, j, i) = den_gas_total*(vel_gas_phi+delta_gas_vel2);
          phydro->u(IM3, k, j, i) = 0.0;

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
              //Real den_dust = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real den_dust_1 = initial_D2G[dust_id]*den_gas_1;
              Real den_dust_2 = ring_D2G[dust_id]*den_gas_1*den_gas_2;
              Real den_dust   = den_dust_1 + den_dust_2;

              Real vel_dust_r   = weight_dust[dust_id]*pre_diff/(2.0*rad*omega_dyn);
              Real vel_dust_phi = VelProfileCyl_dust(rad, phi, z);
              if (porb->orbital_advection_defined)
                vel_dust_phi -= vel_K;

              pdustfluids->df_u(rho_id, k, j, i) = den_dust;
              pdustfluids->df_u(v1_id,  k, j, i) = den_dust*vel_dust_r;
              pdustfluids->df_u(v2_id,  k, j, i) = den_dust*vel_dust_phi;
              pdustfluids->df_u(v3_id,  k, j, i) = 0.0;
            }
          }
        }
      }
    }
  } else {
    for (int k=ks; k<=ke; ++k) {
      Real x3 = pcoord->x3v(k);
      for (int j=js; j<=je; ++j) {
        Real x2 = pcoord->x2v(j);
        for (int i=is; i<=ie; ++i) {
          Real x1 = pcoord->x1v(i);
          Real rad, phi, z;
          GetCylCoord(pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates

          Real cs2         = PoverRho(rad, phi, z);
          Real sound_speed = std::sqrt(cs2);
          Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
          Real vis_vel_r   = -1.5*(alpha_vis*cs2/rad/omega_dyn);

          Real x_dis = rad*std::cos(phi);
          Real y_dis = rad*std::sin(phi);

          Real dis_vortex_Square = SQR(x_dis - x0_vortex) + SQR(y_dis - y0_vortex);
          Real dis_vortex        = std::sqrt(dis_vortex_Square);
          //Real ellipse = SQR(rad - r0_vortex)/SQR(width_vortex) + r0_vortex*SQR(phi - PI)/SQR(10*width_vortex);

          // compute initial conditions in cylindrical coordinates
          Real den_gas     = DenProfileCyl_gas(rad, phi, z);
          Real vel_gas_phi = VelProfileCyl_gas(rad, phi, z);
          Real vel_gas_r   = vis_vel_r;

          if (dis_vortex <= width_vortex) {
          //if (ellipse <= 1.0) {
            Real num_1        = -SQR(rad) + SQR(r0_vortex) + dis_vortex_Square;
            Real num_2        = 2.0*rad*r0_vortex;
            Real num_3        = 2.0*dis_vortex*r0_vortex;
            Real vortex_speed = vel_vortex*sound_speed;

            vel_gas_r   += vortex_speed*(num_1 - num_2*std::cos(phi))*std::sin(phi)/num_3;
            vel_gas_phi += vortex_speed*(num_1*std::cos(phi) + num_2*SQR(std::sin(phi)))/num_3;
          }

          if (porb->orbital_advection_defined)
            vel_gas_phi -= vK(porb, x1, x2, x3);

          phydro->u(IDN, k, j, i) = den_gas;
          phydro->u(IM1, k, j, i) = vel_gas_r;
          phydro->u(IM2, k, j, i) = den_gas*vel_gas_phi;
          phydro->u(IM3, k, j, i) = 0.0;

          if (NON_BAROTROPIC_EOS) {
            Real cs2 = PoverRho(rad, phi, z);
            phydro->u(IEN, k, j, i)  = cs2*phydro->u(IDN, k, j, i)*igm1;
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
              Real den_dust     = DenProfileCyl_dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real vel_dust_phi = VelProfileCyl_dust(rad, phi, z);
              if (porb->orbital_advection_defined)
                vel_dust_phi -= vK(porb, x1, x2, x3);

              pdustfluids->df_u(rho_id, k, j, i) = den_dust;
              pdustfluids->df_u(v1_id,  k, j, i) = 0.0;
              pdustfluids->df_u(v2_id,  k, j, i) = den_dust*vel_dust_phi;
              pdustfluids->df_u(v3_id,  k, j, i) = 0.0;
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

  if ((!Isothermal_Flag) && (beta > 0.0) && NON_BAROTROPIC_EOS)
    ThermalRelaxation(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          //GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          Real &rad           = pmb->pcoord->x1v(i);
          const Real &gas_rho = prim(IDN, k, j, i);
          Real &st_time       = stopping_time(n, k, j, i);

          //Constant Stokes number in disk problems
          st_time = Stokes_number[n]*std::pow(rad, 1.5)*inv_sqrt_gm0;
          //st_time = Stokes_number[n]*std::pow(rad, 1.5)*inv_sqrt_gm0/gas_rho;
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

            Real &diffusivity = nu_dust(n, k, j, i);
            diffusivity       = nu_gas/(1.0 + SQR(Stokes_number[n]*inv_gas_rho));

            Real &soundspeed  = cs_dust(n, k, j, i);
            soundspeed        = std::sqrt(diffusivity*inv_Omega_K);
          }
        }
      }
    }
  return;
}


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
  v2 = vel_Keplerian+delta_dust_vphi;
  v3 = 0.0;

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
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
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
  rad = pco->x1v(i);
  phi = pco->x2v(j);
  z   = pco->x3v(k);

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


int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> &df_w = pmb->pdustfluids->df_w;
      Real rad(0.0), phi(0.0), z(0.0);
  Real maxeps  = 0.0;
  Real max_rad = 0.0;
  Real min_rad = 100.0;

  int k = pmb->ks;
  for (int j=pmb->js; j<=pmb->je; j++) {
    for (int i=pmb->is; i<=pmb->ie; i++) {
      GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k); // convert to cylindrical coordinates
      Real epsr_g = (std::abs(w(IDN,k,j,i+1) - 2.0*w(IDN,k,j,i) + w(IDN,k,j,i-1))
                   + std::abs(w(IDN,k,j+1,i) - 2.0*w(IDN,k,j,i) + w(IDN,k,j-1,i)))/w(IDN,k,j,i);

      Real epsp = (std::abs(w(IPR,k,j,i+1) - 2.0*w(IPR,k,j,i) + w(IPR,k,j,i-1))
                 + std::abs(w(IPR,k,j+1,i) - 2.0*w(IPR,k,j,i) + w(IPR,k,j-1,i)))/w(IPR,k,j,i);

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

  Real inv_mean_rad = 2.0/(min_rad + max_rad);

  // refine : curvature > 0.01
  if ((max_rad >= refine_rad_min) && ( min_rad <= refine_rad_max ) && (maxeps > (refine_factor*inv_mean_rad))) return 1;
  // derefinement: curvature < 0.005
  if ((max_rad < refine_rad_min) || ( min_rad > refine_rad_max ) || (maxeps < (derefine_factor*inv_mean_rad))) return -1;
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

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
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

        Real vis_vel_r    = -1.5*(alpha_vis*cs_square/rad_ghost/omega_dyn);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;

        gas_rho_ghost  = DenProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        gas_vel1_ghost = vis_vel_r;
        gas_vel2_ghost = vel_gas_phi;
        gas_vel3_ghost = 0.0;

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

            dust_rho_ghost     = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi  = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi      -= orb_defined*vel_K;

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_phi;
            dust_vel3_ghost = 0.0;
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

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;
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

        Real vis_vel_r    = -1.5*(alpha_vis*cs_square/rad_ghost/omega_dyn);
        Real vel_gas_phi  = VelProfileCyl_gas(rad_ghost, phi_ghost, z_ghost);
        vel_gas_phi      -= orb_defined*vel_K;

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

            dust_rho_ghost     = DenProfileCyl_dust(rad_ghost, phi_ghost, z_ghost, initial_D2G[dust_id], Hratio[dust_id]);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_ghost*omega_dyn);
            Real vel_dust_phi  = VelProfileCyl_dust(rad_ghost, phi_ghost, z_ghost);
            vel_dust_phi      -= orb_defined*vel_K;

            dust_vel1_ghost = vel_dust_r;
            dust_vel2_ghost = vel_dust_phi;
            dust_vel3_ghost = 0.0;
          }
        }
      }
    }
  }
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
        gas_pres = PoverRho(rad, phi, z)*gas_dens;
        gas_erg  = gas_pres*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_dens;
      }
    }
  }
  return;
}


void WaveDampingGas(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
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

  Real inv_2sigma2 = 1./(2.0*SQR(sigma_bump));

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

        if (rad_arr(i) <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real den_gas_1     = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real den_gas_2     = A_bump * std::exp(-SQR(rad_arr(i) - r0_bump)*inv_2sigma2);
          Real gas_rho_0     = den_gas_1 *(1.0 + den_gas_2);
          Real dev_den_gas_2 = 2.0*(r0_bump - rad_arr(i))*inv_2sigma2*den_gas_2;

          Real cs2_0    = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
          Real pre_diff = (pslope + dslope)*cs2_0 + rad_arr(i)*cs2_0*den_gas_1*dev_den_gas_2/gas_rho_0;

          Real vis_vel_r    = -1.5*(alpha_vis*cs2_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_bump(rad_arr(i), phi_arr(i), z_arr(i), pre_diff);
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);
          Real vel_gas_z    = 0.0;

          Real gas_vel1_0 = vis_vel_r;
          Real gas_vel2_0 = vel_gas_phi;
          Real gas_vel3_0 = vel_gas_z;
          Real gas_pre_0  = cs2_0*gas_rho_0;

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

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

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

        if (rad_arr(i) >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
          R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
          inv_damping_tau(i) = (damping_rate*omega_dyn(i));

          Real den_gas_1     = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
          Real den_gas_2     = A_bump * std::exp(-SQR(rad_arr(i) - r0_bump)*inv_2sigma2);
          Real gas_rho_0     = den_gas_1 *(1.0 + den_gas_2);
          Real dev_den_gas_2 = 2.0*(r0_bump - rad_arr(i))*inv_2sigma2*den_gas_2;

          Real cs2_0    = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
          Real pre_diff = (pslope + dslope)*cs2_0 + rad_arr(i)*cs2_0*den_gas_1*dev_den_gas_2/gas_rho_0;

          Real vis_vel_r    = -1.5*(alpha_vis*cs2_0/rad_arr(i)/omega_dyn(i));
          Real vel_gas_phi  = VelProfileCyl_bump(rad_arr(i), phi_arr(i), z_arr(i), pre_diff);
          vel_gas_phi      -= orb_defined*vK(pmb->porb, x1, x2, x3);
          Real vel_gas_z    = 0.0;

          Real gas_vel1_0 = vis_vel_r;
          Real gas_vel2_0 = vel_gas_phi;
          Real gas_vel3_0 = vel_gas_z;
          Real gas_pre_0  = cs2_0*gas_rho_0;

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

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_pre  = (gas_pre_0  - gas_pre )*R_func(i)*inv_damping_tau(i)*dt;

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


void WaveDampingDust(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

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

  Real inv_2sigma2 = 1./(2.0*SQR(sigma_bump));


  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real x1 = pmb->pcoord->x1v(i);
          GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

          if (rad_arr(i) <= radius_inner_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
            R_func(i)          = SQR((rad_arr(i) - radius_inner_damping)*inv_inner_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs2_0    = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
            Real pre_diff = (pslope + dslope)*cs2_0;

            Real vel_K         = vK(pmb->porb, x1, x2, x3);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_arr(i)*omega_dyn(i));
            Real vel_dust_phi  = VelProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i));
            vel_dust_phi      -= orb_defined*vel_K;
            Real vel_dust_z    = 0.0;

            Real den_gas_1 = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
            Real den_gas_2 = A_bump * std::exp(-SQR(rad_arr(i) - r0_bump)*inv_2sigma2);
            Real gas_rho_0 = den_gas_1 *(1.0 + den_gas_2);

            Real dust_rho_0  = initial_D2G[dust_id]*gas_rho_0;
            Real dust_vel1_0 = vel_dust_r;
            Real dust_vel2_0 = vel_dust_phi;
            Real dust_vel3_0 = vel_dust_z;

            Real &dust_rho  = prim_df(rho_id, k, j, i);
            Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            Real delta_dust_rho  = (dust_rho_0  - dust_rho )*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            dust_rho  += delta_dust_rho;
            dust_vel1 += delta_dust_vel1;
            dust_vel2 += delta_dust_vel2;
            dust_vel3 += delta_dust_vel3;

            dust_dens = dust_rho;
            dust_mom1 = dust_dens*dust_vel1;
            dust_mom2 = dust_dens*dust_vel2;
            dust_mom3 = dust_dens*dust_vel3;
          }

          if (rad_arr(i) >= radius_outer_damping) {
            // See de Val-Borro et al. 2006 & 2007
            omega_dyn(i)       = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
            R_func(i)          = SQR((rad_arr(i) - radius_outer_damping)*inv_outer_damp);
            inv_damping_tau(i) = (damping_rate*omega_dyn(i));

            Real cs2_0    = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
            Real pre_diff = (pslope + dslope)*cs2_0;

            Real vel_K         = vK(pmb->porb, x1, x2, x3);
            Real vel_dust_r    = weight_dust[dust_id]*pre_diff/(2.0*rad_arr(i)*omega_dyn(i));
            Real vel_dust_phi  = VelProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i));
            vel_dust_phi      -= orb_defined*vel_K;
            Real vel_dust_z    = 0.0;

            Real den_gas_1 = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
            Real den_gas_2 = A_bump * std::exp(-SQR(rad_arr(i) - r0_bump)*inv_2sigma2);
            Real gas_rho_0 = den_gas_1 *(1.0 + den_gas_2);

            Real dust_rho_0  = initial_D2G[dust_id]*gas_rho_0;
            Real dust_vel1_0 = vel_dust_r;
            Real dust_vel2_0 = vel_dust_phi;
            Real dust_vel3_0 = vel_dust_z;

            Real &dust_rho  = prim_df(rho_id, k, j, i);
            Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            Real delta_dust_rho  = (dust_rho_0  - dust_rho )*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
            Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

            dust_rho  += delta_dust_rho;
            dust_vel1 += delta_dust_vel1;
            dust_vel2 += delta_dust_vel2;
            dust_vel3 += delta_dust_vel3;

            dust_dens = dust_rho;
            dust_mom1 = dust_dens*dust_vel1;
            dust_mom2 = dust_dens*dust_vel2;
            dust_mom3 = dust_dens*dust_vel3;
          }
        }
      }
    }
  }
  return;
}



void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

  int nc1 = pmb->ncells1;
  OrbitalVelocityFunc &vK = pmb->porb->OrbitalVelocity;

  Real orb_defined;
  (pmb->porb->orbital_advection_defined) ? orb_defined = 1.0 : orb_defined = 0.0;

  AthenaArray<Real> rad_arr, phi_arr, z_arr, cs_square, vel_K, pre_diff, omega_dyn,
  den_gas_1, den_gas_2, den_gas_total, dev_den_gas_2,
  den_dust_total, vel_dust_r, vel_dust_phi, vel_dust_z, den_dust_1, den_dust_2;

  rad_arr.NewAthenaArray(nc1);
  phi_arr.NewAthenaArray(nc1);
  z_arr.NewAthenaArray(nc1);
  cs_square.NewAthenaArray(nc1);
  vel_K.NewAthenaArray(nc1);
  pre_diff.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);

  den_gas_1.NewAthenaArray(nc1);
  den_gas_2.NewAthenaArray(nc1);
  den_gas_total.NewAthenaArray(nc1);
  dev_den_gas_2.NewAthenaArray(nc1);

  den_dust_1.NewAthenaArray(nc1);
  den_dust_2.NewAthenaArray(nc1);
  den_dust_total.NewAthenaArray(nc1);
  vel_dust_r.NewAthenaArray(nc1);
  vel_dust_phi.NewAthenaArray(nc1);
  vel_dust_z.NewAthenaArray(nc1);

  Real inv_2sigma2 = 1./(2.0*SQR(sigma_bump));

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        GetCylCoord(pmb->pcoord, rad_arr(i), phi_arr(i), z_arr(i), i, j, k);

        den_gas_1(i)     = DenProfileCyl_gas(rad_arr(i), phi_arr(i), z_arr(i));
        den_gas_2(i)     = A_bump * std::exp(-SQR(rad_arr(i) - r0_bump)*inv_2sigma2);
        den_gas_total(i) = den_gas_1(i) *(1.0 + den_gas_2(i));
        dev_den_gas_2(i) = 2.0*(r0_bump - rad_arr(i))*inv_2sigma2*den_gas_2(i);

        cs_square(i) = PoverRho(rad_arr(i), phi_arr(i), z_arr(i));
        pre_diff(i)  = (pslope + dslope)*cs_square(i) +
                        rad_arr(i)*cs_square(i)*den_gas_1(i)*dev_den_gas_2(i)/den_gas_total(i);
        vel_K(i)     = vK(pmb->porb, x1, x2, x3);
        omega_dyn(i) = std::sqrt(gm0/(rad_arr(i)*rad_arr(i)*rad_arr(i)));
      }

      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          den_dust_1(i)     = initial_D2G[dust_id]*den_gas_1(i);
          den_dust_2(i)     = ring_D2G[dust_id]*den_dust_1(i)*den_gas_2(i);
          den_dust_total(i) = den_dust_1(i) + den_dust_2(i);

          //const Real &gas_rho = pmb->phydro->w(IDN, k, j, i);
          //den_dust_total(i)   = initial_D2G[dust_id]*gas_rho;

          vel_dust_r(i)    = weight_dust[dust_id]*pre_diff(i)/(2.0*rad_arr(i)*omega_dyn(i));
          vel_dust_phi(i)  = VelProfileCyl_dust(rad_arr(i), phi_arr(i), z_arr(i));
          vel_dust_phi(i) -= orb_defined*vel_K(i);

          Real &dust_rho  = prim_df(rho_id, k, j, i);
          Real &dust_vel1 = prim_df(v1_id,  k, j, i);
          Real &dust_vel2 = prim_df(v2_id,  k, j, i);
          Real &dust_vel3 = prim_df(v3_id,  k, j, i);

          Real &dust_dens = cons_df(rho_id, k, j, i);
          Real &dust_mom1 = cons_df(v1_id,  k, j, i);
          Real &dust_mom2 = cons_df(v2_id,  k, j, i);
          Real &dust_mom3 = cons_df(v3_id,  k, j, i);

          dust_rho  = den_dust_total(i);
          dust_vel1 = vel_dust_r(i);
          dust_vel2 = vel_dust_phi(i);
          dust_vel3 = vel_dust_z(i);

          dust_dens = dust_rho;
          dust_mom1 = dust_rho*dust_vel1;
          dust_mom2 = dust_rho*dust_vel2;
          dust_mom3 = dust_rho*dust_vel3;
        }
      }
    }
  }
  return;
}


void MeshBlock::UserWorkInLoop() {

  Real &time = pmy_mesh->time;
  Real &dt   = pmy_mesh->dt;
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - NGHOST; int ju = je + NGHOST;
  int il = is - NGHOST; int iu = ie + NGHOST;

  if (Isothermal_Flag)
    LocalIsothermalEOS(this, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  if (Damping_Flag)
    WaveDampingGas(this, time, dt, il, iu, jl, ju, kl, ku, phydro->w, phydro->u);

  if ((NDUSTFLUIDS > 0) && (time_drag != 0.0) && (time < time_drag))
    FixedDust(this, il, iu, jl, ju, kl, ku, pdustfluids->df_w, pdustfluids->df_u);

  return;
}


void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
{
  Coordinates *pco = pcoord;
  Real no_orb_adv;
  (!porb->orbital_advection_defined) ? no_orb_adv = 1.0: no_orb_adv = 0.0;
  OrbitalVelocityFunc &vK = porb->OrbitalVelocity;

	int rho_id = 0;
	int v1_id  = 1;
	int v2_id  = 2;
	int v3_id  = 3;

	for(int k=ks; k<=ke; k++) {
		for(int j=js; j<=je; j++) {
#pragma omp simd
			for(int i=is; i<=ie; i++) {
        //Real &dust_rho  = pdustfluids->df_w(rho_id, k, j, i);
        //Real &dust_vel1 = pdustfluids->df_w(v1_id,  k, j, i);
        //Real &dust_vel2 = pdustfluids->df_w(v2_id,  k, j, i);
        //Real &dust_vel3 = pdustfluids->df_w(v3_id,  k, j, i);

        Real &gas_rho  = phydro->w(IDN, k, j, i);
        Real &gas_vel1 = phydro->w(IM1, k, j, i);
        Real &gas_vel2 = phydro->w(IM2, k, j, i);
        Real &gas_vel3 = phydro->w(IM3, k, j, i);
        Real &gas_pres = phydro->w(IPR, k, j, i);

        // Dust-Gas Ratio
        Real &ratio = user_out_var(0, k, j, i);
        //ratio       = dust_rho/gas_rho;
        ratio       = phydro->w(IDN, k, j, i);

        // Sound Speed
        Real &sound_speed = user_out_var(1, k, j, i);
        //sound_speed       = std::sqrt(ruser_meshblock_data[0](i));
        sound_speed       = std::sqrt(gas_pres/gas_rho);

        Real &vortensity = user_out_var(2, k, j, i);
        Real vorticity_1 = (pco->h2v(i+1)*phydro->w(IM2, k, j, i+1)
                          - pco->h2v(i)*phydro->w(IM2, k, j, i))/pco->dx1v(i);
        Real vorticity_2 = (phydro->w(IM1, k, j+1, i) - phydro->w(IM1, k, j, i))/pco->dx2v(j);
        Real vorticity   = (vorticity_1 - vorticity_2)/pco->h2v(i);
        vortensity       = vorticity/gas_rho;

        Real &dust_flux  = user_out_var(3, k, j, i);
        //dust_flux        = dust_rho*dust_vel1;
        dust_flux        = gas_rho*gas_vel1;
			}
		}
	}

	return;
}
