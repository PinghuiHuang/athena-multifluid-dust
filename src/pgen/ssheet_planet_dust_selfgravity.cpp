//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//======================================================================================
//! \file ssheet.cpp
//! \brief Shearing wave problem generator.
//! Several different initial conditions:
//!  - ipert = 1  pure shearing background flow
//!               JG: Johnson & Gammie 2005, ApJ, 626, 978
//======================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <fstream>    // ofstream
#include <iomanip>    // setprecision
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
#include "../fft/athena_fft.hpp"
#include "../globals.hpp"
#include "../gravity/fft_gravity.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"

#if MAGNETIC_FIELDS_ENABLED
#error "This problem generator does not work with MHD."
#endif

#if (!SELF_GRAVITY_ENABLED)
#error "This problem generator requires self-gravity!"
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//#if NON_BAROTROPIC_EOS
//#error "This problem generator requires isothermal equation of state!"
//#endif

namespace {

Real x1size, x2size, x3size, sound_speed, gm1, inv_gm1, d0, qshear, Omega0, beta, four_pi_G,
kappap, kappap2, Kai0, etaVk, AN, BN, Psi, time_drag, user_dt, rs, gmp, gMth, t_planet_growth,
x1min, x1max, damping_rate, radius_inner_damping, radius_outer_damping,
inner_ratio_region, outer_ratio_region, inner_width_damping, outer_width_damping;
Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS];

bool DustDrift_Flag, Damping_Flag;
int PlanetaryGravityOrder;

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void VerticalGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
      int is, int ie, int js, int je, int ks, int ke);

Real UserTimeStep(MeshBlock *pmb);
} // namespace

void WaveDamping(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df);
void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);

void SsheetUpperX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void SsheetLowerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  int shboxcoord = pin->GetInteger("orbital_advection", "shboxcoord");
  if (shboxcoord == 2) {
    std::stringstream msg;
    msg << "The shboxcoord must be equaled to 1!" << std::endl;
    ATHENA_ERROR(msg);
  }

  four_pi_G = pin->GetReal("problem","four_pi_G");
  SetFourPiG(four_pi_G);

  qshear = pin->GetReal("orbital_advection", "qshear");
  Omega0 = pin->GetReal("orbital_advection", "Omega0");

  gmp  = pin->GetOrAddReal("problem", "GMp", 0.0);  // GM of the planet
  beta = pin->GetOrAddReal("problem", "beta", 0.0);

  t_planet_growth = pin->GetOrAddReal("problem", "t_planet_growth", 5.0)*TWO_PI; // orbital number to grow the planet

  PlanetaryGravityOrder = pin->GetOrAddInteger("problem", "PlanetaryGravityOrder", 2);
  if ((PlanetaryGravityOrder != 2) || (PlanetaryGravityOrder != 4))
    PlanetaryGravityOrder = 2;

  DustDrift_Flag = pin->GetBoolean("problem", "DustDrift_Flag");
  Damping_Flag   = pin->GetOrAddBoolean("problem", "Damping_Flag", true);
  damping_rate   = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  time_drag = pin->GetOrAddReal("dust",    "time_drag", 0.0);
  user_dt   = pin->GetOrAddReal("time",    "user_dt",   0.0);
  etaVk     = pin->GetOrAddReal("problem", "etaVk",     0.0);
  d0        = pin->GetOrAddReal("problem", "d0",        1.0);

  if (NON_BAROTROPIC_EOS) {
    gm1         = (pin->GetReal("hydro", "gamma") - 1.0);
    inv_gm1     = 1.0/gm1;
    sound_speed = pin->GetReal("hydro", "init_sound_speed");
  } else {
    sound_speed = pin->GetReal("hydro", "iso_sound_speed");
  }

  rs  = pin->GetOrAddReal("problem", "rs",  0.6);  // softening length of the gravitational potential of planets
  rs *= (sound_speed/Omega0);

  gMth = SQR(sound_speed)*sound_speed/Omega0;

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 1.2);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, -TWO_3RD);
  radius_outer_damping = x1max*pow(outer_ratio_region, -TWO_3RD);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  kappap  = 2.0*(2.0 - qshear);
  kappap2 = SQR(kappap);
  Kai0    = 2.0*etaVk*sound_speed;
  AN      = 1.0;
  BN      = 1.0;
  Psi     = 1.0/(SQR(AN) + kappap2*SQR(BN));

  if (!shear_periodic) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ssheet.cpp ProblemGenerator" << std::endl
        << "This problem generator requires shearing box."   << std::endl;
    ATHENA_ERROR(msg);
  }

  if (mesh_size.nx2 == 1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in ssheet.cpp ProblemGenerator" << std::endl
        << "This problem does NOT work on a 1D grid." << std::endl;
    ATHENA_ERROR(msg);
  }

  x1size = mesh_size.x1max - mesh_size.x1min;
  x2size = mesh_size.x2max - mesh_size.x2min;
  x3size = mesh_size.x3max - mesh_size.x3min;

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      // Dust to gas ratio
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
    }
    EnrollUserDustStoppingTime(MyStoppingTime);
    EnrollDustDiffusivity(MyDustDiffusivity);
  }

  EnrollUserExplicitSourceFunction(MySource);

  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(UserTimeStep);

  if (mesh_bcs[BoundaryFace::inner_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x3, SsheetUpperX3);
  }
  if (mesh_bcs[BoundaryFace::outer_x3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x3, SsheetLowerX3);
  }

  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  int shboxcoord = porb->shboxcoord;
  int il = is - NGHOST; int iu = ie + NGHOST;
  int jl = js - NGHOST; int ju = je + NGHOST;
  int kl = ks;          int ku = ke;

  if (block_size.nx3 > 1) {
    kl = ks - NGHOST;
    ku = ke + NGHOST;
  }

  //Real phia(0.0);

  Real Hg = sound_speed/Omega0;
  Real no_orb_adv;
  (!porb->orbital_advection_defined) ? no_orb_adv = 1.0: no_orb_adv = 0.0;

  // update the physical variables as initial conditions
  for (int k=kl; k<=ku; k++) {
    Real x3 = pcoord->x3v(k);
    for (int j=jl; j<=ju; j++) {
      Real x2 = pcoord->x2v(j);
      for (int i=il; i<=iu; i++) {
        Real x1    = pcoord->x1v(i);
        Real K_vel = qshear*Omega0*x1;

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        gas_dens = d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));
        gas_mom1 = 0.0;
        gas_mom2 = -no_orb_adv*gas_dens*K_vel;
        gas_mom3 = 0.0;

        Real phia = SQR(TWO_PI);
        phia = -gas_dens*four_pi_G/phia;

        //gas_mom1 = phia;
        //gas_mom2 = phia - no_orb_adv*gas_dens*K_vel;
        //gas_mom3 = phia;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_erg = phydro->u(IEN,k,j,i);
          gas_erg = SQR(sound_speed)*gas_dens*inv_gm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_dens;
        }

        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n)
            pscalars->s(n, k, j, i) = phydro->u(IDN, k, j, i);
        }
      }

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          for (int i=il; i<=iu; i++) {
            Real x1    = pcoord->x1v(i);
            Real K_vel = qshear*Omega0*x1;

            Real &dust_dens = pdustfluids->df_u(rho_id, k, j, i);
            Real &dust_mom1 = pdustfluids->df_u(v1_id,  k, j, i);
            Real &dust_mom2 = pdustfluids->df_u(v2_id,  k, j, i);
            Real &dust_mom3 = pdustfluids->df_u(v3_id,  k, j, i);

            dust_dens = initial_D2G[n]*phydro->u(IDN, k, j, i);
            dust_mom1 = 0.0;
            dust_mom2 = -no_orb_adv*dust_dens*K_vel;
            dust_mom3 = 0.0;
          }
        }
      }
    }
  }

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================


namespace {

void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  if (gmp > 0.0)
    PlanetaryGravity(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  if ((NDUSTFLUIDS > 0) && (time >= time_drag) && DustDrift_Flag)
    PressureGradient(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  if (NON_BAROTROPIC_EOS && (beta > 0.0))
    ThermalRelaxation(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  if (pmb->block_size.nx3 > 1)
    VerticalGravity(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku) {

    Real inv_Omega = 1.0/Omega0;

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


void MyDustDiffusivity(DustFluids *pdf, MeshBlock *pmb,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust,
      AthenaArray<Real> &cs_dust, int is, int ie, int js, int je, int ks, int ke) {

    Real inv_Omega = 1.0/Omega0;

    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            Real &nu_gas      = pmb->phydro->hdif.nu(0, k, j, i);
            Real &diffusivity = nu_dust(dust_id, k, j, i);
            diffusivity       = nu_gas/(1.0 + SQR(Stokes_number[dust_id]));

            Real &soundspeed  = cs_dust(dust_id, k, j, i);
            soundspeed        = std::sqrt(diffusivity*inv_Omega);
          }
        }
      }
    }
  return;
}



void PlanetaryGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  Real time_growth = t_planet_growth/Omega0*(gmp/gMth);
  Real planet_gm = gmp*std::sin(0.5*PI*std::min(time/time_growth, 1.0));

  int nc1 = pmb->ncells1;

  AthenaArray<Real> x_dis, y_dis, z_dis, acc_x, acc_y, acc_z;
  x_dis.NewAthenaArray(nc1); y_dis.NewAthenaArray(nc1); z_dis.NewAthenaArray(nc1);
  acc_x.NewAthenaArray(nc1); acc_y.NewAthenaArray(nc1); acc_z.NewAthenaArray(nc1);

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x_dis = pmb->pcoord->x1v(i);
        Real y_dis = pmb->pcoord->x2v(j);
        Real z_dis = pmb->pcoord->x3v(k);

        Real dis_square = SQR(x_dis) + SQR(y_dis) + SQR(z_dis);

        Real gravity;
        gravity = ((PlanetaryGravityOrder == 2)) ?
        (planet_gm/std::pow(dis_square + SQR(rs), 1.5)) :
        (planet_gm*(5.0*SQR(rs)+2.0*dis_square)/(2.0*std::pow(SQR(rs)+dis_square, 2.5)));

        acc_x(i) = gravity*x_dis; // radial acceleration
        acc_y(i) = gravity*y_dis; // asimuthal acceleration
        acc_z(i) = gravity*z_dis; // vertical acceleration

        const Real &gas_rho  = prim(IDN, k, j, i);
        const Real &gas_vel1 = prim(IVX, k, j, i);
        const Real &gas_vel2 = prim(IVY, k, j, i);
        const Real &gas_vel3 = prim(IVZ, k, j, i);

        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);

        Real delta_mom_x = -dt*gas_rho*acc_x(i);
        Real delta_mom_y = -dt*gas_rho*acc_y(i);
        Real delta_mom_z = -dt*gas_rho*acc_z(i);

        gas_mom1 += delta_mom_x;
        gas_mom2 += delta_mom_y;
        gas_mom3 += delta_mom_z;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_erg  = cons(IEN, k, j, i);
          gas_erg       += (delta_mom_x*gas_vel1 + delta_mom_y*gas_vel2 + delta_mom_z*gas_vel3);
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
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            const Real &dust_rho = prim_df(rho_id, k, j, i);

            Real &dust_mom1 = cons_df(v1_id, k, j, i);
            Real &dust_mom2 = cons_df(v2_id, k, j, i);
            Real &dust_mom3 = cons_df(v3_id, k, j, i);

            Real delta_mom_dust_x = -dt*dust_rho*acc_x(i);
            Real delta_mom_dust_y = -dt*dust_rho*acc_y(i);
            Real delta_mom_dust_z = -dt*dust_rho*acc_z(i);

            dust_mom1 += delta_mom_dust_x;
            dust_mom2 += delta_mom_dust_y;
            dust_mom3 += delta_mom_dust_z;
          }
        }
      }
    }
  }
  return;
}


void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  //for (int k=pmb->ks; k<=pmb->ke; ++k) {
    //for (int j=pmb->js; j<=pmb->je; ++j) {
//#pragma omp simd
      //for (int i=pmb->is; i<=pmb->ie; ++i) {
        //const Real &gas_rho  = prim(IDN, k, j, i);
        //Real press_gra       = gas_rho*Kai0*Omega0*dt;
        //Real &gas_mom1       = cons(IM1, k, j, i);
        //gas_mom1            += press_gra;
      //}
    //}
  //}

	for (int n=0; n<NDUSTFLUIDS; ++n) {
		int dust_id = n;
		int rho_id  = 4*dust_id;
		int v1_id   = rho_id + 1;
		int v2_id   = rho_id + 2;
		int v3_id   = rho_id + 3;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
					const Real &dust_rho  = prim_df(rho_id, k, j, i);
					Real press_gra        = dust_rho*Kai0*Omega0*dt;
					Real &dust_mom1       = cons_df(v1_id, k, j, i);
					dust_mom1            -= press_gra;
				}
			}
		}
	}
  return;
}


void ThermalRelaxation(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  Real inv_beta = 1.0/beta;
  Real mygam    = pmb->peos->GetGamma();
  Real igm1     = 1.0/(mygam - 1.0);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &gas_pre = prim(IPR, k, j, i);

        Real &gas_erg   = cons(IEN, k, j, i);
        Real delta_erg  = (gas_pre - gas_rho*sound_speed)*igm1*Omega0*inv_beta*dt;
        gas_erg        -= delta_erg;
      }
    }
  }
  return;
}


void VerticalGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  int nc1 = pmb->ncells1;
  AthenaArray<Real> vert_gravity(nc1);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real vertical_dis = pmb->pcoord->x3v(k);
        vert_gravity(i)   = -SQR(Omega0)*vertical_dis;

        const Real &gas_rho = prim(IDN, k, j, i);
        Real &gas_mom2      = cons(IM2, k, j, i);
        Real &gas_mom3      = cons(IM3, k, j, i);

        gas_mom3 += gas_rho*vert_gravity(i)*dt;
      }

      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          const Real &dust_rho = prim_df(rho_id, k, j, i);
          Real &dust_mom2      = cons_df(v2_id,  k, j, i);
          Real &dust_mom3      = cons_df(v3_id,  k, j, i);

          dust_mom3 += dust_rho*vert_gravity(i)*dt;
        }
      }
    }
  }
  return;
}

Real UserTimeStep(MeshBlock *pmb) {

  Real &min_user_dt = user_dt;
  return min_user_dt;
}

} // namespace


void WaveDamping(MeshBlock *pmb, const Real time, const Real dt, int il, int iu,
    int jl, int ju, int kl, int ku, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df) {

  Real mygam = pmb->peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);

  Real inv_inner_damp = 1.0/inner_width_damping;
  Real inv_outer_damp = 1.0/outer_width_damping;

  Real no_orb_adv;
  (!pmb->porb->orbital_advection_defined) ? no_orb_adv = 1.0: no_orb_adv = 0.0;

  int nc1 = pmb->ncells1;
  Real Hg = sound_speed/Omega0;

  AthenaArray<Real> K_vel, R_func, inv_damping_tau;
  K_vel.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pmb->pcoord->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1  = pmb->pcoord->x1v(i);
        K_vel(i) = qshear*Omega0*x1;

        if (x1 <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          R_func(i)          = SQR((x1 - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = (damping_rate*Omega0);

          Real gas_rho_0  = d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));
          Real gas_vel1_0 = 0.0;
          Real gas_vel2_0 = -no_orb_adv*K_vel(i);
          Real gas_vel3_0 = 0.0;

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_pre      = prim(IPR, k, j, i);
            Real &gas_erg      = cons(IEN, k, j, i);
            Real gas_pre_0     = SQR(sound_speed)*gas_rho_0;
            Real delta_gas_pre = (gas_pre_0 - gas_pre)*R_func(i)*inv_damping_tau(i)*dt;

            gas_pre += delta_gas_pre;
            Real Ek  = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
            gas_erg  = gas_pre*igm1 + Ek/gas_dens;
          }
        }

        if (x1 >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          R_func(i)          = SQR((x1 - radius_outer_damping)*inv_outer_damp);
          inv_damping_tau(i) = (damping_rate*Omega0);

          Real gas_rho_0  = d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));
          Real gas_vel1_0 = 0.0;
          Real gas_vel2_0 = -no_orb_adv*K_vel(i);
          Real gas_vel3_0 = 0.0;

          Real &gas_rho  = prim(IDN, k, j, i);
          Real &gas_vel1 = prim(IM1, k, j, i);
          Real &gas_vel2 = prim(IM2, k, j, i);
          Real &gas_vel3 = prim(IM3, k, j, i);

          Real &gas_dens = cons(IDN, k, j, i);
          Real &gas_mom1 = cons(IM1, k, j, i);
          Real &gas_mom2 = cons(IM2, k, j, i);
          Real &gas_mom3 = cons(IM3, k, j, i);

          Real delta_gas_rho  = (gas_rho_0  - gas_rho )*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

          gas_rho  += delta_gas_rho;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;

          gas_dens = gas_rho;
          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_pre      = prim(IPR, k, j, i);
            Real &gas_erg      = cons(IEN, k, j, i);
            Real gas_pre_0     = SQR(sound_speed)*gas_rho_0;
            Real delta_gas_pre = (gas_pre_0 - gas_pre)*R_func(i)*inv_damping_tau(i)*dt;

            gas_pre += delta_gas_pre;
            Real Ek  = 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3));
            gas_erg  = gas_pre*igm1 + Ek/gas_dens;
          }
        }
      }
    }
  }
}


void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {

  Real Hg = sound_speed/Omega0;
  Real no_orb_adv;
  (!pmb->porb->orbital_advection_defined) ? no_orb_adv = 1.0: no_orb_adv = 0.0;

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        Real x3 = pmb->pcoord->x3v(k);
        for (int j=jl; j<=ju; ++j) {
          Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real x1    = pmb->pcoord->x1v(i);
            Real K_vel = qshear*Omega0*x1;

            Real &dust_rho  = prim_df(rho_id, k, j, i);
            Real &dust_vel1 = prim_df(v1_id,  k, j, i);
            Real &dust_vel2 = prim_df(v2_id,  k, j, i);
            Real &dust_vel3 = prim_df(v3_id,  k, j, i);

            Real &dust_dens = cons_df(rho_id, k, j, i);
            Real &dust_mom1 = cons_df(v1_id,  k, j, i);
            Real &dust_mom2 = cons_df(v2_id,  k, j, i);
            Real &dust_mom3 = cons_df(v3_id,  k, j, i);

            dust_rho   = initial_D2G[dust_id]*d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));
            dust_vel1  = 0.0;
            dust_vel2  = 0.0;
            dust_vel3  = 0.0;
            dust_vel2 -= no_orb_adv*K_vel;

            dust_dens = dust_rho;
            dust_mom1 = dust_rho*dust_vel1;
            dust_mom2 = dust_rho*dust_vel2;
            dust_mom3 = dust_rho*dust_vel3;
          }
        }
      }
    }
  }
  return;
}


void SsheetUpperX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  Real Hg = sound_speed/Omega0;
  Real no_orb_adv;
  (!pmb->porb->orbital_advection_defined) ? no_orb_adv = 1.0: no_orb_adv = 0.0;

  for (int k=1; k<=ngh; ++k) {
		Real x3 = pmb->pcoord->x3v(kl - k);
    for (int j=jl; j<=ju; ++j) {
			Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1    = pmb->pcoord->x1v(i);
        Real K_vel = qshear*Omega0*x1;

        Real &gas_rho_ghost  = pmb->phydro->w(IDN, kl - k, j, i);
        Real &gas_vel1_ghost = pmb->phydro->w(IM1, kl - k, j, i);
        Real &gas_vel2_ghost = pmb->phydro->w(IM2, kl - k, j, i);
        Real &gas_vel3_ghost = pmb->phydro->w(IM3, kl - k, j, i);

        gas_rho_ghost  = d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));
        gas_vel1_ghost = 0.0;
        gas_vel2_ghost = -no_orb_adv*K_vel;
        gas_vel3_ghost = 0.0;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, kl - k, j, i);
          gas_pres_ghost       = SQR(sound_speed)*gas_rho_ghost;
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
					for (int i=il; i<=iu; ++i) {
						Real x1    = pmb->pcoord->x1v(i);
						Real K_vel = qshear*Omega0*x1;

            Real gas_rho_ghost = d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));

            Real &dust_rho_ghost  = pmb->pdustfluids->df_w(rho_id, kl - k, j, i);
            Real &dust_vel1_ghost = pmb->pdustfluids->df_w(v1_id,  kl - k, j, i);
            Real &dust_vel2_ghost = pmb->pdustfluids->df_w(v2_id,  kl - k, j, i);
            Real &dust_vel3_ghost = pmb->pdustfluids->df_w(v3_id,  kl - k, j, i);

            dust_rho_ghost  = initial_D2G[n]*gas_rho_ghost;
            dust_vel1_ghost = 0.0;
            dust_vel2_ghost = -no_orb_adv*K_vel;
            dust_vel3_ghost = 0.0;
          }
        }
      }
    }
  }
	return;
}


void SsheetLowerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {


  Real Hg = sound_speed/Omega0;
  Real no_orb_adv;
  (!pmb->porb->orbital_advection_defined) ? no_orb_adv = 1.0: no_orb_adv = 0.0;

  for (int k=1; k<=ngh; ++k) {
		Real x3 = pmb->pcoord->x3v(ku + k);
    for (int j=jl; j<=ju; ++j) {
			Real x2 = pmb->pcoord->x2v(j);
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real x1    = pmb->pcoord->x1v(i);
        Real K_vel = qshear*Omega0*x1;

        Real &gas_rho_ghost  = pmb->phydro->w(IDN, ku + k, j, i);
        Real &gas_vel1_ghost = pmb->phydro->w(IM1, ku + k, j, i);
        Real &gas_vel2_ghost = pmb->phydro->w(IM2, ku + k, j, i);
        Real &gas_vel3_ghost = pmb->phydro->w(IM3, ku + k, j, i);

        gas_rho_ghost  = d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));
        gas_vel1_ghost = 0.0;
        gas_vel2_ghost = -no_orb_adv*K_vel;
        gas_vel3_ghost = 0.0;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pres_ghost = prim(IEN, ku + k, j, i);
          gas_pres_ghost       = SQR(sound_speed)*gas_rho_ghost;
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
					for (int i=il; i<=iu; ++i) {
						Real x1    = pmb->pcoord->x1v(i);
						Real K_vel = qshear*Omega0*x1;

            Real gas_rho_ghost = d0*std::exp(-SQR(x3)/(2.0*SQR(Hg)));

            Real &dust_rho_ghost  = pmb->pdustfluids->df_w(rho_id, ku + k, j, i);
            Real &dust_vel1_ghost = pmb->pdustfluids->df_w(v1_id,  ku + k, j, i);
            Real &dust_vel2_ghost = pmb->pdustfluids->df_w(v2_id,  ku + k, j, i);
            Real &dust_vel3_ghost = pmb->pdustfluids->df_w(v3_id,  ku + k, j, i);

            dust_rho_ghost  = initial_D2G[n]*gas_rho_ghost;
            dust_vel1_ghost = 0.0;
            dust_vel2_ghost = -no_orb_adv*K_vel;
            dust_vel3_ghost = 0.0;
          }
        }
      }
    }
  }
	return;
}



//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

void MeshBlock::UserWorkInLoop() {

  Real &time = pmy_mesh->time;
  Real &dt   = pmy_mesh->dt;
  Real mygam = peos->GetGamma();
  Real igm1  = 1.0/(mygam - 1.0);
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;
  int ku = ke + dk;
  int jl = js - NGHOST;
  int ju = je + NGHOST;
  int il = is - NGHOST;
  int iu = ie + NGHOST;

  if (Damping_Flag)
    WaveDamping(this, time, dt, il, iu, jl, ju, kl, ku,
        phydro->w, pdustfluids->df_w, phydro->u, pdustfluids->df_u);

  if ((time_drag != 0.0) && (time < time_drag))
    FixedDust(this, il, iu, jl, ju, kl, ku, pdustfluids->df_w, pdustfluids->df_u);

  return;
}


void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  MeshBlock *pmb = my_blocks(0);
  int is = pmb->is, ie = pmb->ie;
  int js = pmb->js, je = pmb->je;
  int ks = pmb->ks, ke = pmb->ke;

  int nlim = pin->GetInteger("time","nlim");

  if (SELF_GRAVITY_ENABLED) {
    if (nlim == 0) {
      // timing measure after loop
      int ncycle = pin->GetInteger("problem","ncycle");
      if (Globals::my_rank == 0) {
        std::cout << "=====================================================" << std::endl;
        std::cout << "Call Poisson Solver  " << ncycle << " times          " << std::endl;
        std::cout << "=====================================================" << std::endl;
      }
      clock_t tstart = clock();

#ifdef OPENMP_PARALLEL
      double omp_start_time = omp_get_wtime();
#endif

      for (int n=0; n < ncycle; n++) {
        for (int b=0; b<nblocal; ++b) {
          pmb = my_blocks(b);
          pmb->pgrav->phi.ZeroClear();
        }
        if (SELF_GRAVITY_ENABLED == 1) pfgrd->Solve(1,1);
        else if (SELF_GRAVITY_ENABLED == 2) pmgrd->Solve(1);
      }

#ifdef OPENMP_PARALLEL
      double omp_time = omp_get_wtime() - omp_start_time;;
#endif
      clock_t tstop = clock();
      float cpu_time = (tstop>tstart ? static_cast<Real>(tstop-tstart) : 1.0) /
                       static_cast<Real>(CLOCKS_PER_SEC);
      std::int64_t zones = GetTotalCells();
      std::int64_t mb_zones = GetTotalCells()/nbtotal*nblist[Globals::my_rank];
      double zc_cpus = static_cast<double>(mb_zones*ncycle)/cpu_time;
      double zc_cpus2 =
          static_cast<double>(mb_zones
                              *static_cast<double>(std::log2(mb_zones))
                              *ncycle)/cpu_time;

      if (Globals::my_rank == 0) {
        std::cout << "Timing Possison Solver                               " << std::endl;
        std::cout << "number of zones in Mesh = " << zones << std::endl;
        std::cout << "Mesh configuration = " << mesh_size.nx1 << "x"
                  << mesh_size.nx2 << "x" << mesh_size.nx3 << std::endl;
        std::cout << "number of zones in MeshBlock = " << mb_zones << std::endl;
        std::cout << "MeshBlock configuration = " << pmb->block_size.nx1 << "x"
                  << pmb->block_size.nx2 << "x" << pmb->block_size.nx3 << std::endl;
        std::cout << "number of processors = " << Globals::nranks << std::endl;
        std::cout << "processor configuration = "
                  << nrbx1 << "x" << nrbx2 << "x" << nrbx3 << std::endl;
        std::cout << "cpu time used  = " << cpu_time << std::endl;
        std::cout << "cpu time used/cycle  = " << cpu_time/static_cast<Real>(ncycle)
                  << std::endl;
        std::cout << "zone-cycles/cpu_second = " << zc_cpus << std::endl;
        std::cout << "zone-cycles(NlogN)/cpu_second = " << zc_cpus2 << std::endl;
#ifdef OPENMP_PARALLEL
        std::cout << "=====================================================" << std::endl;
        float zc_omps = static_cast<Real>(zones*ncycle)/omp_time;
        std::cout << "omp number of threads = " << GetNumMeshThreads() << std::endl;
        std::cout << "omp wtime used = " << omp_time << std::endl;
        std::cout << "zone-cycles/omp_wsecond = " << zc_omps << std::endl;
#endif
        std::cout << "=====================================================" << std::endl;
      }

      Real err1 = 0.0, maxphi = 0.0; // err2 = 0.0
      for (int b=0; b<nblocal; ++b) {
        pmb = my_blocks(b);
        Hydro *phydro = pmb->phydro;
        Gravity *pgrav = pmb->pgrav;
        for (int k=ks; k<=ke; ++k) {
          for (int j=js; j<=je; ++j) {
            for (int i=is; i<=ie; ++i) {
              err1 += std::abs(pgrav->phi(k,j,i) - phydro->u(IM2,k,j,i));
              //          err2 += pgrav->phi(k,j,i)/phydro->u(IM2,k,j,i);
              maxphi = std::max(pgrav->phi(k,j,i),maxphi);
            }
          }
        }
      }
#ifdef MPI_PARALLEL
      MPI_Allreduce(MPI_IN_PLACE, &err1, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &maxphi, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif

      err1 = err1/static_cast<Real>(zones);

      Real x1size = mesh_size.x1max - mesh_size.x1min;
      Real x2size = mesh_size.x2max - mesh_size.x2min;
      Real x3size = mesh_size.x3max - mesh_size.x3min;
      Real four_pi_G = pin->GetReal("problem", "four_pi_G");
      Real phiamp = SQR(TWO_PI/x1size);
      phiamp += SQR(TWO_PI/x2size);
      phiamp += SQR(TWO_PI/x3size);
      phiamp = 1.0*four_pi_G/phiamp;

      if (Globals::my_rank == 0) {
        std::cout << std::scientific
                  << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1);
        std::cout << "=====================================================" << std::endl;
        std::cout << "L1 : " << err1 <<" MaxPhi: " << maxphi
                  << " Amp: " << phiamp << std::endl;
        std::cout << "=====================================================" << std::endl;
      }
    }
  }

  return;
}
