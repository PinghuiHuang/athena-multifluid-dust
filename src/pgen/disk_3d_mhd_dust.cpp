//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//         spherical polar coordinates.  Initial conditions are in vertical hydrostatic
//         equilibrium.
//======================================================================================

// C++ headers
#include <algorithm>  // min
#include <cfloat>     // FLT_MIN
#include <cmath>      // sqrt
#include <cstdlib>    // srand
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
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"


namespace {
// User-defined physical source term
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
// User Stopping time
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time,
    const int il, const int iu, const int jl, const int ju, const int kl, const int ku);
// User-defined Grid spacing
Real CompressedX2(Real x, RegionSize rs);
// User-defined Diffusivity
void DiffusivityNonidealMHD(FieldDiffusion *pfdif, MeshBlock *pmb, const AthenaArray<Real> &w,
            const AthenaArray<Real> &bmag, int is, int ie, int js, int je, int ks, int ke);

// Vector potentials for poloidal fields
Real AphiOpen(const Real x1, const Real x2, const Real x3);
Real AphiLoop(const Real x1, const Real x2, const Real x3, Real R1, Real R2, int nloop);
void AddPoloidalField(MeshBlock *pmb);
Real MyTimeStep(MeshBlock *pmb);

// problem parameters
Real GM = 1.0, R0 = 1.0;
Real rho0, alpha, HoR0, HoRc, theta_trans;
Real beta, mu, Am_in, Am0, Bz0, A30, RBmin, taddBp;
Real eta_ofac_in, eta_ofac_mid, theta_Rin, theta_Rout, fcool;
Real Rbuf, rho_floor, time_drag;
int  finest_lev;
bool ad_hyp, ad_lin, ad_log;
Real user_dt;

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS],
weight_dust[NDUSTFLUIDS];

}

// User-defined boundary conditions for disk simulations
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void FixedDust(MeshBlock *pmb, int il, int iu, int jl, int ju, int kl, int ku,
    AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);
//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {

// Get parameters
  GM          = pin->GetOrAddReal("problem", "GM", 1.0);
  rho0        = pin->GetReal("problem",      "rho0");
  alpha       = pin->GetOrAddReal("problem", "alpha", 2.0);
  HoR0        = pin->GetOrAddReal("problem", "HoR0",  0.1);
  HoRc        = pin->GetOrAddReal("problem", "HoRc",  0.5);
  theta_trans = pin->GetOrAddReal("problem", "theta_trans", 0.4);
  fcool       = pin->GetOrAddReal("problem", "fcool", 1.0e10);
  user_dt     = pin->GetOrAddReal("problem", "user_dt", 0.0);

  if (MAGNETIC_FIELDS_ENABLED) {
    beta   = pin->GetOrAddReal("problem", "beta", 1.0e4);
    Bz0    = sqrt(2*rho0*SQR(HoR0)/beta);
    Am_in  = pin->GetOrAddReal("problem", "Am_in",  1.0);
    Am0    = pin->GetOrAddReal("problem", "Am0",    0.3);
    mu     = pin->GetOrAddReal("problem", "mu",     2.0);
    RBmin  = pin->GetOrAddReal("problem", "RBmin",  0.0);
    taddBp = pin->GetOrAddReal("problem", "taddBp", 0.0);

    ad_hyp = pin->GetOrAddBoolean("problem", "ad_hyp", false);
    ad_lin = pin->GetOrAddBoolean("problem", "ad_lin", false);
    ad_log = pin->GetOrAddBoolean("problem", "ad_log", false);

    //resistivity parameters
    eta_ofac_in  = pin->GetOrAddReal("problem", "eta_ofac_in",  0.1);
    eta_ofac_mid = pin->GetOrAddReal("problem", "eta_ofac_mid", 0.0);

    int fieldopt = pin->GetOrAddInteger("problem", "fieldopt", 1);

    if (fieldopt == 2) {// field loop
      int nloop  = pin->GetOrAddInteger("problem", "nloop", 2);
      Real R1    = pin->GetOrAddReal("problem", "Rloop1", 5.0*R0);
      Real R2    = pin->GetOrAddReal("problem", "Rloop2", 50.0*R0);
      Real Rc    = 0.5*(R1+R2);
      Real kl    = 2.0*PI*nloop/(R2-R1);
      Real pres0 = GM/Rc*SQR(HoR0)*rho0*pow(Rc/R0,-alpha);
      A30        = sqrt(2.0*pres0/400.0)/kl;
    }
  }

  Rbuf       = pin->GetOrAddReal("problem", "Rbuf", 2.0);
  rho_floor  = pin->GetReal("problem", "rho_floor");
  finest_lev = pin->GetInteger("problem", "finest_lev");

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      weight_dust[n]   = 2.0/(Stokes_number[n] + SQR(1.0+initial_D2G[n])/Stokes_number[n]);
    }
  }
  time_drag = pin->GetOrAddReal("dust", "time_drag", 0.0);

  // enroll user-defined grid spacing
  EnrollUserMeshGenerator(X2DIR, CompressedX2);

  // Compute the theta-profiles for density for prescribe temperature
  int nx2bin = mesh_size.nx2;
  nx2bin = nx2bin<<finest_lev;

  // allocate user arrays to store temperature/density profiles
  AllocateRealUserMeshDataField(3);
  ruser_mesh_data[0].NewAthenaArray(nx2bin); // temperature
  ruser_mesh_data[1].NewAthenaArray(nx2bin); // density
  ruser_mesh_data[2].NewAthenaArray(nx2bin); // dust density

  // allocate auxiliary arrays
  AthenaArray<Real> thetaf,thetac,lnsinthf,lnsinthc,F;
  thetaf.NewAthenaArray(nx2bin+1);
  thetac.NewAthenaArray(nx2bin);
  lnsinthf.NewAthenaArray(nx2bin+1);
  lnsinthc.NewAthenaArray(nx2bin);
  F.NewAthenaArray(nx2bin+1);

  for (int i=0;i<=nx2bin;++i) {
    Real rx   = (Real)(i)/(Real)(nx2bin);
    thetaf(i) = MeshGenerator_[X2DIR](rx,mesh_size);
    if ((thetaf(i)<=1.0e-2) || (thetaf(i)>=3.13159265359))
      lnsinthf(i) = log(sin(1.0e-2));
    else
      lnsinthf(i) = log(sin(thetaf(i)));
  }
  for (int i=0;i<nx2bin;++i) {
    thetac(i) = ((sin(thetaf(i+1)) - thetaf(i+1)*cos(thetaf(i+1))) -
                 (sin(thetaf(i)) - thetaf(i)*cos(thetaf(i))))/
                 (cos(thetaf(i)) - cos(thetaf(i+1)));
    lnsinthc(i) = log(sin(thetac(i)));
  }

  // compute the temperature function g(theta)=SQR(HoR_local)
  Real fac1 = 0.5*(HoRc+HoR0);
  Real fac2 = 0.5*(HoRc-HoR0);
  Real logfac1 = 0.5*(log(HoRc)+log(HoR0));
  Real logfac2 = 0.5*(log(HoRc)-log(HoR0));

  for (int i=0;i<nx2bin;++i) {
    Real delta = fabs(thetac(i)-0.5*PI);
    Real myHoR = fac1 + fac2 * tanh(2.0*(delta-theta_trans)/HoR0);
    //Real myHoR = exp(logfac1 + logfac2 * tanh((delta-theta_trans)/HoR0));
    ruser_mesh_data[0](i) = SQR(myHoR);
  }

  // now solve for F(theta)=g(theta)*f(theta) at grid interfaces
  // By definition, F(theta)=1 at theta=pi/2 (ntheta is dividable by 2)
  // Eventually, density profile f(theta) can be deduced from F and g.
  F(nx2bin/2) = SQR(HoR0);
  for (int i = nx2bin/2-1;i >= 0;--i) {
    Real dlnF = (1.0/ruser_mesh_data[0](i)-(alpha+1.0))
               *(lnsinthf(i+1)-lnsinthf(i));
    F(i)      = exp(log(F(i+1))-dlnF);
  }

  for (int i = nx2bin/2+1;i <= nx2bin;++i) {
    Real dlnF = (1.0/ruser_mesh_data[0](i-1)-(alpha+1.0))*(lnsinthf(i)-lnsinthf(i-1));
    F(i)      = exp(log(F(i-1))+dlnF);
  }

  for (int i = 0;i<nx2bin;++i) {
    ruser_mesh_data[1](i) = exp(0.5*(log(F(i))+log(F(i+1))))/ruser_mesh_data[0](i);
  }

  if (NDUSTFLUIDS > 0) {
    for (int i = 0;i<nx2bin;++i) {
      ruser_mesh_data[2](i) = exp(1./Hratio[0]*0.5*(log(F(i))+log(F(i+1))))/ruser_mesh_data[0](i);
    }
  }

  thetaf.DeleteAthenaArray();
  thetac.DeleteAthenaArray();
  lnsinthf.DeleteAthenaArray();
  lnsinthc.DeleteAthenaArray();
  F.DeleteAthenaArray();

  // enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiskInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiskOuterX1);
  }
  //if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    //EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiskInnerX2);
  //}
  //if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    //EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiskOuterX2);
  //}

  if (NDUSTFLUIDS > 0)
    EnrollUserDustStoppingTime(MyStoppingTime);

  // Enroll user-defined physical source terms
  EnrollUserExplicitSourceFunction(MySource);
  //EnrollFieldDiffusivity(DiffusivityNonidealMHD);

  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
{

  AllocateRealUserMeshBlockDataField(3);
// List of User field:
// 0: temperature profile (1d array)
// 1: initial density profile (1d array)

  ruser_meshblock_data[0].NewAthenaArray(block_size.nx2+2*NGHOST);
  ruser_meshblock_data[1].NewAthenaArray(block_size.nx2+2*NGHOST);
  ruser_meshblock_data[2].NewAthenaArray(block_size.nx2+2*NGHOST);

// set temperature profile
  Real fac1    = 0.5*(HoRc+HoR0);
  Real fac2    = 0.5*(HoRc-HoR0);
  Real logfac1 = 0.5*(log(HoRc)+log(HoR0));
  Real logfac2 = 0.5*(log(HoRc)-log(HoR0));

  for (int j=js-NGHOST;j<=je+NGHOST;++j) {
    Real delta = fabs(pcoord->x2v(j)-0.5*PI);
    Real myHoR = fac1 + fac2 * tanh(2.0*(delta-theta_trans)/HoR0);
    //Real myHoR = exp(logfac1 + logfac2 * tanh((delta-theta_trans)/HoR0));
    ruser_meshblock_data[0](j) = SQR(myHoR);
  }

// set initial density profile
  int dlev = pmy_mesh->root_level+finest_lev-loc.level;
  int ifac = 1<<dlev;

  for (int j=js-NGHOST;j<=je+NGHOST;++j) {
    int j1 = ifac*((j-js)+loc.lx2*block_size.nx2);
    int j2 = j1+ifac-1;
    Real val = 1.0;
    for (int t=j1; t<=j2; ++t)
      val *= pmy_mesh->ruser_mesh_data[1](t);
    for (int t=0; t<dlev; ++t)
      val = sqrt(val);
    ruser_meshblock_data[1](j) = val;
  }

  //set values at physical boundary
  if (int(pbval->block_bcs[BoundaryFace::inner_x2])>0) {
    for (int j=1; j<=NGHOST; ++j) {
      ruser_meshblock_data[1](js-j) = ruser_meshblock_data[1](js+j-1);
    }
  }

  if (int(pbval->block_bcs[BoundaryFace::outer_x2])>0) {
    for (int j=1; j<=NGHOST; ++j) {
      ruser_meshblock_data[1](je+j) = ruser_meshblock_data[1](je-j+1);
    }
  }
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{

  // First, set magnetic field
  int nx1 = block_size.nx1+2*NGHOST+1;
  int nx2 = block_size.nx2+2*NGHOST+1;

  AthenaArray<Real> A3,area,len,len_p1;
  A3.NewAthenaArray(nx2,nx1);
  area.NewAthenaArray(nx1);
  len.NewAthenaArray(nx1);
  len_p1.NewAthenaArray(nx1);

  Real betat1  = pin->GetOrAddReal("problem", "betat1", 0.0);
  int fieldopt = pin->GetOrAddInteger("problem", "fieldopt", 1);

  if (MAGNETIC_FIELDS_ENABLED) {
    // Set values for Aphi (vector potential)
    if (fieldopt == 1) // add open poloidal field
    {
      Real Phimin = 2.0/(3.0-alpha)*Bz0*pow(RBmin/R0, 1.0-0.5*(alpha-1.0));

#pragma omp for schedule(static)
      for (int j=js; j<=je+1; ++j) {
        Real x2 = pcoord->x2f(j);
        Real sintheta = sin(x2);
        for (int i=is; i<=ie+1; ++i) {
          Real x1  = pcoord->x1f(i);
          Real R   = x1*sintheta;
          A3(j, i) = std::max(AphiOpen(x1, x2, 0.0)*R-Phimin, 0.0)/(fabs(R)+TINY_NUMBER);
        }
      }
    }

    if (fieldopt == 2) // add poloidal field loops
    {
      int nloop = pin->GetOrAddInteger("problem", "nloop",  2);
      Real R1   = pin->GetOrAddReal("problem",    "Rloop1", 5.0*R0);
      Real R2   = pin->GetOrAddReal("problem",    "Rloop2", 50.0*R0);

      for (int j=js; j<=je+1; ++j) {
        Real x2 = pcoord->x2f(j);
        for (int i=is; i<=ie+1; ++i) {
          Real x1  = pcoord->x1f(i);
          A3(j, i) = AphiLoop(x1, x2, 0.0, R1, R2, nloop);
        }
      }
    }

    if (fieldopt == 3) // No field
    {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          A3(j,i) = 0.0;
        }
      }
    }

    // Obtain poloidal B field
    for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
      for (int j=js; j<=je+1; ++j) {
        pcoord->Face2Area(k, j, is, ie, area);
        pcoord->Edge3Length(k, j, is, ie+1, len);
#pragma simd
        for (int i=is; i<=ie; ++i) {
          pfield->b.x2f(k, j, i) = -(len(i+1)*A3(j, i+1) - len(i)*A3(j, i))/(area(i)+TINY_NUMBER);
        }
      }
    }

    for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
      for (int j=js; j<=je; ++j) {
        pcoord->Face1Area(k, j, is, ie+1, area);
        pcoord->Edge3Length(k, j, is, ie+1, len);
        pcoord->Edge3Length(k, j+1, is, ie+1, len_p1);
#pragma simd
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k, j, i) = (len_p1(i)*A3(j+1, i) - len(i)*A3(j, i))/(area(i)+TINY_NUMBER);
        }
      }
    }

    // Set toroidal field
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real x1   = pcoord->x1v(i);
          Real rho  = rho0*pow(x1/R0, -alpha)*ruser_meshblock_data[1](j);
          Real pres = rho*(GM/x1)*ruser_meshblock_data[0](j);
          pfield->b.x3f(k, j, i) = sqrt(2.0*pres*betat1);
        }
      }
    }

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          const Real& b1_i   = pfield->b.x1f(k,   j,   i  );
          const Real& b1_ip1 = pfield->b.x1f(k,   j,   i+1);
          const Real& b2_j   = pfield->b.x2f(k,   j,   i);
          const Real& b2_jp1 = pfield->b.x2f(k,   j+1, i);
          const Real& b3_k   = pfield->b.x3f(k,   j,   i);
          const Real& b3_kp1 = pfield->b.x3f(k+1, j,   i);

          Real& bcc1 = pfield->bcc(IB1, k, j, i);
          Real& bcc2 = pfield->bcc(IB2, k, j, i);
          Real& bcc3 = pfield->bcc(IB3, k, j, i);

          const Real& x1f_i  = pcoord->x1f(i);
          const Real& x1f_ip = pcoord->x1f(i+1);
          const Real& x1v_i  = pcoord->x1v(i);
          const Real& dx1_i  = pcoord->dx1f(i);

          Real lw = (x1f_ip-x1v_i)/dx1_i;
          Real rw = (x1v_i -x1f_i)/dx1_i;
          bcc1    = lw*b1_i + rw*b1_ip1;

          const Real& x2f_j  = pcoord->x2f(j);
          const Real& x2f_jp = pcoord->x2f(j+1);
          const Real& x2v_j  = pcoord->x2v(j);
          const Real& dx2_j  = pcoord->dx2f(j);

          lw   = (x2f_jp-x2v_j)/dx2_j;
          rw   = (x2v_j -x2f_j)/dx2_j;
          bcc2 = lw*b2_j + rw*b2_jp1;

          const Real& x3f_k  = pcoord->x3f(k);
          const Real& x3f_kp = pcoord->x3f(k+1);
          const Real& x3v_k  = pcoord->x3v(k);
          const Real& dx3_k  = pcoord->dx3f(k);

          lw   = (x3f_kp-x3v_k)/dx3_k;
          rw   = (x3v_k -x3f_k)/dx3_k;
          bcc3 = lw*b3_k + rw*b3_kp1;
        }
      }
    }
  }

// Now, initialize hydro variables
  Real amp = pin->GetOrAddReal("problem", "amp", 0.05);
  int64_t iseed = gid;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      Real gc= ruser_meshblock_data[0](j); //temperature
      Real f = ruser_meshblock_data[1](j); //density
#pragma simd
      for (int i=is; i<=ie; ++i) {
        Real x1    = pcoord->x1v(i);
        Real myamp = x1 > Rbuf ? amp : 0.0;
        Real rho   = rho0*pow(x1/R0,-alpha)*f;
        Real cs    = sqrt((GM/x1)*gc);
        phydro->u(IDN, k, j, i) = rho;
        phydro->u(IM1, k, j, i) = rho*cs*myamp*(ran2(&iseed)-0.5);
        phydro->u(IM2, k, j, i) = rho*cs*myamp*(ran2(&iseed)-0.5);
        phydro->u(IM3, k, j, i) = rho*sqrt(GM*(1.0-(alpha+1.0)*gc)/x1);
        Real pressure = rho*SQR(cs);
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i)= pressure/(peos->GetGamma()-1.0)
            + 0.5*(SQR(phydro->u(IM1, k, j, i)) + SQR(phydro->u(IM2, k, j, i)) +
                   SQR(phydro->u(IM3, k, j, i)))/rho;
        if (MAGNETIC_FIELDS_ENABLED)
          phydro->u(IEN, k, j, i) +=  0.5*(SQR(pfield->bcc(IB1, k, j, i))+
            SQR(pfield->bcc(IB2, k, j, i))+SQR(pfield->bcc(IB3, k, j, i)));
      }

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma simd
          for (int i=is; i<=ie; ++i) {
            pdustfluids->df_u(rho_id, k, j, i) = initial_D2G[dust_id]*phydro->u(IDN, k, j, i);
            pdustfluids->df_u(v1_id,  k, j, i) = initial_D2G[dust_id]*phydro->u(IM1, k, j, i);
            pdustfluids->df_u(v2_id,  k, j, i) = initial_D2G[dust_id]*phydro->u(IM2, k, j, i);
            pdustfluids->df_u(v3_id,  k, j, i) = initial_D2G[dust_id]*phydro->u(IM3, k, j, i);
          }
				}
      }
    }
  }

  A3.DeleteAthenaArray();
  area.DeleteAthenaArray();
  len.DeleteAthenaArray();
  len_p1.DeleteAthenaArray();

  return;
}

namespace {
//--------------------------------------------------------------------------------------
//! \fn static Real CompressedX2(Real x, RegionSize rs)
//  \brief Increase the theta grid size towards the pole
Real CompressedX2(Real x, RegionSize rs)
{
  Real x2rat = fabs(rs.x2rat);
  Real x2mid = 0.5*PI;
  Real x2min = rs.x2min;
  Real x2max = rs.x2max;
  Real lw, rw;

  if (x <= 0.5) {
    if (x2rat == 1.0) {
      rw = (x*2.0), lw = 1.0-(x*2.0);
    } else {
      Real ratn = pow(x2rat, 0.5*rs.nx2);
      Real rnx  = pow(x2rat, x*rs.nx2);
      lw        = (rnx-ratn)/(1.0-ratn);
      rw        = 1.0-lw;
    }
    return x2min*lw+x2mid*rw;
  } else {
    if (x2rat == 1.0) {
      rw = (x-0.5)*2.0, lw = 1.0-(x-0.5)*2.0;
    } else {
      Real ratn = pow(1.0/x2rat, 0.5*rs.nx2);
      Real rnx  = pow(1.0/x2rat, (x-0.5)*rs.nx2);
      lw        = (rnx-ratn)/(1.0-ratn);
      rw        = 1.0-lw;
    }
    return x2mid*lw+x2max*rw;
  }
}

//--------------------------------------------------------------------------------------
//!\f User source term to set density floor and adjust temperature
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s)
{
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
    for (int j=js; j<=je; ++j) {
#pragma simd
      for (int i=is; i<=ie; ++i) {
        Real x1 = pmb->pcoord->x1v(i);
        cons(IDN, k, j, i) = std::max(cons(IDN, k, j, i), rho_floor*pow(x1/R0, -alpha));
      }
    }
  }

	if (NDUSTFLUIDS > 0) {
		for (int n=0; n<NDUSTFLUIDS; ++n) {
			int dust_id = n;
			int rho_id  = 4*dust_id;
			for (int k=ks; k<=ke; ++k) {
#pragma omp for schedule(static)
				for (int j=js; j<=je; ++j) {
#pragma simd
					for (int i=is; i<=ie; ++i) {
						Real x1 = pmb->pcoord->x1v(i);
						cons_df(rho_id, k, j, i) = std::max(cons_df(rho_id, k, j, i), initial_D2G[dust_id]*rho_floor*pow(x1/R0, -alpha));
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

  Real inv_sqrt_GM = 1.0/std::sqrt(GM);
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        Real sintheta = std::sin(pmb->pcoord->x2v(j));
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &st_time = stopping_time(dust_id, k, j, i);
          Real &x1 = pmb->pcoord->x1v(i);

          //Constant Stokes number in disk problems
          st_time = Stokes_number[dust_id]*std::pow(x1*sintheta, 1.5)*inv_sqrt_GM;
        }
      }
    }
  }
  return;
}


Real MyTimeStep(MeshBlock *pmb)
{
  return user_dt;
}

//----------------------------------------------------------------------------------------
//!\f: User-defined diffusivities for non-ideal MHD effexts
void DiffusivityNonidealMHD(FieldDiffusion *pfdif, MeshBlock *pmb,
                            const AthenaArray<Real> &w, const AthenaArray<Real> &bmag,
                            int is, int ie, int js, int je, int ks, int ke)
{
  Coordinates *pco = pmb->pcoord;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
        Real theta = pco->x2v(j);
        Real delta = fabs(pco->x2v(j)-0.5*PI);
        // resistivity near inner radial boundary for stabilization
        // midplane resistivity to stabilize current sheet
        Real fac_rin = 0.5*(1.0-tanh((delta-1.2)/HoR0));
        Real fac_mid = 0.5*(1.0-tanh((delta-0.5*HoR0)/HoR0));

        //avoid overstrong diffusion at midplane
        //fac_rin = fac_rin*std::min(1.0-fac_mid,1.0);

        if ((int(pmb->pbval->block_bcs[BoundaryFace::inner_x2]) > 0) && (j<pmb->js+NGHOST))
          fac_rin = 0.0;
        if ((int(pmb->pbval->block_bcs[BoundaryFace::outer_x2]) > 0) && (j>pmb->je-NGHOST))
          fac_rin = 0.0;

#pragma simd
        for (int i=is; i<=ie; i++) {
          Real x1 = pco->x1v(i);
          Real myfac = 0.5*(1.0-tanh((x1-Rbuf)/0.5));

          Real myeta_in  = eta_ofac_in*sqrt(GM*x1)*SQR(HoR0);
          Real myeta_mid = eta_ofac_mid*sqrt(GM*x1)*SQR(HoR0);

          pfdif->etaB(FieldDiffusion::DiffProcess::ohmic, k, j, i) = myfac*fac_rin*myeta_in;//+fac_mid*myeta_mid;
        }
      }
    }


//   if (pfdif->eta_ad != 0.0) { // ambipolar diffusivity is turned on
//     Real Am1    = 100.0;
//     Real logAm1 = log(Am1);
//     Real logAm0 = log(Am0);
//     for (int k=ks; k<=ke; k++) {
//       for (int j=js; j<=je; j++) {

//         Real sintheta = fabs(sin(pco->x2v(j)));
//         Real delta    = fabs(pco->x2v(j)-0.5*PI);
//         Real myAm     = exp(0.5*(logAm1+logAm0) + 0.5*(logAm1-logAm0)*tanh(2.0*(delta-theta_trans)/HoR0));

//         Real fac = 1.0;
//         if ((pmb->pbval->block_bcs[BoundaryFace::inner_x2] > 0) && (j<pmb->js+NGHOST))
//           fac = 0.0;
//         if ((pmb->pbval->block_bcs[BoundaryFace::outer_x2] > 0) && (j>pmb->je-NGHOST))
//           fac = 0.0;

// #pragma omp simd
//         for (int i=is; i<=ie; i++) {
//           Real rho  = w(IDN,k,j,i);
//           Real x1   = pco->x1v(i);
//           Real R    = x1*sintheta;
//           Real Omg  = sqrt(GM/(R*R*R));
//           Real myeta= SQR(bmag(k,j,i))/(myAm*rho*Omg);
//           pfdif->etaB(FieldDiffusion::DiffProcess::ambipolar, k,j,i) = fac*myeta;
//     }}}
//   }
// }


//   Ambipolar diffusion with Am = 1.0 for R < Rbuf
//
  if (pfdif->eta_ad != 0.0) { // ambipolar diffusivity is turned on
    Real Am1    = 100.0;
    Real logAm1 = log(Am1);
    Real Am_new;

    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        Real sintheta = fabs(sin(pco->x2v(j)));
        Real delta    = fabs(pco->x2v(j)-0.5*PI);

        Real fac = 1.0;
        if ((int(pmb->pbval->block_bcs[BoundaryFace::inner_x2]) > 0) && (j<pmb->js+NGHOST))
          fac = 0.0;
        if ((int(pmb->pbval->block_bcs[BoundaryFace::outer_x2]) > 0) && (j>pmb->je-NGHOST))
          fac = 0.0;
#pragma simd
        for (int i=is; i<=ie; i++) {
          Real x1 = pco->x1v(i);
          Real R  = x1*sintheta;

          if (ad_lin) {
            if (R<Rbuf)
              Am_new = Am_in;
            else if (R>4.0)
              Am_new = Am0;
            else
              Am_new = Am_in + (Am0-Am_in)*(R-Rbuf)/(4.0-Rbuf);
          }

          if (ad_log) {
            if (R<Rbuf)
              Am_new = Am_in;
            else if (R>4.0)
              Am_new = Am0;
            else
              Am_new = exp( log(Am_in) + ( log(Am0)-log(Am_in) )*(R-Rbuf)/(4.0-Rbuf) );
          }

          if (ad_hyp)
            Am_new = 0.5*(Am_in+Am0) + 0.5*(Am0-Am_in)*tanh((R-3.0)/0.5);

          Real logAm0 = log(Am_new);
          Real myAm   = exp(0.5*(logAm1+logAm0) + 0.5*(logAm1-logAm0)*tanh(2.0*(delta-theta_trans)/HoR0));
          Real rho    = w(IDN, k, j, i);
          Real Omg    = sqrt(GM/(R*R*R));
          Real myeta  = SQR(bmag(k,j,i))/(myAm*rho*Omg);
          pfdif->etaB(FieldDiffusion::DiffProcess::ambipolar, k, j, i) = fac*myeta;
        }
      }
    }
  }
}

//--------------------------------------------------------------------------------------
//! \fn static Real AphiOpen(const Real x1,const Real x2,const Real x3)
//  \brief Aphi: 3-component of vector potential for open poloidal fields
Real AphiOpen(const Real x1, const Real x2, const Real x3)
{
    Real R = fabs(x1*sin(x2));
    R      = std::max(R, 1.0e-12);
    Real theta = x2;
    if (theta<=1.0e-4 || theta>=PI-1.0e-4)
      theta = 1.0e-4;

    return 2.0/(3.0-alpha)*Bz0*pow(R/R0, -0.5*(alpha-1.0))/pow(1.0+1.0/SQR(tan(theta)*mu), 0.625);
    //return 2.0/(4.0-alpha-qT)*Bz0*R0*pow(R/R0,-0.5*(alpha+qT)+1.0)/pow(1.0+1.0/SQR(tan(x2)*mu),0.625);
}

//--------------------------------------------------------------------------------------
//! \fn static Real AphiLoop(const Real x1,const Real x2,const Real x3)
//  \brief Aphi: 3-component of vector potential for loop poloidal fields
Real AphiLoop(const Real x1, const Real x2, const Real x3, Real R1, Real R2, int nloop)
{
    Real R      = x1*sin(x2);
    Real k      = 2.0*PI*nloop/(R2-R1);
    Real dtheta = x2 - 0.5*PI;
    Real rho    = exp(-0.5*SQR(dtheta/HoR0));
    Real Q      = std::max(rho-0.2, 0.0);

    Real fac = R > R1 ? 1.0:0.0;
    fac      = R < R2 ? fac:0.0;

    return fac*A30*SQR(Q)*sin(k*(R-R1));
}

//--------------------------------------------------------------------------------------
//! \fn void AddPoloidalField(MeshBlock *pmb)
//  \brief Impose poloidal field in the middle of the simulation
void AddPoloidalField(MeshBlock *pmb)
{
    int is  = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie  = pmb->ie; int je = pmb->je; int ke = pmb->ke;
    int nx1 = pmb->block_size.nx1+2*NGHOST+1;
    int nx2 = pmb->block_size.nx2+2*NGHOST+1;
    Hydro *phydro = pmb->phydro;
    Field *pfd = pmb->pfield;

    AthenaArray<Real> A3,area,len,len_p1;
    A3.NewAthenaArray(nx2,nx1);
    area.NewAthenaArray(nx1);
    len.NewAthenaArray(nx1);
    len_p1.NewAthenaArray(nx1);

    int dk = NGHOST;
    if (pmb->block_size.nx3 == 1) dk=0;
    int djm = NGHOST;
    int djp = NGHOST;
    if (int(pmb->pbval->block_bcs[BoundaryFace::inner_x2]) > 0) djm = 0;
    if (int(pmb->pbval->block_bcs[BoundaryFace::outer_x2]) > 0) djp = 0;

    if (MAGNETIC_FIELDS_ENABLED) {
      Real Phimin = AphiOpen(RBmin, 0.5*PI, 0.0)*RBmin;
#pragma omp for schedule(static)
      for (int j=js-djm; j<=je+djp+1; ++j) {
        Real x2 = pmb->pcoord->x2f(j);
        Real sintheta = sin(x2);
        for (int i=is-NGHOST; i<=ie+NGHOST+1; ++i) {
          Real x1  = pmb->pcoord->x1f(i);
          Real R   = x1*sintheta;
          A3(j, i) = std::max(AphiOpen(x1, x2, 0.0)*R-Phimin, 0.0)/(fabs(R)+TINY_NUMBER);
        }
      }

      for (int k=ks-dk; k<=ke+dk; ++k) {
#pragma omp for schedule(static)
        for (int j=js-djm; j<=je+djp+1; ++j) {
          pmb->pcoord->Face2Area(k,   j, is-NGHOST, ie+NGHOST,   area);
          pmb->pcoord->Edge3Length(k, j, is-NGHOST, ie+NGHOST+1, len);
#pragma simd
          for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
            pfd->b.x2f(k, j, i) += -(len(i+1)*A3(j, i+1) - len(i)*A3(j, i))/(area(i)+TINY_NUMBER);
          }
        }
      }

      for (int k=ks-dk; k<=ke+dk; ++k) {
#pragma omp for schedule(static)
        for (int j=js-djm; j<=je+djp; ++j) {
          pmb->pcoord->Face1Area(k, j, is-NGHOST, ie+NGHOST+1, area);
          pmb->pcoord->Edge3Length(k, j,   is-NGHOST, ie+NGHOST+1, len);
          pmb->pcoord->Edge3Length(k, j+1, is-NGHOST, ie+NGHOST+1, len_p1);
#pragma simd
          for (int i=is-NGHOST; i<=ie+NGHOST+1; ++i) {
            pfd->b.x1f(k, j, i) += (len_p1(i)*A3(j+1, i) - len(i)*A3(j, i))/(area(i)+TINY_NUMBER);
          }
        }
      }

      for (int k=ks-dk; k<=ke+dk; k++) {
        for (int j=js-djm; j<=je+djp; j++) {
          for (int i=is-NGHOST; i<=ie+NGHOST; i++) {
            const Real& b1_i   = pfd->b.x1f(k, j,   i);
            const Real& b1_ip1 = pfd->b.x1f(k, j,   i+1);
            const Real& b2_j   = pfd->b.x2f(k, j,   i);
            const Real& b2_jp1 = pfd->b.x2f(k, j+1, i);

            Real& bcc1 = pfd->bcc(IB1, k, j, i);
            Real& bcc2 = pfd->bcc(IB2, k, j, i);
            Real  EB0  = 0.5*(SQR(bcc1)+SQR(bcc2));

            const Real& x1f_i  = pmb->pcoord->x1f(i);
            const Real& x1f_ip = pmb->pcoord->x1f(i+1);
            const Real& x1v_i  = pmb->pcoord->x1v(i);
            const Real& dx1_i  = pmb->pcoord->dx1f(i);
            Real lw = (x1f_ip-x1v_i)/dx1_i;
            Real rw = (x1v_i -x1f_i)/dx1_i;
            bcc1 = lw*b1_i + rw*b1_ip1;

            const Real& x2f_j  = pmb->pcoord->x2f(j);
            const Real& x2f_jp = pmb->pcoord->x2f(j+1);
            const Real& x2v_j  = pmb->pcoord->x2v(j);
            const Real& dx2_j  = pmb->pcoord->dx2f(j);
            lw = (x2f_jp-x2v_j)/dx2_j;
            rw = (x2v_j -x2f_j)/dx2_j;
            bcc2 = lw*b2_j + rw*b2_jp1;
            Real EB1 = 0.5*(SQR(bcc1)+SQR(bcc2));

            if (NON_BAROTROPIC_EOS)
              phydro->u(IEN, k, j, i) += EB1-EB0;
          }
        }
      }
    }

    A3.DeleteAthenaArray();
    area.DeleteAthenaArray();
    len.DeleteAthenaArray();
    len_p1.DeleteAthenaArray();
    return;
}
}
//--------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
//
void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh)
{
  Real x0  = pco->x1v(il);
  Real Omg = sqrt(GM/(R0*R0*R0));
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      Real gc = pmb->ruser_meshblock_data[0](j); //temperature
      Real f  = pmb->ruser_meshblock_data[1](j); //density
      Real sintheta = sin(pco->x2v(j));

      Real rhos = prim(IDN, k, j, il);
      Real v1s  = prim(IM1, k, j, il);
      Real v2s  = prim(IM2, k, j, il);
      Real v3s  = prim(IM3, k, j, il);
      Real Ts   = prim(IEN, k, j, il)/prim(IDN, k, j, il);
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        Real x   = pco->x1v(il-i);
        Real rho = rho0*pow(x/R0,-alpha)*f;
        Real cs2 = (GM/x)*gc;
        //prim(IDN,k,j,il-i) = rho;
        //prim(IM1,k,j,il-i) = 0.0;
        //prim(IM2,k,j,il-i) = 0.0;
        //prim(IM3,k,j,il-i) = sqrt(GM*(1.0-(alpha+1.0)*gc)/x);
        //if (NON_BAROTROPIC_EOS)
          //prim(IEN,k,j,il-i) = rho*cs2;

        prim(IDN, k, j, il-i) = rhos*pow(x/x0, -alpha); // rho; //
        prim(IM1, k, j, il-i) = 0.0;//v1s > 0.0 ? 0.0 : v1s;
        prim(IM2, k, j, il-i) = 0.0;//v2s;//*(x/x0);
        prim(IM3, k, j, il-i) = std::min(Omg*x*sintheta, sqrt(GM*(1.0-(alpha+1.0)*gc)/x));//v3s*sqrt(x0/x);
        prim(IEN, k, j, il-i) = prim(IDN, k, j, il-i)*cs2;//prim(IDN,k,j,il-i)*Ts*(x0/x); //
      }

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma simd
					for (int i=1; i<=ngh; ++i) {
						Real x   = pco->x1v(il-i);
						Real rho = rho0*pow(x/R0,-alpha)*f;

						prim_df(rho_id, k, j, il-i) = initial_D2G[dust_id]*prim(IDN, k, j, il-i); // rho; //
						prim_df(v1_id,  k, j, il-i) = 0.0;//v1s > 0.0 ? 0.0 : v1s;
						prim_df(v2_id,  k, j, il-i) = 0.0;//v2s;//*(x/x0);
						prim_df(v3_id,  k, j, il-i) = initial_D2G[dust_id]*prim(IM3, k, j, il-i);//v3s*sqrt(x0/x);
					}
				}
			}
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    x0 = pco->x1f(il);
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma simd
        for (int i=1; i<=ngh; ++i) {
          Real x = pco->x1f(il-i);
          b.x1f(k, j, (il-i)) = b.x1f(k, j, il)*SQR(x0/x);
        }
      }
    }

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k, j, (il-i)) = b.x2f(k, j, il);
        }
      }
    }

    x0 = pco->x1v(il);
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma simd
        for (int i=1; i<=ngh; ++i) {
          Real x = pco->x1v(il-i);
          b.x3f(k, j, (il-i)) = b.x3f(k, j, il)*(x0/x);
        }
      }
    }
  }
  return;
}


void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  Real x0 = pco->x1v(iu);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      Real rhoe = prim(IDN, k, j, iu);
      Real v1e  = prim(IM1, k, j, iu);
      Real v2e  = prim(IM2, k, j, iu);
      Real v3e  = prim(IM3, k, j, iu);
      Real Te   = prim(IEN, k, j, iu)/prim(IDN, k, j, iu);
#pragma simd
      for (int i=1; i<=ngh; ++i) {
        Real x = pco->x1v(iu+i);

        prim(IDN, k, j, iu+i) = rhoe*pow(x/x0, -alpha);
        prim(IM1, k, j, iu+i) = v1e < 0.0 ? 0.0 : v1e;
        prim(IM2, k, j, iu+i) = v2e;
        prim(IM3, k, j, iu+i) = v3e*sqrt(x0/x);
        if (NON_BAROTROPIC_EOS)
          prim(IEN, k, j, iu+i) = prim(IDN, k, j, iu+i)*Te*(x0/x);
      }

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma simd
					for (int i=1; i<=ngh; ++i) {
						Real x = pco->x1v(iu+i);

						prim_df(rho_id, k, j, iu+i) = initial_D2G[dust_id]*rhoe*pow(x/x0, -alpha);
						prim_df(v1_id,  k, j, iu+i) = v1e < 0.0 ? 0.0 : v1e;
						prim_df(v2_id,  k, j, iu+i) = v2e;
						prim_df(v3_id,  k, j, iu+i) = v3e*sqrt(x0/x);
					}
				}
			}
    }
  }

  if (MAGNETIC_FIELDS_ENABLED) {
    x0 = pco->x1f(iu+1);
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma simd
        for (int i=1; i<=(NGHOST); ++i) {
          Real x = pco->x1f(iu+i+1);
          b.x1f(k, j, (iu+i+1)) = b.x1f(k, j, (iu+1))*SQR(x0/x);
        }
      }
    }
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
#pragma simd
        for (int i=1; i<=ngh; ++i) {
          b.x2f(k, j, (iu+i)) = b.x2f(k, j, iu);
        }
      }
    }
    x0 = pco->x1v(iu);
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma simd
        for (int i=1; i<=ngh; ++i) {
          Real x = pco->x1v(iu+i);
          b.x3f(k, j, (iu+i)) = b.x3f(k, j, iu)*(x0/x);
        }
      }
    }
  }
  return;
}


void DiskInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real y0  = pco->x2v(jl);
  Real Omg = sqrt(GM/(R0*R0*R0));

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      Real gc = pmb->ruser_meshblock_data[0](jl-j); //temperature
      Real f  = pmb->ruser_meshblock_data[1](jl-j); //density
      Real sintheta = sin(pco->x2v(jl-j));

      for (int i=il; i<=iu; ++i) {
        Real x   = pco->x1v(i);
        Real rho = rho0*pow(x/R0,-alpha)*f;
        Real cs2 = (GM/x)*gc;

        Real rhos = prim(IDN, k, jl, i);
        Real v1s  = prim(IM1, k, jl, i);
        Real v2s  = prim(IM2, k, jl, i);
        Real v3s  = prim(IM3, k, jl, i);
        Real Ts   = prim(IEN, k, jl, i)/prim(IDN, k, jl, i);

        Real y = pco->x2v(jl-j);
        prim(IDN, k, jl-j, i) = rho; //rhos*pow(y/y0,1.0);
        prim(IM1, k, jl-j, i) = v1s;
        prim(IM2, k, jl-j, i) = v2s<0 ? v2s:0;
        prim(IM3, k, jl-j, i) = std::min(Omg*x*sintheta, sqrt(GM*(1.0-(alpha+1.0)*gc)/x));
        if (NON_BAROTROPIC_EOS)
          prim(IEN, k, jl-j, i) = prim(IDN, k, jl-j, i)*cs2;
      }

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
					for (int i=il; i<=iu; ++i) {
						Real x   = pco->x1v(i);
						Real rho = rho0*pow(x/R0,-alpha)*f;
						Real cs2 = (GM/x)*gc;

						Real rhos = prim(IDN, k, jl, i);
						Real v1s  = prim(IM1, k, jl, i);
						Real v2s  = prim(IM2, k, jl, i);
						Real v3s  = prim(IM3, k, jl, i);
						Real Ts   = prim(IEN, k, jl, i)/prim(IDN, k, jl, i);

						Real y = pco->x2v(jl-j);
						prim_df(rho_id, k, jl-j, i) = initial_D2G[dust_id]*prim(IDN, k, jl-j, i); //rhos*pow(y/y0,1.0);
						prim_df(v1_id,  k, jl-j, i) = initial_D2G[dust_id]*prim(IM1, k, jl-j, i);
						prim_df(v2_id,  k, jl-j, i) = initial_D2G[dust_id]*prim(IM2, k, jl-j, i);
						prim_df(v3_id,  k, jl-j, i) = initial_D2G[dust_id]*prim(IM3, k, jl-j, i);
					}
				}
			}
    }
  }
  return;
}


void DiskOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                  AthenaArray<Real> &prim_df, FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
  Real y0  = pco->x2v(ju);
  Real Omg = sqrt(GM/(R0*R0*R0));

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      Real gc = pmb->ruser_meshblock_data[0](ju+j); //temperature
      Real f  = pmb->ruser_meshblock_data[1](ju+j); //density
      Real sintheta = sin(pco->x2v(ju+j));

      for (int i=il; i<=iu; ++i) {
        Real x   = pco->x1v(i);
        Real rho = rho0*pow(x/R0, -alpha)*f;
        Real cs2 = (GM/x)*gc;

        Real rhoe = prim(IDN, k, ju, i);
        Real v1e  = prim(IM1, k, ju, i);
        Real v2e  = prim(IM2, k, ju, i);
        Real v3e  = prim(IM3, k, ju, i);
        Real Te   = prim(IEN, k, ju, i)/prim(IDN, k, ju, i);

        Real y = pco->x2v(ju+j);
        prim(IDN, k, ju+j, i) = rho; //rhos*pow(y/y0,1.0);
        prim(IM1, k, ju+j, i) = v1e;
        prim(IM2, k, ju+j, i) = v2e>0 ? v2e:0;
        prim(IM3, k, ju+j, i) = std::min(Omg*x*sintheta, sqrt(GM*(1.0-(alpha+1.0)*gc)/x));
        if (NON_BAROTROPIC_EOS)
          prim(IEN, k, ju+j, i) = prim(IDN, k, ju+j, i)*cs2;
      }

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
					for (int i=il; i<=iu; ++i) {
						Real x   = pco->x1v(i);
						Real rho = rho0*pow(x/R0, -alpha)*f;
						Real cs2 = (GM/x)*gc;

						Real rhoe = prim_df(rho_id, k, ju, i);
						Real v1e  = prim_df(v1_id,  k, ju, i);
						Real v2e  = prim_df(v2_id,  k, ju, i);
						Real v3e  = prim_df(v3_id,  k, ju, i);

						Real y = pco->x2v(ju+j);
						prim_df(rho_id, k, ju+j, i) = initial_D2G[dust_id]*rho; //rhos*pow(y/y0,1.0);
						prim_df(v1_id,  k, ju+j, i) = v1e;
						prim_df(v2_id,  k, ju+j, i) = v2e>0 ? v2e:0;
						prim_df(v3_id,  k, ju+j, i) = std::min(Omg*x*sintheta, sqrt(GM*(1.0-(alpha+1.0)*gc)/x));
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
        Real gc = pmb->ruser_meshblock_data[0](j); //temperature
        Real f  = pmb->ruser_meshblock_data[1](j); //density
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &dust_rho  = prim_df(rho_id, k, j, i);
          Real &dust_vel1 = prim_df(v1_id,  k, j, i);
          Real &dust_vel2 = prim_df(v2_id,  k, j, i);
          Real &dust_vel3 = prim_df(v3_id,  k, j, i);

          Real &dust_dens = cons_df(rho_id, k, j, i);
          Real &dust_mom1 = cons_df(v1_id,  k, j, i);
          Real &dust_mom2 = cons_df(v2_id,  k, j, i);
          Real &dust_mom3 = cons_df(v3_id,  k, j, i);

          Real x1  = pmb->pcoord->x1v(i);
          Real rho = rho0*pow(x1/R0, -alpha)*f;
          Real cs  = sqrt((GM/x1)*gc);

          prim_df(rho_id, k, j, i) = initial_D2G[dust_id]*rho;
          prim_df(v1_id,  k, j, i) = 0.0;
          prim_df(v2_id,  k, j, i) = 0.0;
          prim_df(v3_id,  k, j, i) = sqrt(GM*(1.0-(alpha+1.0)*gc)/x1);

          cons_df(rho_id, k, j, i) = initial_D2G[dust_id]*rho;
          cons_df(v1_id,  k, j, i) = initial_D2G[dust_id]*rho*0.0;
          cons_df(v2_id,  k, j, i) = initial_D2G[dust_id]*rho*0.0;
          cons_df(v3_id,  k, j, i) = initial_D2G[dust_id]*rho*sqrt(GM*(1.0-(alpha+1.0)*gc)/x1);
        }
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//!\f: Userwork in loop: apply external B field and handle user-defined output
//
void MeshBlock::UserWorkInLoop()
{
  Real time  = pmy_mesh->time;
  Real dt    = pmy_mesh->dt;
  Real mygam = peos->GetGamma();
  Real gam1  = 1.0/(mygam-1.0);
  int dk     = NGHOST;
  if (block_size.nx3 == 1) dk = 0;

  int kl = ks - dk;     int ku = ke + dk;
  int jl = js - NGHOST; int ju = je + NGHOST;
  int il = is - NGHOST; int iu = ie + NGHOST;

// now adjust temperature
  for (int k=ks-dk; k<=ke+dk; ++k) {
    for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
      Real sintheta = fabs(sin(pcoord->x2v(j)));
      Real delta    = fabs(pcoord->x2v(j)-0.5*PI);
#pragma simd
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        Real x1 = pcoord->x1v(i);
        Real R  = x1*sintheta;

        //// apply cooling for t>iso_time
        //Real frac_t= std::max((time/6.28-iso_time)/10.0 ,0.0);
        //Real fcool = fcool_t1 + std::min(frac_t,1.0)*(fcool_t2-fcool_t1);

        Real tau0    = 1.0/fcool;
        Real tau1    = 1.0e-10;   //isothermal
        Real logtau0 = log(tau0);
        Real logtau1 = log(tau1);
        //Real tau_multT = 0.5*(tau1+tau0)+0.5*(tau1-tau0)*tanh(2.0*(delta-theta_trans)/HoR0);
        Real tau_multT = exp(0.5*(logtau1+logtau0)+0.5*(logtau1-logtau0)*tanh(2.0*(delta-theta_trans)/HoR0));

        Real OmgK = sqrt(GM/(R*R*R));
        Real dtoverP = (OmgK/2.0/PI)*dt;
        Real facR = std::max(std::min(2.0*(Rbuf-R),1.0),0.0);

        // density and velocity adjustment
        Real rho    = phydro->w(IDN, k, j, i);
        Real vr     = phydro->w(IM1, k, j, i);
        Real vtheta = phydro->w(IM2, k, j, i);
        if (x1 <= Rbuf) {
          Real dvr    = (0.0-vr)*(std::min(dtoverP/0.01,1.0));
          Real dvt    = (0.0-vtheta)*(std::min(dtoverP/0.01,1.0));
          Real vr     = vr+dvr;
          Real vtheta = vtheta+dvt;
        }

        Real vphi = phydro->w(IM3, k, j, i);
        Real M1   = rho*vr;
        Real M2   = rho*vtheta;
        Real M3   = rho*vphi;

        // temperature adjustment
        Real Tw  = phydro->w(IEN, k, j, i)/rho;
        Real myT = GM/x1*ruser_meshblock_data[0](j);
        Real dT  = (myT-Tw)*(std::min(dtoverP/tau_multT, 1.0));

        Real newT = Tw+dT;
        newT = std::min(std::max(newT, 0.2*myT), 5.0*myT);

        // apply density floor and Alfven limiter
        Real rhonew0 = std::max(rho, rho_floor*pow(x1/R0, -alpha));
        Real rhonew  = rhonew0;
        phydro->u(IEN, k, j, i) = 0.0;
        if (MAGNETIC_FIELDS_ENABLED) {
          Real b1    = pfield->bcc(IB1, k, j, i);
          Real b2    = pfield->bcc(IB2, k, j, i);
          Real b3    = pfield->bcc(IB3, k, j, i);
          Real bsq   = SQR(b1)+SQR(b2)+SQR(b3);
          Real vA2   = bsq/rhonew0;
          Real enhac = vA2 < SQR(2.0*x1) ? 1.0 : vA2/SQR(2.0*x1);
          rhonew = rhonew0*(1.0+(enhac-1.0)*std::min(facR, 1.0));
          phydro->u(IEN, k, j, i) += 0.5*bsq;
        }

        // update primitive and conserved variables
        phydro->w(IDN, k, j, i) = rhonew;
        phydro->u(IDN, k, j, i) = rhonew;
        phydro->w(IM1, k, j, i) = M1/rhonew;
        phydro->u(IM1, k, j, i) = M1;
        phydro->w(IM2, k, j, i) = M2/rhonew;
        phydro->u(IM2, k, j, i) = M2;
        phydro->w(IM3, k, j, i) = M3/rhonew;
        phydro->u(IM3, k, j, i) = M3;

        Real Ek = 0.5*(SQR(M1)+SQR(M2)+SQR(M3))/rhonew;
        phydro->w(IEN, k, j, i) = rhonew0*newT;  // internal energy unchanged
        phydro->u(IEN, k, j, i)+= Ek+phydro->w(IEN, k, j, i)*gam1;
      }
    }
  }

 //Apply poloidal field (one shot)
  if (MAGNETIC_FIELDS_ENABLED) {
    if ((time+dt >= taddBp) && (time < taddBp)) {
      AddPoloidalField(this);
    }
  }

  //if ((NDUSTFLUIDS > 0) && (time_drag != 0.0) && (time < time_drag))
    //FixedDust(this, il, iu, jl, ju, kl, ku, pdustfluids->df_w, pdustfluids->df_u);

 return;
}
