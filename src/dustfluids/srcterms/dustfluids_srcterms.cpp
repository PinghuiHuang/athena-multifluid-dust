//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_srcterms.cpp
//! \brief Class to implement source terms in the dust fluids equations

// C headers

// C++ headers
#include <cstring>    // strcmp
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../../orbital_advection/orbital_advection.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_srcterms.hpp"

//! DustFLuidsSourceTerms constructor

DustFluidsSourceTerms::DustFluidsSourceTerms(DustFluids *pdf, ParameterInput *pin) {
  pmy_dustfluids_ = pdf;
  dustfluids_sourceterms_defined = false;

  // read point mass or constant acceleration parameters from input block

  // set the point source only when the coordinate is spherical or 2D
  // It works even for cylindrical with the orbital advection.
  flag_point_mass_ = false;
  gm_ = pin->GetOrAddReal("problem","GM",0.0);
  bool orbital_advection_defined
         = (pin->GetOrAddInteger("orbital_advection","OAorder",0)!=0)?
           true : false;
  if (gm_ != 0.0) {
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") != 0
        && std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in DustFluidsSourceTerms constructor" << std::endl
          << "The point mass gravity works only in the cylindrical and "
          << "spherical polar coordinates." << std::endl
          << "Check <problem> GM parameter in the input file." << std::endl;
      ATHENA_ERROR(msg);
    }
    if (orbital_advection_defined) {
      dustfluids_sourceterms_defined = true;
    } else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0
               && pdf->pmy_block->block_size.nx3>1) {
      std::stringstream msg;
      msg << "### FATAL ERROR in DustFluidsSourceTerms constructor" << std::endl
          << "The point mass gravity deos not work in the 3D cylindrical "
          << "coordinates without orbital advection." << std::endl
          << "Check <problem> GM parameter in the input file." << std::endl;
      ATHENA_ERROR(msg);
    } else {
      flag_point_mass_ = true;
      dustfluids_sourceterms_defined = true;
    }
  }

  g1_ = pin->GetOrAddReal("dust", "grav_acc1", 0.0);
  if (g1_ != 0.0) dustfluids_sourceterms_defined = true;

  g2_ = pin->GetOrAddReal("dust", "grav_acc2", 0.0);
  if (g2_ != 0.0) dustfluids_sourceterms_defined = true;

  g3_ = pin->GetOrAddReal("dust", "grav_acc3", 0.0);
  if (g3_ != 0.0) dustfluids_sourceterms_defined = true;


  // read shearing box parameters from input block
  Omega_0_    = pin->GetOrAddReal("orbital_advection",    "Omega0",     0.0);
  qshear_     = pin->GetOrAddReal("orbital_advection",    "qshear",     0.0);
  ShBoxCoord_ = pin->GetOrAddInteger("orbital_advection", "shboxcoord", 1);

  // check flag for shearing source
  flag_shearing_source_ = 0;
  if(orbital_advection_defined) { // orbital advection source terms
    if(ShBoxCoord_ == 1) {
      flag_shearing_source_ = 1;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in DustFluidsSourceTerms constructor" << std::endl
          << "OrbitalAdvection does NOT work with shboxcoord = 2." << std::endl
          << "Check <orbital_advection> shboxcoord parameter in the input file."
          << std::endl;
      ATHENA_ERROR(msg);
    }
  } else if ((Omega_0_ !=0.0) && (qshear_ != 0.0)
             && std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0) {
    flag_shearing_source_ = 2; // shearing box source terms
  } else if ((Omega_0_ != 0.0) &&
             (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0
              || std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0)) {
    flag_shearing_source_ = 3; // rotating system source terms
  }

  if (flag_shearing_source_ != 0)
    dustfluids_sourceterms_defined = true;

  if (SELF_GRAVITY_ENABLED)
    dustfluids_sourceterms_defined = true;

  UserSourceTerm = pmy_dustfluids_->pmy_block->pmy_mesh->UserSourceTerm_;
  if (UserSourceTerm != nullptr)
    dustfluids_sourceterms_defined = true;

  // scratch array for polar averaging
  if ((std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) && (pdf->pmy_block->block_size.nx3>1)){
    int ncells1 = pdf->pmy_block->block_size.nx1 + 2*NGHOST;
    int ncells3 = pdf->pmy_block->block_size.nx3 + 2*NGHOST;
    dust_avg_.NewAthenaArray(NDUSTVARS, ncells3, ncells1);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DustFluidsSourceTerms::AddDustFluidsSourceTerms
//! \brief Adds source terms to conserved variables

void DustFluidsSourceTerms::AddDustFluidsSourceTerms(const Real time, const Real dt,
                     const AthenaArray<Real> *flux_df, const AthenaArray<Real> &prim_df,
                     AthenaArray<Real> &cons_df) {
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;

  bool polar_inner = (pmb->pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar"));
  bool polar_outer = (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar"));

  bool polar_wedge_inner = (pmb->pbval->block_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("polar_wedge"));
  bool polar_wedge_outer = (pmb->pbval->block_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("polar_wedge"));

  // accleration due to point mass (MUST BE AT ORIGIN)
  if (flag_point_mass_)
    PointMassDustFluids(dt, flux_df, prim_df, cons_df);

  // constant acceleration (e.g. for RT instability)
  if (g1_ != 0.0 || g2_ != 0.0 || g3_ != 0.0)
    ConstantAccelerationDustFluids(dt, flux_df, prim_df, cons_df);

  // Add new source terms here
  if (SELF_GRAVITY_ENABLED)
    SelfGravityDustFluids(dt, flux_df, prim_df, cons_df);

  // Sorce terms for orbital advection, shearing box, or rotating system
  if (flag_shearing_source_ == 1)
    OrbitalAdvectionSourceTermsDustFluids(dt, flux_df, prim_df, cons_df);
  else if (flag_shearing_source_ == 2)
    ShearingBoxSourceTermsDustFluids(dt, flux_df, prim_df, cons_df);
  else if (flag_shearing_source_ == 3)
    RotatingSystemSourceTermsDustFluids(dt, flux_df, prim_df, cons_df);

  // polar averaging
  if ((std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) && (pmb->block_size.nx3 > 1)) {
    if ((polar_inner || polar_wedge_inner)) {
      PolarAveragingDustFluids(cons_df, pmb->js,   4);
      PolarAveragingDustFluids(cons_df, pmb->js+1, 2);
    }
    if ((polar_outer || polar_wedge_outer)) {
      PolarAveragingDustFluids(cons_df, pmb->je,   4);
      PolarAveragingDustFluids(cons_df, pmb->je-1, 2);
    }
  }

  return;
}


void DustFluidsSourceTerms::PolarAveragingDustFluids(AthenaArray<Real> &cons_df,
                                                      int j, int nlayer) {

  MeshBlock *pmb = pmy_dustfluids_->pmy_block;
  int is = pmb->is; int ks = pmb->ks;
  int ie = pmb->ie; int ke = pmb->ke;
  Real fac = 1.0/SQR(nlayer);

  for (int n=0; n<NDUSTVARS; ++n)
    for (int k=ks; k<=ke; ++k)
#pragma omp simd
      for (int i=is; i<=ie; ++i)
        dust_avg_(n, k, i) = 0.0;

  for (int k=ks; k<=ke; ++k) {
    for (int l=-nlayer+1; l<=nlayer-1; ++l) {
      int myk = k+l;
      Real wght = (nlayer-fabs(l))*fac;
      myk = myk <= ke ? myk : myk-pmb->block_size.nx3;
      myk = myk >= ks ? myk : myk+pmb->block_size.nx3;
      for (int n=0; n<NDUSTVARS; ++n) {
#pragma omp simd
        for (int i=is; i<=ie; ++i)
          dust_avg_(n, k, i) += cons_df(n, myk, j, i)*wght;
      }
    }
  }

  for (int n=0; n<NDUSTVARS; ++n) {
    for (int k=ks; k<=ke; ++k) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        cons_df(n, k, j, i) = dust_avg_(n, k, i);
      }
    }
  }

  return;
}
