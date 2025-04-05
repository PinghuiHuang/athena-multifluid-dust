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
#include "../parameter_input.hpp"

// problem parameters which are useful to make global to this file
namespace {
Real v0, t0, x0, user_dt, iso_cs, gamma_gas;
Real MyTimeStep(MeshBlock *pmb);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  user_dt   = pin->GetOrAddReal("problem", "user_dt", 1e-1);
  iso_cs    = pin->GetOrAddReal("hydro", "iso_sound_speed", 1e-1);
  gamma_gas = pin->GetReal("hydro","gamma");
  EnrollUserTimeStepFunction(MyTimeStep);
  return;
}


namespace {
Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes viscous shear flow.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  //Real v1=0.0, v2=0.0, v3=0.0;
  //Real d0 = 1.0; // p0=1.0;
  Real x1, x2, x3; // x2 and x3 are set but unused

  //  Initialize density and momenta in Cartesian grids
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        x1=pcoord->x1v(i);
        x2=pcoord->x2v(j);
        x3=pcoord->x3v(k);

        if (NDUSTFLUIDS == 1) {
          //Test 1: gas and 1 dust fludis, NDUSTFLUIDS == 1
          phydro->u(IDN, k, j, i) = 0.2;
          phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i)*1.0;
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = 0.0;

          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i)  = SQR(iso_cs)*phydro->u(IDN, k, j, i)/(gamma_gas - 1.0);
            phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                          + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
          }

          pdustfluids->df_u(0, k, j, i) = 1.0;
          pdustfluids->df_u(1, k, j, i) = pdustfluids->df_u(0, k, j, i)*2.0;
          pdustfluids->df_u(2, k, j, i) = 0.0;
          pdustfluids->df_u(3, k, j, i) = 0.0;
        }

        if (NDUSTFLUIDS == 2) {
          // Test 2: gas and 2 dust fludis, NDUSTFLUIDS == 2
          phydro->u(IDN, k, j, i) = 0.2;
          phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i)*1.0;
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = 0.0;

          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i)  = SQR(iso_cs)*phydro->u(IDN, k, j, i)/(gamma_gas - 1.0);
            phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                          + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
          }

          pdustfluids->df_u(0, k, j, i) = 1.0;
          pdustfluids->df_u(1, k, j, i) = pdustfluids->df_u(0, k, j, i)*2.0;
          pdustfluids->df_u(2, k, j, i) = 0.0;
          pdustfluids->df_u(3, k, j, i) = 0.0;

          pdustfluids->df_u(4, k, j, i) = 1.8;
          pdustfluids->df_u(5, k, j, i) = pdustfluids->df_u(4, k, j, i)*3.0;
          pdustfluids->df_u(6, k, j, i) = 0.0;
          pdustfluids->df_u(7, k, j, i) = 0.0;
        }

        if (NDUSTFLUIDS == 5) {
          // Test 3: gas and 5 dust fludis, NDUSTFLUIDS == 5
          phydro->u(IDN, k, j, i) = 1.;
          phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i)*-1.0;
          phydro->u(IM2, k, j, i) = 0.0;
          phydro->u(IM3, k, j, i) = 0.0;

          if (NON_BAROTROPIC_EOS) {
            phydro->u(IEN, k, j, i)  = SQR(iso_cs)*phydro->u(IDN, k, j, i)/(gamma_gas - 1.0);
            phydro->u(IEN, k, j, i) += 0.5*(SQR(phydro->u(IM1, k, j, i))+SQR(phydro->u(IM2, k, j, i))
                                          + SQR(phydro->u(IM3, k, j, i)))/phydro->u(IDN, k, j, i);
          }

          pdustfluids->df_u(0,  k, j, i)  = 1.5;
          pdustfluids->df_u(1,  k, j, i)  = pdustfluids->df_u(0, k, j, i)*2.0;
          pdustfluids->df_u(2,  k, j, i)  = 0.0;
          pdustfluids->df_u(3,  k, j, i)  = 0.0;

          pdustfluids->df_u(4,  k, j, i)  = 2.0;
          pdustfluids->df_u(5,  k, j, i)  = pdustfluids->df_u(4, k, j, i)*3.1;
          pdustfluids->df_u(6,  k, j, i)  = 0.0;
          pdustfluids->df_u(7,  k, j, i)  = 0.0;

          pdustfluids->df_u(8,  k, j, i)  = 2.5;
          pdustfluids->df_u(9,  k, j, i)  = pdustfluids->df_u(8, k, j, i)*-2.5;
          pdustfluids->df_u(10, k, j, i) = 0.0;
          pdustfluids->df_u(11, k, j, i) = 0.0;

          pdustfluids->df_u(12, k, j, i) = 3.0;
          pdustfluids->df_u(13, k, j, i) = pdustfluids->df_u(12, k, j, i)*0.5;
          pdustfluids->df_u(14, k, j, i) = 0.0;
          pdustfluids->df_u(15, k, j, i) = 0.0;

          pdustfluids->df_u(16, k, j, i) = 3.5;
          pdustfluids->df_u(17, k, j, i) = pdustfluids->df_u(16, k, j, i)*-4.1;
          pdustfluids->df_u(18, k, j, i) = 0.0;
          pdustfluids->df_u(19, k, j, i) = 0.0;
        }

      }
    }
  }
  return;
}
