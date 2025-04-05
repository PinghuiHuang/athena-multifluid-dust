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

#if (NDUSTFLUIDS!=2)
#error "This problem generator requires NDUSTFLUIDS == 2!"
#endif

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

        if (NDUSTFLUIDS == 2) {
          // Test 2: gas and 2 dust fludis, NDUSTFLUIDS == 2
          Real &gas_den  = phydro->u(IDN,k,j,i);
          Real &gas_mom1 = phydro->u(IM1,k,j,i);
          Real &gas_mom2 = phydro->u(IM2,k,j,i);
          Real &gas_mom3 = phydro->u(IM3,k,j,i);

          gas_den  = 1.0;
          gas_mom1 = 2.0;
          gas_mom2 = 0.0;
          gas_mom3 = 0.0;

          if (NON_BAROTROPIC_EOS) {
            Real &gas_erg  = phydro->u(IEN,k,j,i);
            gas_erg        = SQR(iso_cs)*gas_den/(gamma_gas - 1.0);
            gas_erg       += 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))/gas_den;
          }

          Real &dust_1_den  = pdustfluids->df_u(0, k, j, i);
          Real &dust_1_mom1 = pdustfluids->df_u(1, k, j, i);
          Real &dust_1_mom2 = pdustfluids->df_u(2, k, j, i);
          Real &dust_1_mom3 = pdustfluids->df_u(3, k, j, i);

          Real &dust_2_den  = pdustfluids->df_u(4, k, j, i);
          Real &dust_2_mom1 = pdustfluids->df_u(5, k, j, i);
          Real &dust_2_mom2 = pdustfluids->df_u(6, k, j, i);
          Real &dust_2_mom3 = pdustfluids->df_u(7, k, j, i);

          dust_1_den  = 0.5;
          dust_1_mom1 = 3.0;
          dust_1_mom2 = 0.0;
          dust_1_mom3 = 0.0;

          dust_2_den  = 2.0;
          dust_2_mom1 = 0.8;
          dust_2_mom2 = 0.0;
          dust_2_mom3 = 0.0;
        }

      }
    }
  }
  return;
}
