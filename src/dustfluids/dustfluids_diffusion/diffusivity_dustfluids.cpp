//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusivity_dustfluids.cpp
//! \brief Compute diffusivities of dust fluids.

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <cstring>    // strcmp
#include <limits>
#include <sstream>
#include <string>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../defs.hpp"
#include "../../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

class Hydro;
class HydroDiffusion;


void DustFluidsDiffusion::ConstantDustDiffusivity(DustFluids *pdf,
    MeshBlock *pmb, AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust,
    int is, int ie, int js, int je, int ks, int ke) {

  Real inv_eddy_time = 1.0/eddy_time_;
  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          Real &diffusivity = nu_dust(dust_id, k, j, i);
          diffusivity       = pdf->const_nu_dust[dust_id];

          Real &soundspeed  = cs_dust(dust_id, k, j, i);
          soundspeed        = std::sqrt(diffusivity*inv_eddy_time);
        }
      }
    }
  }
  return;
}


void DustFluidsDiffusion::ZeroDustDiffusivity(AthenaArray<Real> &dust_diffusivity,
    AthenaArray<Real> &dust_cs) {

  dust_diffusivity.ZeroClear();
  dust_cs.ZeroClear();

  return;
}
