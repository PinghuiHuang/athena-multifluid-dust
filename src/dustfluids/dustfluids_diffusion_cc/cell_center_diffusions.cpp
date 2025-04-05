//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cell_center_diffusions.cpp
//! \brief implementation of functions in class DustFluidsCellCenterDiffusion

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
#include "../../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "../dustfluids_diffusion/dustfluids_diffusion.hpp"
#include "cell_center_diffusions.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

class DustFluids;
class DustFluidsDiffusion;


DustFluidsCellCenterDiffusion::DustFluidsCellCenterDiffusion(MeshBlock *pmb,
        DustFluids *pdf, ParameterInput *pin) :
  diff_mom_cc(NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  coarse_diff_mom_cc_(NDUSTVARS, pmb->ncc3, pmb->ncc2, pmb->ncc1,
          (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
          AthenaArray<Real>::DataStatus::empty)),
  diff_cc_flux{{NDUSTVARS, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
          {NDUSTVARS, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
          (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
           AthenaArray<Real>::DataStatus::empty)},
          {NDUSTVARS, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
          (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
           AthenaArray<Real>::DataStatus::empty)}},
  pmy_dustfluids_(pdf), pmb_(pmb), pco_(pmb->pcoord),
  diffccbvar(pmb, &diff_mom_cc, &coarse_diff_mom_cc_, diff_cc_flux,
              DustDiffusionBoundaryQuantity::cons_diff) {

  Mesh *pm = pmb->pmy_mesh;

  if (pdf->dfdif.dustfluids_diffusion_defined) {
    pmb->RegisterMeshBlockData(diff_mom_cc);

    // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
    if (pm->multilevel)
      refinement_idx = pmb->pmr->AddToRefinement(&diff_mom_cc, &coarse_diff_mom_cc_);

    // Enroll DustDiffusionBoundaryQuantity object
    diffccbvar.bvar_index = pmb->pbval->bvars.size();
    pmb->pbval->bvars.push_back(&diffccbvar);
    pmb->pbval->bvars_main_int.push_back(&diffccbvar);

    if (STS_ENABLED)
      pmb->pbval->bvars_sts.push_back(&diffccbvar);
  }

}
