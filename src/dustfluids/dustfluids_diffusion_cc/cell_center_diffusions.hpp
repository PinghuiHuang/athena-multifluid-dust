#ifndef CELLCENTER_DIFFUSION_HPP_
#define CELLCENTER_DIFFUSION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cell_center_diffusions.hpp
//! \brief defines class DustFluidsCellCenterDiffusion
//! Contains data and functions in class DustFluidsCellCenterDiffusion

// C headers

// C++ headers
#include <cstring>    // strcmp
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../bvals/cc/dustdiffusion/bvals_dustdiffusion.hpp"
#include "../dustfluids.hpp"


//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cell_center_diffusions.hpp
//! \brief defines cell center diffusive momentum of dust fluids

// Forward declarations
class DustFluids;
class ParameterInput;
class Coordinates;

//! \class DustFluidsCellCenterDiffusion
//! \brief cell center diffusive momentum

class DustFluidsCellCenterDiffusion {
 friend class DustFluidsDiffusion;
 public:
  DustFluidsCellCenterDiffusion(MeshBlock *pmb, DustFluids *pdf, ParameterInput *pin);

  AthenaArray<Real> diff_mom_cc;

  // storage for mesh refinement, SMR/AMR
  AthenaArray<Real> coarse_diff_mom_cc_; // used in mesh refinement
  int refinement_idx{-1};                // vector of pointers in MeshRefinement class

  AthenaArray<Real> diff_cc_flux[3];     // face-averaged flux vector

  DustDiffusionBoundaryVariable diffccbvar;

  void CalculateDiffusiveMomentum(const AthenaArray<Real> &prim_df, const AthenaArray<Real> &w);

 private:
  DustFluids  *pmy_dustfluids_; // ptr to DustFluids containing this DustFluidsDiffusion
  MeshBlock   *pmb_;            // ptr to meshblock containing this DustFluidsDiffusion
  Coordinates *pco_;            // ptr to coordinates class
};
#endif // CELLCENTER_DIFFUSION_HPP_
