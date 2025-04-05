#ifndef BVALS_CC_DUSTDIFFUSION_HPP_
#define BVALS_CC_DUSTDIFFUSION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_dustdiffusion.hpp
//! \brief

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../bvals_cc.hpp"

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//! \brief

class DustDiffusionBoundaryVariable : public CellCenteredBoundaryVariable {
 public:
  DustDiffusionBoundaryVariable(MeshBlock *pmb,
                        AthenaArray<Real> *var_diff_cc, AthenaArray<Real> *coarse_var,
                        AthenaArray<Real> *var_flux,
                        DustDiffusionBoundaryQuantity dustdiffusion_type);
                                                // AthenaArray<Real> &prim);
  virtual ~DustDiffusionBoundaryVariable() = default;

  // switch between DustDiffusion class members "cons_diff" and "prim_diff" (or "df_u" and "df_u1", ...)
  void SwapDustDiffusionQuantity(AthenaArray<Real> &var_diff_cc, DustDiffusionBoundaryQuantity dustdiffusion_type);
  void SelectCoarseBuffer(DustDiffusionBoundaryQuantity dustdiffusion_type);

  void AddDustDiffusionShearForInit();
  void ShearQuantities(AthenaArray<Real> &shear_cc_, bool upper) override;
  void SetDustDiffusionShearingBoxBoundaryBuffers();
  void SetDustDiffusionFluxShearingBoxBoundaryBuffers();

  //!@{
  //! BoundaryPhysics: need to flip sign of velocity vectors for Reflect*()
  void ReflectInnerX1(Real time, Real dt,
                      int il, int jl, int ju, int kl, int ku, int ngh) override;
  void ReflectOuterX1(Real time, Real dt,
                      int iu, int jl, int ju, int kl, int ku, int ngh) override;
  void ReflectInnerX2(Real time, Real dt,
                      int il, int iu, int jl, int kl, int ku, int ngh) override;
  void ReflectOuterX2(Real time, Real dt,
                      int il, int iu, int ju, int kl, int ku, int ngh) override;
  void ReflectInnerX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int kl, int ngh) override;
  void ReflectOuterX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int ku, int ngh) override;
  //!@}

  //protected:
 private:
  void SetBoundarySameLevel(Real *buf, const NeighborBlock& nb) override;
  //! DustDiffusion is a unique cell-centered variable because of the relationship between
  //! DustDiffusionBoundaryQuantity::cons_diff and DustDiffusionBoundaryQuantity::prim_diff.
  DustDiffusionBoundaryQuantity dustdiffusion_type_;
  int LoadFluxBoundaryBufferSameLevel(Real *buf, const NeighborBlock& nb) final;
  void SetBoundaryFromCoarser(Real *buf, const NeighborBlock& nb) final;
  void SetBoundaryFromFiner(Real *buf, const NeighborBlock& nb) final;
  void PolarWedgeInnerX2( Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh) final;
  void PolarWedgeOuterX2( Real time, Real dt, int il, int iu, int jl, int kl, int ku, int ngh) final;
};

#endif // BVALS_CC_DUSTDIFFUSION_HPP_
