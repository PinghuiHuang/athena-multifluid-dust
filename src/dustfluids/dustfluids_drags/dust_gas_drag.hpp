#ifndef DRAG_DUSTGAS_HPP_
#define DRAG_DUSTGAS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dust_gas_drag.hpp
//! \brief defines class DustGasDrag
//! Contains data and functions that implement physical (not coordinate) drag terms

// C headers

// C++ headers
#include <cstring>    // strcmp
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../dustfluids.hpp"

// Forward declarations
class DustFluids;
class ParameterInput;

//! \class DustGasDrag
//! \brief data and functions for drags between dust and gas
class DustGasDrag {
 public:
  DustGasDrag(DustFluids *pdf, ParameterInput *pin);

  // Flag
  bool DustFeedback_Flag;  // true or false, the flag of dust feedback term
  bool Dissipation_Flag;   // true or false, the flag of energy dissipation term
  std::string drag_method; // Drag methods
  Real time_drag;          // The time echo when the drags are active

  // Select the drag integrators
  void DragIntegrate(const int stage, const Real dt,
    const AthenaArray<Real> &stopping_time,
    const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  // Matrix Addition
  void Add(const AthenaArray<Real> &a_matrix, const Real b_num,
           const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix);

  void Add(AthenaArray<Real> &a_matrix, const Real b_num, const AthenaArray<Real> &b_matrix);

  void Add(const Real a_num, const Real b_num,
           const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix);

  void Add(const Real a_num, const Real b_num, AthenaArray<Real> &b_matrix);

  // Matrix Multiplication
  void Multiply(const AthenaArray<Real> &a_matrix,
                const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix);

  void MultiplyVector(const AthenaArray<Real> &a_matrix,
                      const AthenaArray<Real> &b_vector, AthenaArray<Real> &c_vector);

  void Multiply(const Real a_num, const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix);

  void Multiply(const Real a_num, AthenaArray<Real> &b_matrix);

  // Matrix Inverse
  void LUdecompose(const AthenaArray<Real> &a_matrix, AthenaArray<Real> &index_vector,
                    AthenaArray<Real> &lu_matrix);

  // Solve A*x = b
  void SolveLinearEquation(const AthenaArray<Real> &index_vector, const AthenaArray<Real> &lu_matrix,
                                 AthenaArray<Real> &b_matrix, AthenaArray<Real> &x_matrix);

  void SolveMultipleLinearEquation(const AthenaArray<Real> &index_vector,
      const AthenaArray<Real> &lu_matrix, AthenaArray<Real> &b_matrix, AthenaArray<Real> &x_matrix);

  // Calculate the inverse of matrix
  void Inverse(const AthenaArray<Real> &index_vector, const AthenaArray<Real> &lu_matrix,
                  AthenaArray<Real> &a_matrix, AthenaArray<Real> &a_inv_matrix);

  // Time Integrators
  // Explitcit Integartors, these are consistent with default integrators in athena++
  void ExplicitFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void ExplicitNoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void RK2ExplicitFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  // Semi-Implicit Integrators
  // Trapezoid Methods (Crank-Nicholson Methods), 2nd order time convergence
  void TrapezoidFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void TrapezoidNoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  // Trapezoid Backward Differentiation Formula 2 methods, 2nd order time convergence
  void TRBDF2Feedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void TRBDF2NoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  // Fully Implicit Integartors
  // Backward Euler methods (Backward Differentiation Formula 1, BDF1), 1st order time convergence
  void BackwardEulerFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void BackwardEulerNoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void VL2BackwardEulerFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void VL2BackwardEulerNoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void RK2BackwardEulerFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void RK2BackwardEulerNoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  // Van Leer 2 Implicit methods, 2nd order time convergence
  void VL2ImplicitFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void VL2ImplicitNoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  // Runge Kutta 2 Implicit methods, 2nd order time convergence
  void RK2ImplicitFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

  void RK2ImplicitNoFeedback(const int stage, const Real dt,
      const AthenaArray<Real> &stopping_time,
      const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

 private:
  std::string integrator_;        // Time Integrator
  int         method_id_;         // The integrator method id
  DustFluids  *pmy_dustfluids_;   // ptr to DustFluids containing this DustGasDrag
  MeshBlock   *pmb_;              // ptr to meshblock containing this DustGasDrag
  Coordinates *pco_;              // ptr to coordinates class
  Hydro       *pmy_hydro_;        // ptr to hydro class
  int         orb_advection_;     // Orbital Advection

  AthenaArray<Real> force_x1, force_x2, force_x3;
  AthenaArray<Real> force_x1_n, force_x2_n, force_x3_n;

  AthenaArray<Real> delta_mom1,       delta_mom2,       delta_mom3;
  AthenaArray<Real> delta_mom1_im,    delta_mom2_im,    delta_mom3_im;
  AthenaArray<Real> delta_mom1_im_II, delta_mom2_im_II, delta_mom3_im_II;
  AthenaArray<Real> delta_mom1_src,   delta_mom2_src,   delta_mom3_src;

  AthenaArray<Real> mom1_prim,   mom2_prim,   mom3_prim;
  AthenaArray<Real> mom1_prim_n, mom2_prim_n, mom3_prim_n;

  AthenaArray<Real> idx_vector, lu_matrix;
  AthenaArray<Real> jacobi, jacobi_n, product, lambda, lambda_inv;

  AthenaArray<Real> biggest_arr, temp_arr;
  AthenaArray<Real> det_arr, scale_arr;
  AthenaArray<Real> sum_arr, xx_arr;
  AthenaArray<int>  mmax_arr;

  AthenaArray<Real> inv_gas_rho, inv_gas_rho_n;
  AthenaArray<Real> alpha, alpha_n;
  AthenaArray<Real> epsilon, epsilon_n;
  AthenaArray<Real> qvalue, weight_gas, weight_dust;

  AthenaArray<Real> temp_rho, temp_inv_rho;
  AthenaArray<Real> temp_mom1, temp_mom2, temp_mom3;
  AthenaArray<Real> temp_total_vel1, temp_total_vel2, temp_total_vel3;
  AthenaArray<Real> temp_A, temp_B, temp_C, temp_D;
};
#endif // DRAG_DUSTGAS_HPP_
