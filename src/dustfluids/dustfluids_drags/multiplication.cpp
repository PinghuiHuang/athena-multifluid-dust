//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file multiplication.cpp
//! Compute the multiplications between matrixes.

// C++ headers
#include <algorithm>   // min,max
#include <cstring>    // strcmp
#include <limits>
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../defs.hpp"
#include "../dustfluids.hpp"
#include "dust_gas_drag.hpp"


// Matrix Multiplication
void DustGasDrag::Multiply(const AthenaArray<Real> &a_matrix,
          const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
      for(int s=0; s<NDUSTGAS; ++s) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          c_matrix(m, n, i) += a_matrix(m, s, i)*b_matrix(s, n, i);
        }
      }
    }
  }
  return;
}


void DustGasDrag::MultiplyVector(const AthenaArray<Real> &a_matrix,
                      const AthenaArray<Real> &b_vector, AthenaArray<Real> &c_vector) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        c_vector(n, i) += a_matrix(m, n, i)*b_vector(m, i);
      }
    }
  }
  return;
}


void DustGasDrag::Multiply(const Real a_num, const AthenaArray<Real> &b_matrix,
              AthenaArray<Real> &c_matrix) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        c_matrix(m, n, i) = a_num*b_matrix(m, n, i);
      }
    }
  }
  return;
}


void DustGasDrag::Multiply(const Real a_num, AthenaArray<Real> &b_matrix) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b_matrix(m, n, i) *= a_num;
      }
    }
  }
  return;
}
