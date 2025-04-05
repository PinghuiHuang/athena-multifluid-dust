//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file addition.cpp
//! Compute the addition between matrixes.

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


// Matrix Addition
void DustGasDrag::Add(const AthenaArray<Real> &a_matrix, const Real b_num,
          const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        c_matrix(m, n, i) = a_matrix(m, n, i) + b_num*b_matrix(m, n, i);
      }
    }
  }
  return;
}


void DustGasDrag::Add(AthenaArray<Real> &a_matrix, const Real b_num,
                const AthenaArray<Real> &b_matrix) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        a_matrix(m, n, i) += b_num*b_matrix(m, n, i);
      }
    }
  }
  return;
}


void DustGasDrag::Add(const Real a_num, const Real b_num,
          const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
      Real delta;
      m == n ? delta = 1.0 : delta = 0.0;
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        c_matrix(m, n, i) = a_num*delta + b_num*b_matrix(m, n, i);
      }
    }
  }
  return;
}


void DustGasDrag::Add(const Real a_num, const Real b_num,
                      AthenaArray<Real> &b_matrix) {

  int is = pmb_->is, ie = pmb_->ie;

  for(int m=0; m<NDUSTGAS; ++m) {
    for(int n=0; n<NDUSTGAS; ++n) {
      Real delta;
      m == n ? delta = 1.0 : delta = 0.0;
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b_matrix(m, n, i) = a_num*delta + b_num*b_matrix(m, n, i);
      }
    }
  }
  return;
}
