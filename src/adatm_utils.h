#ifndef SPLATT_UTIL_H
#define SPLATT_UTIL_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "utils.h"


double locate_ind_special (
  size_t const * const array, // non-decreasing array
  idx_t const length,
  size_t const ele,
  idx_t const nmodes);


idx_t argmin_elem_range(
  idx_t const * const arr,
  idx_t const N,
  idx_t const con);

idx_t array_max(
  idx_t const * const arr,
  idx_t const N);

idx_t array_min(
  idx_t const * const arr,
  idx_t const N);

idx_t array_min_range(
  idx_t const * const arr,
  idx_t const N,
  idx_t const con);

#endif
