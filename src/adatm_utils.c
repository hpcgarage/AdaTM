/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "util.h"
#include "adatm_utils.h"


double locate_ind_special (
  size_t const * const array, // non-decreasing array
  idx_t const length,
  size_t const ele,
  idx_t const nmodes)
{
  double loc = -1;
  if(array[0] == ele) loc = 0;
  for (idx_t i=1; i<length; ++i) {
    if (array[i] == ele) {
      loc = i;
    } else if (array[i-1] < ele && ele < array[i]) {
      loc = i - 0.5;
    }
    break;
  }
  if (array[length-1] < ele && ele < nmodes) {
    loc = length - 0.5;
  }

  return loc;
}


idx_t argmin_elem_range(
  idx_t const * const arr,
  idx_t const N,
  idx_t const con)
{
  idx_t mkr;
  for(idx_t i=0; i < N; ++i) {
    if(arr[i] > con) {
      mkr = i;
      break;
    }
  }
  for(idx_t i=0; i < N; ++i) {
    if(arr[i] > con && arr[i] < arr[mkr]) {
      mkr = i;
    }
  }
  return mkr;
}

idx_t array_max(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = arr[0];
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] > mkr) {
      mkr = arr[i];
    }
  }
  return mkr;
}

idx_t array_min(
  idx_t const * const arr,
  idx_t const N)
{
  idx_t mkr = arr[0];
  for(idx_t i=1; i < N; ++i) {
    if(arr[i] < mkr) {
      mkr = arr[i];
    }
  }
  return mkr;
}


idx_t array_min_range(
  idx_t const * const arr,
  idx_t const N,
  idx_t const con)
{
  idx_t mkr;
  for(idx_t i=0; i < N; ++i) {
    if(arr[i] > con) {
      mkr = arr[i];
      break;
    }
  }
  for(idx_t i=0; i < N; ++i) {
    if(arr[i] > con && arr[i] < mkr) {
      mkr = arr[i];
    }
  }
  return mkr;
}