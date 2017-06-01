#ifndef ADATM_SORT_H
#define ADATM_SORT_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "sptensor.h"



/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/
void pair_sort(
  pair * const pair_list,
  idx_t const size);


void pair_sort_range(
  pair * const pair_list,
  idx_t const start,
  idx_t const end);


#endif
