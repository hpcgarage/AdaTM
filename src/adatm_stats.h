#ifndef ADATM_STATS_H
#define ADATM_STATS_H

#include "base.h"
#include "sptensor.h"
#include "csf.h"
#include "cpd.h"
#include "splatt_mpi.h"
#include "stats.h"



void stats_csf_adaptive(
  splatt_csf const * const ct,
  idx_t const n_csf);

void stats_rcsf_adaptive(
  rcsf_seq_adaptive const * const seq_rcsfs,
  idx_t const n_csf);


#endif
