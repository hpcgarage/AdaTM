#ifndef ADATM_CPD_H
#define ADATM_CPD_H

/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "ftensor.h"
#include "matrix.h"
#include "splatt_mpi.h"


/**
* @brief CP-ALS Core function, called by "splatt_cpd_als_adaptive".
*
* @param[in] tensors input sparse tensor in CSF format.
* @param[in] rs_seq intermediate tensors in vCSF format, each memoized MTTKRP corresponds to an entry of rs_seq.
* @param[in] n_csf The number of MTTKRPs to be calculated from scratch, not necessarily equal to n_grp.
* @param[in] grp_prop The group properties of each MTTKRP group.
* @param[in] n_grp The number of groups.
* @param[in] use_csfs Identify which MTTKRP to be compute from scratch.
* @param[in] use_tags Identify the different algorithms for all MTTKRPs.
* @param[in/out] mats Input factor matrices.
* @param[in/out] lambda The stored norms in CPD.
* @param[in] nfactors the CPD rank, i.e. the column of factor matrices.
* @param[in] rinfo Thread info.
* @param[out] opts Execution settings.
*/
double cpd_als_iterate_adaptive(
  splatt_csf const * const tensors,
  rcsf_seq_adaptive * const rs_seq,
  splatt_idx_t const n_csf,
  group_properties const * const grp_prop,
  splatt_idx_t const n_grp,
  splatt_idx_t const * const use_csfs,
  splatt_idx_t const * const use_tags,
  matrix_t ** mats,
  val_t * const lambda,
  idx_t const nfactors,
  rank_info * const rinfo,
  double const * const opts);


#endif
