#ifndef SPLATT_MTTKRP_H
#define SPLATT_MTTKRP_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "ftensor.h"
#include "csf.h"
#include "thd_info.h"


/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

#define mttkrp_csf splatt_mttkrp_csf
/**
* @brief Matricized Tensor Times Khatri-Rao Product (MTTKRP) with a CSF tensor.
*        This is the primary computation involved in CPD. Output is written to
*        mats[SPLATT_MAX_NMODES].
*
*        TODO: Outputting to mats is dumb. Make the output matrix a function
*              parameter.
*
* @param tensors The CSF tensor(s) to factor.
* @param mats The output and input matrices.
* @param mode Which mode we are computing for.
* @param thds Thread structures. TODO: make this easier to allocate.
* @param opts SPLATT options. This uses SPLATT_OPTION_CSF_ALLOC.
*/
void mttkrp_csf(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  double const * const opts);


void decide_use_csfs(
  idx_t const nmodes,
  group_properties * const grp_prop,
  int const n_grp,
  idx_t const n_csf,
  idx_t * use_csfs,
  idx_t * use_tags);


void mttkrp_csf_adaptive(
  splatt_csf const * const tensors,
  rcsf_seq_adaptive const * const rs_seq,
  idx_t const n_csf,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  group_properties * const grp_prop,
  int const n_grp,
  idx_t const use_csf,
  idx_t const use_tag,
  double const * const opts);



/******************************************************************************
 * DEPRECATED FUNCTIONS
 *****************************************************************************/

void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads);

void mttkrp_splatt_sync_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads);

void mttkrp_splatt_coop_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads);

void mttkrp_giga(
  spmatrix_t const * const spmat,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch);

void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch);

void mttkrp_stream(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode);


#endif
