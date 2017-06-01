#ifndef ADATM_RCSF_H
#define ADATM_RCSF_H

#include "base.h"
#include "csf.h"
#include "adatm_base.h"

/**
* @brief Allocate CSF tensors for each memoized MTTKRP.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in] grp_prop Group properties.
* @param[in] n_grp The number of MTTKRP groups.
* @param[in] opts record execution settings.
* @param[out] n_csf The number of memoized tensor.
* @return n_csf CSF tensors.
*/
splatt_csf * csf_alloc_adaptive(
  sptensor_t * const tt,
  group_properties * const grp_prop,
  idx_t const n_grp,
  double const * const opts,
  idx_t * n_csf);


/**
* @brief Count the number of fibers at mode "mode".
*
* @param[in] tt input sparse tensor in COO format, sorted as "o_opt" order.
* @param[in] mode The given mode.
* @param[in] o_opt The optimal order of products in an MTTKRP.
* @param[in] fprev The previous fiber pointer.
* @param[out] nfibs_per_grp The number of fibers per MTTKRP group.
* @param[out] fnext The next fiber pointer.
*/
void count_nfibs(
  sptensor_t const * const tt,
  idx_t const mode,
  idx_t const * const o_opt,
  idx_t * const fprev,
  idx_t * const nfibs_per_grp,
  idx_t ** const fnext);

/**
* @brief Predict the bytes of CSF tensors for an MTTKRP group.
*
* @param[in] tt input sparse tensor in COO format.
* @param[out] nfibs_per_grp The number of fibers per MTTKRP group.
* @return Predicted CSF bytes.
*/
size_t predict_csf_bytes_adaptive(
  sptensor_t * const tt,
  idx_t const * const nfibs_per_grp);

/**
* @brief Predict the bytes of vCSF tensors for an MTTKRP group.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in] nfactors the CPD rank, i.e. the column of factor matrices.
* @param[in] begin_imten The beginning location of memoized intermediate tensors.
* @param[in] n_imten The number of memoized intermediate tensors.
* @param[out] nfibs_per_grp The number of fibers per MTTKRP group.
* @return Predicted vCSF bytes.
*/
size_t predict_rcsf_bytes_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  idx_t const begin_imten,
  idx_t const n_imten,
  idx_t const * const nfibs_per_grp);

/**
* @brief Predict the number of products of CSF tensors for an MTTKRP group.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in] nfactors the CPD rank, i.e. the column of factor matrices.
* @param[out] nfibs_per_grp The number of fibers per MTTKRP group.
* @return Predicted number of products.
*/
size_t predict_csf_ops_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  idx_t const * const nfibs_per_grp);

/**
* @brief Predict the bytes of vCSF tensors for an MTTKRP group.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in] nfactors the CPD rank, i.e. the column of factor matrices.
* @param[in] begin_imten The beginning location of memoized intermediate tensors.
* @param[in] n_imten The number of memoized intermediate tensors.
* @param[in] num_reMTTKRP The numebr of re-calculated MTTKRPs.
* @param[out] nfibs_per_grp The number of fibers per MTTKRP group.
* @return Predicted number of products.
*/
size_t predict_rcsf_ops_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  idx_t const begin_imten,
  idx_t const n_imten,
  idx_t const num_reMTTKRP,
  idx_t const * const nfibs_per_grp);

/**
* @brief Actually compute the bytes of all CSF tensors for an entire MTTKRP sequence.
*
* @param[in] tensors input sparse tensor in CSF format.
* @param[in] n_csf The number of CSF tensors.
* @param[in] grp_prop Group properties.
* @param[in] n_grp The number of MTTKRP groups.
* @return Predicted storage size of CSF tensors in bytes.
*/
size_t csf_storage_adaptive(
  splatt_csf const * const tensors,
  idx_t const n_csf,
  group_properties * const grp_prop,
  int const n_grp);

/**
* @brief Actually compute the bytes of all vCSF tensors for an MTTKRP group.
*
* @param[in] seq_rcsf saved intermediate tensors in vCSF format.
* @param[in] n_csf The number of MTTKRP groups.
* @return Predicted storage size of vCSF tensors in bytes.
*/
size_t rcsf_storage_adaptive(
  rcsf_seq_adaptive const * const seq_rcsf,
  idx_t const n_csf);

/**
* @brief Allocate vCSF tensors for an MTTKRP group.
*
* @param[in] ct input sparse tensor in CSF format.
* @param[in] nfactors the CPD rank, i.e. the column of factor matrices.
* @param[in] grp_prop Group properties.
* @param[in] n_grp The number of MTTKRP groups.
* @param[in] n_csf The number of CSF tensors.
* @param[in] opts record execution settings.
* @param[out] n_rcsf The number of vCSF tensors in this MTTKRP group.
* @return vCSF tensors of an MTTKRP group.
*/
rcsf_seq_adaptive * rcsf_alloc_adaptive(
  splatt_csf * const ct,
  idx_t const nfactors,
  group_properties * const grp_prop,
  idx_t const n_grp,
  idx_t const n_csf,
  double const * const opts,
  idx_t * n_rcsf);

/**
* @brief Arrange the modes in decreasing order except mode "mode".
*
* @param[in] nmodes The number of modes.
* @param[in] mode Given mode.
* @param[out] rdims The mode order.
*/
void rcsf_reverse_mode_order(
  idx_t const nmodes,
  idx_t const mode,
  idx_t * const rdims);

#endif