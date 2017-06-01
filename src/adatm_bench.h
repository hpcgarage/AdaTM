#ifndef ADATM_BENCH_H
#define ADATM_BENCH_H


/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "matrix.h"
#include "sptensor.h"
#include "reorder.h"
#include "bench.h"


/**
* @brief Benchmark AdaTM for an Nth-MTTKRP sequence.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in/out] mats factor matrices.
* @param[out] opts record execution settings.
*/
void bench_adaptm(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts);


/**
* @brief Determine the product order of N MTTKRPs of a sequence.
*
* @param[in] tt input sparse tensor in COO format.
* @param[out] product_order two-level array to save the product order of each MTTKRP.
*/
void decide_product_order (sptensor_t * const tt, idx_t **product_order);


/**
* @brief Determine the groups of N MTTKRPs of a sequence.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in] nfactors the CPD rank, i.e. the column of factor matrices.
* @param[in] strategy 1: minimum number of operations, i.e. maximal performance; 2: minimum storage space.
* @param[in] product_order the product order of each MTTKRP.
* @param[out] grp_prop output group properties.
*/
idx_t decide_parameters_auto (
  sptensor_t * const tt, 
  idx_t const nfactors,
  int strategy,
  idx_t const ** product_order,
  group_properties ** grp_prop);


/**
* @brief Determine the configuration of an MTTKRP sequence.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in] product_order the product order of each MTTKRP.
* @param[out] configs all possible configurations for the entire MTTKRP sequence.
*/
idx_t decide_candidate_configs (
  sptensor_t * const tt, 
  idx_t const ** product_order,
  configurations_adaptive ** configs);


/**
* @brief Predict the configuration of an MTTKRP sequence.
*
* @param[in] tt input sparse tensor in COO format.
* @param[in] nfactors the CPD rank, i.e. the column of factor matrices.
* @param[in/out] configs all possible configurations for the entire MTTKRP sequence, predict the space (pspace) and number of products (pops).
* @param[in] nconfigs the number of configurations.
*/
void predict_candidates_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  configurations_adaptive * configs,
  idx_t const nconfigs);


/**
* @brief Truncate configurations by given space.
*
* @param[in] configs all possible configurations for the entire MTTKRP sequence, with the predicted space (pspace) and number of products (pops).
* @param[in] nconfigs the number of all possible configurations.
* @param[out] real_config_locs truncate the configurations with the predicts space smaller than the set SLIMIT ("adatm_base.h").
* @return The number of satisfied configurations, the length of real_config_locs.
*/
idx_t limit_candidates_by_space(
  configurations_adaptive const * const configs,
  idx_t const nconfigs,
  idx_t ** const real_config_locs);


/**
* @brief Truncate configurations by strategy.
*
* @param[in] configs all possible configurations for the entire MTTKRP sequence, with the predicted space (pspace) and number of products (pops).
* @param[in] nconfigs the number of all possible configurations.
* @param[in] real_config_locs truncate the configurations with the predicts space smaller than the set SLIMIT ("adatm_base.h").
* @param[in] n_real_configs The length of real_config_locs.
* @param[in] strategy 1: performance oriented; 2: space oriented.
*/
idx_t choose_candidates_by_strategy(
  configurations_adaptive const * const configs,
  idx_t const nconfigs,
  idx_t const * const real_config_locs,
  idx_t const n_real_configs,
  int strategy);


#endif
