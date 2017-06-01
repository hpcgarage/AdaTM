/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "reorder.h"
#include "sort.h"
#include "io.h"
#include "tile.h"
#include "stats.h"
#include "util.h"
#include "bench.h"

#include "adatm_base.h"
#include "adatm_bench.h"
#include "adatm_utils.h"

#include <omp.h>


/******************************************************************************
 * PRIVATE FUNCTIONS from SPLATT
 *****************************************************************************/
static void p_log_mat(
  char const * const ofname,
  matrix_t const * const mat,
  idx_t const * const iperm)
{
  if(iperm != NULL) {
    matrix_t * mat_permed = perm_matrix(mat, iperm, NULL);
    mat_write(mat_permed, ofname);
    mat_free(mat_permed);
  } else {
    mat_write(mat, ofname);
  }
}

static void p_shuffle_mats(
  matrix_t ** mats,
  idx_t * const * const perms,
  idx_t const nmodes)
{
  for(idx_t m=0; m < nmodes; ++m) {
    if(perms[m] != NULL) {
      matrix_t * mperm = perm_matrix(mats[m], perms[m], NULL);
      mat_free(mats[m]);
      mats[m] = mperm;
    }
  }
}

/******************************************************************************
 * PRIVATE FUNCTIONS for AdaTM
 *****************************************************************************/
/* 
 * product_order: first nmodes entries indicate using "unchanged" factor 
 * matrices; last nmode entries indicate using "updated" factor matrices. 
 * Separate "unchanged" and "updated" factors helps to decide memoization. 
 */
void decide_product_order (sptensor_t * const tt, idx_t **product_order)
{
  idx_t const nmodes = tt->nmodes;
  for(idx_t m=0; m<nmodes; ++m) {
    memset(product_order[m], 0, 2 * nmodes * sizeof(idx_t));
    for(idx_t i=m+1; i<nmodes; ++i) {
      product_order[m][i] = 1;
    }
    for(idx_t i=0; i<m; ++i) {
      product_order[m][i + nmodes] = 1;
    }
  }
}


idx_t limit_candidates_by_space(
  configurations_adaptive const * const configs,
  idx_t const nconfigs,
  idx_t ** const real_config_locs)
{
  char * bstr_limit = bytes_str(SLIMIT);
  printf("Storage limitation: %s\n\n", bstr_limit);
  free(bstr_limit);

  idx_t n_real_configs = 0;
  for(idx_t c=0; c<nconfigs; ++c) {
    if(configs[c].pspace < SLIMIT) {
      ++ n_real_configs;
    }
  }
  assert(n_real_configs <= nconfigs);

  * real_config_locs = (idx_t *)splatt_malloc(n_real_configs * sizeof(idx_t));
  idx_t * tmp_real_config_locs = * real_config_locs;
  n_real_configs = 0;
  for(idx_t c=0; c<nconfigs; ++c) {
    if(configs[c].pspace < SLIMIT) {
      tmp_real_config_locs[n_real_configs] = c;
      ++ n_real_configs;
    }
  }

  return n_real_configs;
}


idx_t choose_candidates_by_strategy(
  configurations_adaptive const * const configs,
  idx_t const nconfigs,
  idx_t const * const real_config_locs,
  idx_t const n_real_configs,
  int strategy)
{
  idx_t optimal_loc = 0;
  configurations_adaptive * single_config;
  size_t min_ops = configs[real_config_locs[0]].pops;
  size_t min_space = configs[real_config_locs[0]].pspace;

  switch(strategy) {
  case 1: // minimum number of operations, i.e. maximal performance
    for(idx_t c=1; c<n_real_configs; ++c) {
      single_config = configs + real_config_locs[c];
      if(single_config->pops < min_ops) {
        min_ops = single_config->pops;
        optimal_loc = c;
      }
    }
    break;
  case 2: // minimum storage space
    for(idx_t c=1; c<n_real_configs; ++c) {
      single_config = configs + real_config_locs[c];
      if(single_config->pspace < min_space) {
        min_space = single_config->pspace;
        optimal_loc = c;
      }
    }
    break;
  }

  return optimal_loc;
}


void predict_candidates_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  configurations_adaptive * configs,
  idx_t const nconfigs)
{
  idx_t const nmodes = tt->nmodes;
  size_t csf_bytes = 0;
  size_t rcsf_bytes = 0;
  size_t csf_ops = 0;
  size_t rcsf_ops = 0;

  idx_t ** grp_nfibs = (idx_t **)splatt_malloc(nmodes * sizeof(idx_t *));
  for(idx_t g=0; g<nmodes; ++g) {
    grp_nfibs[g] = (idx_t *)splatt_malloc(nmodes * sizeof(idx_t));
  }
  idx_t n_grp, old_n_grp;
  group_properties * grp_prop;
  idx_t memo_mode, begin_imten, n_imten;
  idx_t * o_opt;
  idx_t * nfibs_per_grp;
  idx_t * fprev = NULL;
  idx_t * fnext = NULL;
  idx_t num_reMTTKRP = 0;

  old_n_grp = configs[0].n_grp;
  grp_prop = configs[0].grp_prop;
  for(idx_t g=0; g<old_n_grp; ++g) {
    fprev = NULL;
    fnext = NULL;
    memo_mode = grp_prop[g].memo_mode;
    begin_imten = grp_prop[g].begin_imten;
    n_imten = grp_prop[g].n_imten;
    o_opt = grp_prop[g].o_opt;
    tt_sort(tt, memo_mode, o_opt);
    nfibs_per_grp = grp_nfibs[memo_mode];
    memset(nfibs_per_grp, 0, nmodes * sizeof(idx_t));
    nfibs_per_grp[nmodes-1] = tt->nnz;
    for(idx_t m=0; m<nmodes-1; ++m) {
      count_nfibs(tt, m, o_opt, fprev, nfibs_per_grp, &fnext);
      if(fprev != NULL) splatt_free(fprev);
      fprev = fnext;
    }
    if(fprev != NULL) splatt_free(fprev);

    if(n_imten != 0) {
      if(g < old_n_grp - 1)
        num_reMTTKRP = grp_prop[g+1].memo_mode - memo_mode - 1;
      else
        num_reMTTKRP = nmodes - memo_mode - 1;
    }
    printf("Group: %lu\n", g);
    iprint_array(nfibs_per_grp, nmodes, "nfibs_per_grp");
    csf_bytes += predict_csf_bytes_adaptive(tt, nfibs_per_grp);
    // printf("csf_bytes: %lu\n", csf_bytes);
    rcsf_bytes += predict_rcsf_bytes_adaptive(tt, nfactors, begin_imten, n_imten, nfibs_per_grp);
    // printf("rcsf_bytes: %lu\n", rcsf_bytes);
    csf_ops += predict_csf_ops_adaptive(tt, nfactors, nfibs_per_grp);
    // printf("csf_ops: %lu\n", csf_ops);
    rcsf_ops += predict_rcsf_ops_adaptive(tt, nfactors, begin_imten, n_imten, num_reMTTKRP, nfibs_per_grp);
    // printf("rcsf_ops: %lu\n", rcsf_ops);
  }
  configs[0].pspace = csf_bytes + rcsf_bytes;
  configs[0].pops = csf_ops + rcsf_ops;

  
  for(idx_t c=1; c<nconfigs; ++c) {
    // printf("c: %lu\n", c); fflush(stdout);
    rcsf_bytes = 0;
    rcsf_ops = 0;
    n_grp = configs[c].n_grp;
    grp_prop = configs[c].grp_prop;
    if(n_grp != old_n_grp) {
      old_n_grp = n_grp;
      csf_bytes = 0;
      csf_ops = 0;
      for(idx_t g=0; g<n_grp; ++g) {
        fprev = NULL;
        fnext = NULL;
        memo_mode = grp_prop[g].memo_mode;
        begin_imten = grp_prop[g].begin_imten;
        n_imten = grp_prop[g].n_imten;
        o_opt = grp_prop[g].o_opt;
        tt_sort(tt, memo_mode, o_opt);
        nfibs_per_grp = grp_nfibs[memo_mode];
        memset(nfibs_per_grp, 0, nmodes * sizeof(idx_t));
        nfibs_per_grp[nmodes-1] = tt->nnz;
        for(idx_t m=0; m<nmodes-1; ++m) {
          count_nfibs(tt, m, o_opt, fprev, nfibs_per_grp, &fnext);
          if(fprev != NULL) splatt_free(fprev);
          fprev = fnext;
        }
        if(fprev != NULL) splatt_free(fprev);
        if(n_imten != 0) {
          if(g < n_grp - 1)
            num_reMTTKRP = grp_prop[g+1].memo_mode - memo_mode - 1;
          else
            num_reMTTKRP = nmodes - memo_mode - 1;
        }
        csf_bytes += predict_csf_bytes_adaptive(tt, nfibs_per_grp);
        rcsf_bytes += predict_rcsf_bytes_adaptive(tt, nfactors, 
          begin_imten, n_imten, nfibs_per_grp);
        csf_ops += predict_csf_ops_adaptive(tt, nfactors, nfibs_per_grp);
        rcsf_ops += predict_rcsf_ops_adaptive(tt, nfactors, 
          begin_imten, n_imten, num_reMTTKRP, nfibs_per_grp);
      }

    } else { // n_grp == old_n_grp
      for(idx_t g=0; g<n_grp; ++g) {
        memo_mode = grp_prop[g].memo_mode;
        begin_imten = grp_prop[g].begin_imten;
        n_imten = grp_prop[g].n_imten;
        if(n_imten != 0) {
          if(g < n_grp - 1)
            num_reMTTKRP = grp_prop[g+1].memo_mode - memo_mode - 1;
          else
            num_reMTTKRP = nmodes - memo_mode - 1;
        }
        rcsf_bytes += predict_rcsf_bytes_adaptive(tt, nfactors, 
          begin_imten, n_imten, nfibs_per_grp);
        rcsf_ops += predict_rcsf_ops_adaptive(tt, nfactors, 
          begin_imten, n_imten, num_reMTTKRP, nfibs_per_grp);
      }
    }
    configs[c].pspace = csf_bytes + rcsf_bytes;
    configs[c].pops = csf_ops + rcsf_ops;    
  } // End loop configs

  for(idx_t g=0; g<nmodes; ++g)
    splatt_free(grp_nfibs[g]);
  splatt_free(grp_nfibs);

  return;
}



idx_t decide_candidate_configs (
  sptensor_t * const tt, 
  idx_t const ** product_order,
  configurations_adaptive ** configs)
{
  idx_t const nmodes = tt->nmodes;
  idx_t * const dims = tt->dims;

  idx_t n_grp;
  idx_t * n_imten;
  idx_t * begin_imten;
  idx_t ** o_opt;


  /* Calculate from formulation */
  idx_t const n_grp_cal = (idx_t) ceil((double)nmodes / sqrt(2 * (nmodes-1)));
  printf("n_grp_cal: %lu\n", n_grp_cal);

  // Consider the most possible number of groups.
  idx_t * const n_grp_vec = (idx_t *)splatt_malloc((n_grp_cal + 1) * sizeof(idx_t)); 
  idx_t opt_conf_loc = 0;
  idx_t aver_len = 0;
  idx_t rest_nmodes = 0;
  idx_t * max_n_imten = (idx_t*)splatt_malloc((n_grp_cal + 1) * sizeof(idx_t));
  idx_t min_max_n_imten;
  idx_t nconfigs = 0;

  /* Calculate nconfigs */
  for(idx_t ng=0; ng<n_grp_cal + 1; ++ng) { // different number of groups
    n_grp = ng + 1;
    n_grp_vec[ng] = n_grp;

    aver_len = nmodes / n_grp;
    rest_nmodes = nmodes % n_grp;
    if(aver_len == 1 && rest_nmodes > 0)
      continue;
    memset(max_n_imten, 0, (n_grp_cal + 1) * sizeof(idx_t));
    for(idx_t i=0; i<n_grp; ++i) {
      if(n_grp != 1) {
        if(rest_nmodes > 0) {
          max_n_imten[i] = aver_len;
          -- rest_nmodes;
        } else {
          max_n_imten[i] = aver_len - 1;
        }
      } else {
        max_n_imten[i] = aver_len - 2;
      }
    }
    min_max_n_imten = array_min_range(max_n_imten, n_grp_cal+1, 0);
    nconfigs += min_max_n_imten;
  }
  // printf("nconfigs: %lu\n", nconfigs); fflush(stdout);

  /* find all possible configurations */
  * configs = (configurations_adaptive *)splatt_malloc(nconfigs * sizeof(configurations_adaptive));
  configurations_adaptive * tmp_configs = * configs;

  idx_t cfg = 0;
  group_properties * tmp_grp_prop;
  idx_t * memo_mode_arr = (idx_t *)splatt_malloc((n_grp_cal + 1) * sizeof(idx_t));
  idx_t * begin_imten_arr = (idx_t *)splatt_malloc((n_grp_cal + 1) * sizeof(idx_t));
  idx_t * sum_times = (idx_t *)splatt_malloc(2 * nmodes * sizeof(idx_t));
  idx_t * tmp_times = (idx_t *)splatt_malloc(2 * nmodes * sizeof(idx_t));
  idx_t * tmp_opt = (idx_t *)splatt_malloc(nmodes * sizeof(idx_t));

  for(idx_t ng=0; ng<n_grp_cal + 1; ++ng) { // different number of groups
    n_grp = n_grp_vec[ng];

    aver_len = nmodes / n_grp;
    rest_nmodes = nmodes % n_grp;
    if(aver_len == 1 && rest_nmodes > 0) {
      printf("Warning: Cannot have %lu groups.\n", n_grp);
      continue;
    }
    if(n_grp >= nmodes) {
      printf("Warning: Should have less than %lu groups.\n", nmodes);
      continue;
    }
    memset(max_n_imten, 0, (n_grp_cal + 1) * sizeof(idx_t));
    memset(begin_imten_arr, 0, (n_grp_cal + 1) * sizeof(idx_t));
    memset(memo_mode_arr, 0, (n_grp_cal + 1) * sizeof(idx_t));
    if(n_grp != 1) {
      for(idx_t g=0; g<n_grp; ++g) {
        if(rest_nmodes > 0) {
          max_n_imten[g] = aver_len;
          -- rest_nmodes;
        } else {
          max_n_imten[g] = aver_len - 1;
        }
        begin_imten_arr[g] = max_n_imten[g] + 1;
      }
      memo_mode_arr[0] = 0;
      for(idx_t g=1; g<n_grp; ++g) {
        memo_mode_arr[g] = memo_mode_arr[g-1] + max_n_imten[g-1] + 1;
      }
    } else { // For 3D tensor
      max_n_imten[0] = nmodes - 2;
      max_n_imten[1] = 0;
      begin_imten_arr[0] = nmodes - 1;
      begin_imten_arr[1] = 1;
      memo_mode_arr[0] = 0;
      memo_mode_arr[1] = nmodes-1;
      n_grp = 2;
    }
    min_max_n_imten = array_min_range(max_n_imten, n_grp_cal+1, 0);
    // printf("min_max_n_imten: %lu\n", min_max_n_imten);
    // iprint_array(max_n_imten, (n_grp_cal + 1), "max_n_imten");
    


    for(idx_t i=0; i<min_max_n_imten; ++i) {
      (tmp_configs + cfg)->n_grp = n_grp;
      (tmp_configs + cfg)->grp_prop = (group_properties *)splatt_malloc(n_grp * sizeof(group_properties));
      tmp_grp_prop = (tmp_configs + cfg)->grp_prop;

      for(idx_t g=0; g<n_grp; ++g) {
        (tmp_grp_prop + g)->memo_mode = memo_mode_arr[g];
        (tmp_grp_prop + g)->begin_imten = begin_imten_arr[g];
        if((tmp_grp_prop + g)->memo_mode != nmodes-1)
          (tmp_grp_prop + g)->n_imten = max_n_imten[g] - i; 
        else
          (tmp_grp_prop + g)->n_imten = 0;
        (tmp_grp_prop + g)->o_opt = (idx_t *)splatt_malloc(nmodes * sizeof(idx_t));

        /* Calculate o_opt */
        idx_t memo_m = tmp_grp_prop[g].memo_mode;
        
        memset(tmp_opt, 0, nmodes * sizeof(idx_t));
        memset(sum_times, 0, 2 * nmodes * sizeof(idx_t));
        memset(tmp_times, 0, 2 * nmodes * sizeof(idx_t));
        tmp_grp_prop[g].o_opt[0] = memo_m;
        tmp_opt[0] = memo_m;
        // iprint_array(memo_mode_arr, (n_grp_cal + 1), "memo_mode_arr");
        // printf("[n_grp: %lu, i: %lu, g: %lu]  memo_m: %lu, max_n_imten[g]: %lu\n", n_grp, i, g, memo_m, max_n_imten[g]);
        for(idx_t j=memo_m; j<=memo_m + max_n_imten[g]; ++j) {
          for(idx_t k=0; k<2 * nmodes; ++k) {
            sum_times[k] += product_order[j][k];
          }
        }
        for(idx_t k=0; k<2 * nmodes; ++k) {
          if(product_order[memo_m][k] > 0)
            tmp_times[k] = sum_times[k];
        }
        // iprint_array(tmp_times, 2 * nmodes, "tmp_times");
        for(idx_t j=1; j<nmodes; ++j) {
          idx_t tmp_min = argmin_elem_range(tmp_times, 2 * nmodes, 0);
          assert(tmp_min > 0);
          tmp_times[tmp_min] = 0;
          if(tmp_min >= nmodes)
            tmp_min -= nmodes;
          tmp_opt[j] = tmp_min; 
        }
        // iprint_array(tmp_opt, nmodes, "tmp_opt");
        
        /* sort in non-decreasing order for o_opt[1, ..., begin_imten] */
        idx_t sort_size = nmodes - tmp_grp_prop[g].begin_imten;
        pair * tmp_pairs = (pair *)splatt_malloc(sort_size * sizeof(pair));
        for(idx_t j=0; j<sort_size; ++j) {
          tmp_pairs[j].x = tmp_opt[j + tmp_grp_prop[g].begin_imten];
          tmp_pairs[j].y = dims[tmp_opt[j + tmp_grp_prop[g].begin_imten]];
        }
        pair_sort(tmp_pairs, sort_size);
        // print_pair_array(tmp_pairs, sort_size, "tmp_pairs after sort");

        for(idx_t j=0; j<sort_size; ++j) {
          tmp_grp_prop[g].o_opt[j + tmp_grp_prop[g].begin_imten] = tmp_pairs[j].x;
        }
        for(idx_t j=1; j<tmp_grp_prop[g].begin_imten; ++j) {
          tmp_grp_prop[g].o_opt[j] = tmp_opt[j];
        }

        splatt_free(tmp_pairs);
      } // Loop g in [0,n_grp), loop each group
      // print_group_properties(tmp_grp_prop, n_grp, nmodes, "grp_prop");
      ++ cfg;
    } // Loop i in [0,min_max_n_imten), loop different n_imten's.

  } // Loop ng in [0, n_grp_cal], different number of groups.
  assert(cfg == nconfigs);


  splatt_free(n_grp_vec);
  splatt_free(max_n_imten);
  splatt_free(memo_mode_arr);
  splatt_free(begin_imten_arr);
  splatt_free(sum_times);
  splatt_free(tmp_times);
  splatt_free(tmp_opt);

  return nconfigs;
}



idx_t decide_parameters_auto (
  sptensor_t * const tt, 
  idx_t const nfactors,
  int strategy,
  idx_t const ** product_order,
  group_properties ** grp_prop)
{
  idx_t const nmodes = tt->nmodes;

  configurations_adaptive * configs;
  idx_t nconfigs = decide_candidate_configs(tt, product_order, &configs);
  printf("nconfigs: %lu\n", nconfigs); fflush(stdout);

  /* Calculate pspace and ptime */
  predict_candidates_adaptive(tt, nfactors, configs, nconfigs);
  print_configs(configs, 0, nconfigs, nmodes, "predicted configs");

  idx_t * real_config_locs;
  idx_t n_real_configs = limit_candidates_by_space(configs, nconfigs, &real_config_locs);
  printf("n_real_configs: %lu\n", n_real_configs); fflush(stdout);
  iprint_array(real_config_locs, nconfigs, "real_config_locs");

  idx_t optimal_loc = choose_candidates_by_strategy(configs, nconfigs, real_config_locs, n_real_configs, strategy);
  printf("optimal_loc: %lu\n", optimal_loc); fflush(stdout);
  // 5d
  // idx_t optimal_loc = 34; //0; 34; 51; 62; 70; 76;
  // 10d
  // idx_t optimal_loc = 69; //0; 69; 103; 125; 141; 154; 164; 173;
  // 15d
  // idx_t optimal_loc = 83; // 0; 83; 124; 151; 171; 187; 200; 211;
  // printf("Set optimal_loc: %lu\n", optimal_loc); fflush(stdout);


  idx_t n_grp = configs[optimal_loc].n_grp;
  * grp_prop = (group_properties *)splatt_malloc(n_grp * sizeof(group_properties));
  group_properties * tmp_grp_prop = * grp_prop;
  group_properties * optimal_grp_group = configs[optimal_loc].grp_prop;
  for(idx_t g=0; g<n_grp; ++g) {
    (tmp_grp_prop + g)->memo_mode = (optimal_grp_group + g)->memo_mode;
    (tmp_grp_prop + g)->begin_imten = (optimal_grp_group + g)->begin_imten;
    (tmp_grp_prop + g)->n_imten = (optimal_grp_group + g)->n_imten;
    (tmp_grp_prop + g)->o_opt = (idx_t*)splatt_malloc(nmodes * sizeof(idx_t));
    for(idx_t m=0; m<nmodes; ++m) 
      (tmp_grp_prop + g)->o_opt[m] = (optimal_grp_group + g)->o_opt[m];
  }
  // print_group_properties(tmp_grp_prop, n_grp, nmodes, "Optimal grp_prop");

  for(idx_t c=0; c<nconfigs; ++c) {
    for(idx_t g=0; g<configs[c].n_grp; ++g) {
      splatt_free(configs[c].grp_prop[g].o_opt);
    }
    splatt_free(configs[c].grp_prop);
  }
  splatt_free(configs);

  return n_grp;
}




/******************************************************************************
 * AdaTM PUBLIC FUNCTIONS
 *****************************************************************************/
void bench_adaptm(
  sptensor_t * const tt,
  matrix_t ** mats,
  bench_opts const * const opts)
{
  printf("Running AdaTM.\n");

  int strategy = 1;
  idx_t const nmodes = tt->nmodes;
  idx_t const niters = opts->niters;
  printf("niters: %lu\n", niters);
  idx_t const * const threads = opts->threads;
  idx_t const nruns = opts->nruns;
  char matname[64];

  /* shuffle matrices if permutation exists */
  p_shuffle_mats(mats, opts->perm->perms, tt->nmodes);

  sp_timer_t itertime;
  sp_timer_t modetime;

  double * cpd_opts = splatt_default_opts();
  cpd_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_ADAPTIVE_TIME_EFFICIENT;
  cpd_opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
  cpd_opts[SPLATT_OPTION_NTHREADS] = threads[nruns-1];

  idx_t const nfactors = mats[0]->J;
  /* add 64 bytes to avoid false sharing */
  thd_info * thds = thd_init(threads[nruns-1], 3,
    (nfactors * nfactors * sizeof(val_t)) + 64,
    TILE_SIZES[0] * nfactors * sizeof(val_t) + 64,
    (tt->nmodes * nfactors * sizeof(val_t)) + 64);
  
  /* determine the product order of factor matrices for each MTTKRP, 
      distinguish new and old factors. */
  idx_t ** product_order = (idx_t **)splatt_malloc(nmodes * sizeof(idx_t*));
  for(idx_t i=0; i<nmodes; ++i) {
    product_order[i] = (idx_t *)splatt_malloc(2 * nmodes * sizeof(idx_t));
  }
  decide_product_order(tt, product_order);
  // printf("product_order:\n");
  // for(idx_t i=0; i<nmodes; ++i) {
  //   iprint_array(product_order[i], 2 * nmodes, "product_order[i]");
  // }


  /* determine the parameters of MTTKRP chain. */
  group_properties * grp_prop;
  idx_t n_grp = decide_parameters_auto (tt, nfactors, strategy, product_order, &grp_prop);
  printf("Optimal predicted grp_prop:\n");
  printf("n_grp: %lu\n", n_grp);
  print_group_properties(grp_prop, n_grp, nmodes, "Optimal grp_prop");
  for(idx_t i=0; i<nmodes; ++i) {
    splatt_free(product_order[i]);
  }
  splatt_free(product_order);


  /* Use these parameters to build "n_grp" csf tensors. */
  idx_t n_csf = 0;
  splatt_csf * cs = csf_alloc_adaptive(tt, grp_prop, n_grp, cpd_opts, &n_csf);
  printf("n_csf: %lu\n", n_csf);


  printf("** CSF **\n");
  unsigned long cs_bytes = csf_storage_adaptive(cs, n_csf, grp_prop, n_grp);
  char * bstr = bytes_str(cs_bytes);
  printf("CSF-STORAGE: %s\n\n", bstr);
  free(bstr);

  stats_csf_adaptive(cs, n_csf);
  printf("\n");

  // for(idx_t i=0; i<n_csf; ++i) {
  //   printf("CSF %lu\n", i);
  //   p_print_csf (cs + i);
  // }

  /* Store Intermediate CSF */
  idx_t n_rcsf = 0;
  rcsf_seq_adaptive * rs_seq = rcsf_alloc_adaptive (cs, nfactors, grp_prop, n_grp, n_csf, cpd_opts, &n_rcsf);
  printf("n_rcsf: %lu\n", n_rcsf);

  printf("** RCSF **\n");
  unsigned long rcsf_bytes = rcsf_storage_adaptive(rs_seq, n_rcsf);
  char * rbstr = bytes_str(rcsf_bytes);
  printf("RCSF-STORAGE: %s\n", rbstr);
  free(rbstr);

  stats_rcsf_adaptive(rs_seq, n_rcsf);
  printf("\n");
  fflush(stdout);

  // for(idx_t i=0; i<n_csf; ++i) {
  //   printf("RCSF %lu\n", i);
  //   print_rcsf_adaptive (rs_seq + i);
  // }

  /* Determine the CSF tensor to be used for each MTTKRP */
  idx_t * use_csfs = (idx_t *)splatt_malloc(nmodes * sizeof(idx_t));
  idx_t * use_tags = (idx_t *)splatt_malloc(nmodes * sizeof(idx_t));
  decide_use_csfs(nmodes, grp_prop, n_grp, n_csf, use_csfs, use_tags);
  iprint_array(use_csfs, nmodes, "use_csfs");
  iprint_array(use_tags, nmodes, "use_tags");


  // for (idx_t mi=0; mi<nmodes; ++mi) {
  //   printf("Factor matrix %lu\n", mi);
  //   print_mat(mats[mi]);
  // }
  // printf("Factor matrix %d\n", MAX_NMODES);
  // print_mat(mats[MAX_NMODES]);
  
  timer_start(&timers[TIMER_MISC]);
  /* for each # threads */
  for(idx_t t=0; t < nruns; ++t) {
    idx_t const nthreads = threads[t];
    omp_set_num_threads(nthreads);
    if(nruns > 1) {
      printf("## THREADS %" SPLATT_PF_IDX "\n", nthreads);
    }

    for(idx_t i=0; i < niters; ++i) {
      timer_fstart(&itertime);
      /* time each mode */
      for(idx_t m=0; m < nmodes; ++m) {
      // for(idx_t m=0; m < 1; ++m) {
        // Keep the same MTTKRP order with bench_csf_reuse
        timer_fstart(&modetime);
        mttkrp_csf_adaptive(cs, rs_seq, n_csf, mats, m, thds, grp_prop, n_grp, use_csfs[m], use_tags[m], cpd_opts);

        // printf("============ MTTKRP %lu ==============\n", m);
        // for(idx_t i=0; i<n_csf; ++i) {
        //   printf("RCSF %lu\n", i);
        //   print_rcsf_adaptive (rs_seq + i);
        // }
        // printf("Updated Factor matrix %lu\n", MAX_NMODES);
        // print_mat(mats[MAX_NMODES]);

        timer_stop(&modetime);
        printf("  mode %" SPLATT_PF_IDX " %0.3fs\n", m+1, modetime.seconds);
        // if(opts->write && t == nruns-1 && i == 0) {
        //   idx_t oldI = mats[MAX_NMODES]->I;
        //   mats[MAX_NMODES]->I = cs->dims[m];
        //   sprintf(matname, "csf_mode%"SPLATT_PF_IDX".mat", m+1);
        //   p_log_mat(matname, mats[MAX_NMODES], opts->perm->iperms[m]);
        //   mats[MAX_NMODES]->I = oldI;
        // }
      }
      timer_stop(&itertime);
      printf("    its = %3"SPLATT_PF_IDX" (%0.3fs)\n", i+1, itertime.seconds);
    }

    /* output load balance info */
    if(nruns > 1 || nthreads > 1) {
      thd_times(thds, threads[nruns-1]);
      thd_reset(thds, threads[nruns-1]);
      printf("\n");
    }
  }
  timer_stop(&timers[TIMER_MISC]);


  /* clean up */
  splatt_free(use_tags);
  splatt_free(use_csfs);
  csf_free(cs, cpd_opts);
  thd_free(thds, threads[nruns-1]);
  free(cpd_opts);
  // tt_free(tt);

  /* fix any matrices that we shuffled */
  p_shuffle_mats(mats, opts->perm->iperms, nmodes);
}