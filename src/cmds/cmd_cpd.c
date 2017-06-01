
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "splatt_cmds.h"
#include "../io.h"
#include "../sptensor.h"
#include "../stats.h"
#include "../cpd.h"
#include "../util.h"

#include "../adatm_cpd.h"
#include "../adatm_stats.h"


static char cpd_args_doc[] = "TENSOR";
static char cpd_doc[] =
  "splatt-cpd -- Compute the CPD of a sparse tensor.\n";

#define TT_NOWRITE 253
#define TT_TOL 254
#define TT_TILE 255
static struct argp_option cpd_options[] = {
  {"iters", 'i', "NITERS", 0, "maximum number of iterations to use (default: 50)"},
  {"tol", TT_TOL, "TOLERANCE", 0, "minimum change for convergence (default: 1e-5)"},
  {"rank", 'r', "RANK", 0, "rank of decomposition to find (default: 10)"},
  {"threads", 't', "NTHREADS", 0, "number of threads to use (default: #cores)"},
  {"tile", TT_TILE, 0, 0, "use tiling during SPLATT"},
  {"nowrite", TT_NOWRITE, 0, 0, "do not write output to file"},
  {"verbose", 'v', 0, 0, "turn on verbose output (default: no)"},
  { 0 }
};


typedef struct
{
  char * ifname;   /** file that we read the tensor from */
  int write;       /** do we write output to file? */
  double * opts;   /** splatt_cpd options */
  idx_t nfactors;
} cpd_cmd_args;


/**
* @brief Fill a cpd_opts struct with default values.
*
* @param args The cpd_opts struct to fill.
*/
static void default_cpd_opts(
  cpd_cmd_args * args)
{
  args->opts = splatt_default_opts();
  /* Reset to AdaTM setting. */
  args->opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_ADAPTIVE_TIME_EFFICIENT;
  args->ifname    = NULL;
  args->write     = DEFAULT_WRITE;
  args->nfactors  = DEFAULT_NFACTORS;
}



static error_t parse_cpd_opt(
  int key,
  char * arg,
  struct argp_state * state)
{
  cpd_cmd_args * args = state->input;
  char * buf;
  int cnt = 0;

  /* -i=50 should also work... */
  if(arg != NULL && arg[0] == '=') {
    ++arg;
  }

  switch(key) {
  case 'i':
    args->opts[SPLATT_OPTION_NITER] = (double) atoi(arg);
    break;
  case TT_TOL:
    args->opts[SPLATT_OPTION_TOLERANCE] = atof(arg);
    break;
  case 't':
    args->opts[SPLATT_OPTION_NTHREADS] = (double) atoi(arg);
    break;
  case 'v':
    timer_inc_verbose();
    args->opts[SPLATT_OPTION_VERBOSITY] += 1;
    break;
  case TT_TILE:
    args->opts[SPLATT_OPTION_TILE] = SPLATT_DENSETILE;
    break;
  case TT_NOWRITE:
    args->write = 0;
    break;
  case 'r':
    args->nfactors = atoi(arg);
    break;

  case ARGP_KEY_ARG:
    if(args->ifname != NULL) {
      argp_usage(state);
      break;
    }
    args->ifname = arg;
    break;
  case ARGP_KEY_END:
    if(args->ifname == NULL) {
      argp_usage(state);
      break;
    }
  }
  return 0;
}

static struct argp cpd_argp =
  {cpd_options, parse_cpd_opt, cpd_args_doc, cpd_doc};


/******************************************************************************
 * ADATM-CPD
 *****************************************************************************/
void splatt_cpd_cmd_adaptive(
  int argc,
  char ** argv)
{
  printf("Use splatt_cpd_cmd_adaptive.\n");
  int strategy = 1;

  /* assign defaults and parse arguments */
  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  sptensor_t * tt = NULL;

  print_header();

  tt = tt_read(args.ifname);
  if(tt == NULL) {
    return;
  }

  /* print basic tensor stats? */
  splatt_verbosity_type which_verb = args.opts[SPLATT_OPTION_VERBOSITY];
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
  }

  idx_t const nmodes = tt->nmodes;
  idx_t const nfactors = args.nfactors;
  /* 
   * Determine the product order of factor matrices for each MTTKRP, 
   * distinguish new and old factors. 
   */
  idx_t ** product_order = (idx_t **)splatt_malloc(nmodes * sizeof(idx_t*));
  for(idx_t i=0; i<nmodes; ++i) {
    product_order[i] = (idx_t *)splatt_malloc(2 * nmodes * sizeof(idx_t));
  }
  decide_product_order(tt, product_order);

  /* determine the parameters of Nth-MTTKRP sequence. */
  group_properties * grp_prop;
  idx_t n_grp = decide_parameters_auto (tt, nfactors, strategy, product_order, &grp_prop);
  printf("Optimal predicted grp_prop:\n");
  printf("n_grp: %lu\n", n_grp);
  print_group_properties(grp_prop, n_grp, nmodes, "Optimal grp_prop");
  for(idx_t i=0; i<nmodes; ++i) {
    splatt_free(product_order[i]);
  }
  splatt_free(product_order);


  /* Use these parameters to build "n_grp" CSF tensors. */
  idx_t n_csf = 0;
  splatt_csf * csf = csf_alloc_adaptive(tt, grp_prop, n_grp, args.opts, &n_csf);
  printf("n_csf: %lu\n", n_csf);

  tt_free(tt);

  printf("** CSF **\n");
  unsigned long cs_bytes = csf_storage_adaptive(csf, n_csf, grp_prop, n_grp);
  char * bstr = bytes_str(cs_bytes);
  printf("CSF-STORAGE: %s\n\n", bstr);
  free(bstr);

  stats_csf_adaptive(csf, n_csf);
  printf("\n");


  /* Store intermediate vCSFs */
  idx_t n_rcsf = 0;
  rcsf_seq_adaptive * rs_seq = rcsf_alloc_adaptive (csf, nfactors, grp_prop, n_grp, n_csf, args.opts, &n_rcsf);
  printf("n_rcsf: %lu\n", n_rcsf);

  printf("** RCSF **\n");
  unsigned long rcsf_bytes = rcsf_storage_adaptive(rs_seq, n_rcsf);
  char * rbstr = bytes_str(rcsf_bytes);
  printf("RCSF-STORAGE: %s\n", rbstr);
  free(rbstr);

  stats_rcsf_adaptive(rs_seq, n_rcsf);
  printf("\n");
  fflush(stdout);


  /* Determine which MTTKRP(s) to be calculated from scratch and the rest 
   * MTTKRP(s) to be computed using the saved intermediate tensors.
   */
  idx_t * use_csfs = (idx_t *)splatt_malloc(nmodes * sizeof(idx_t));
  idx_t * use_tags = (idx_t *)splatt_malloc(nmodes * sizeof(idx_t));
  decide_use_csfs(nmodes, grp_prop, n_grp, n_csf, use_csfs, use_tags);
  iprint_array(use_csfs, nmodes, "use_csfs");
  iprint_array(use_tags, nmodes, "use_tags");


  /* print CPD stats? */
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    cpd_stats(csf, args.nfactors, args.opts);
  }

  splatt_kruskal factored;

  /* do the factorization! */
  int ret = splatt_cpd_als_adaptive(csf, rs_seq, n_csf, grp_prop, n_grp, use_csfs, use_tags, args.nfactors, args.opts, &factored);
  if(ret != SPLATT_SUCCESS) {
    fprintf(stderr, "splatt_cpd_als_adaptive returned %d. Aborting.\n", ret);
    exit(1);
  }

  printf("Final fit: %"SPLATT_PF_VAL"\n", factored.fit);

  /* write output */
  if(args.write == 1) {
    vec_write(factored.lambda, args.nfactors, "lambda.mat");

    for(idx_t m=0; m < nmodes; ++m) {
      char * matfname = NULL;
      asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

      matrix_t tmpmat;
      tmpmat.rowmajor = 1;
      tmpmat.I = csf->dims[m];
      tmpmat.J = args.nfactors;
      tmpmat.vals = factored.factors[m];

      mat_write(&tmpmat, matfname);
      free(matfname);
    }
  }

  /* cleanup */
  splatt_free(use_tags);
  splatt_free(use_csfs);
  splatt_csf_free(csf, args.opts);
  free(args.opts);

  /* free factor matrix allocations */
  splatt_free_kruskal(&factored);

}


/******************************************************************************
 * SPLATT CPD
 *****************************************************************************/
void splatt_cpd_cmd(
  int argc,
  char ** argv)
{
  printf("Running SPLATT CPD.\n");
  /* assign defaults and parse arguments */
  cpd_cmd_args args;
  default_cpd_opts(&args);
  argp_parse(&cpd_argp, argc, argv, ARGP_IN_ORDER, 0, &args);

  sptensor_t * tt = NULL;

  print_header();

  tt = tt_read(args.ifname);
  if(tt == NULL) {
    return;
  }

  /* print basic tensor stats? */
  splatt_verbosity_type which_verb = args.opts[SPLATT_OPTION_VERBOSITY];
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    stats_tt(tt, args.ifname, STATS_BASIC, 0, NULL);
  }

  splatt_csf * csf = splatt_csf_alloc(tt, args.opts);

  idx_t nmodes = tt->nmodes;
  tt_free(tt);

  /* print CPD stats? */
  if(which_verb >= SPLATT_VERBOSITY_LOW) {
    cpd_stats(csf, args.nfactors, args.opts);
  }

  splatt_kruskal factored;

  /* do the factorization! */
  int ret = splatt_cpd_als(csf, args.nfactors, args.opts, &factored);
  if(ret != SPLATT_SUCCESS) {
    fprintf(stderr, "splatt_cpd_als returned %d. Aborting.\n", ret);
    exit(1);
  }

  printf("Final fit: %"SPLATT_PF_VAL"\n", factored.fit);

  /* write output */
  if(args.write == 1) {
    vec_write(factored.lambda, args.nfactors, "lambda.mat");

    for(idx_t m=0; m < nmodes; ++m) {
      char * matfname = NULL;
      asprintf(&matfname, "mode%"SPLATT_PF_IDX".mat", m+1);

      matrix_t tmpmat;
      tmpmat.rowmajor = 1;
      tmpmat.I = csf->dims[m];
      tmpmat.J = args.nfactors;
      tmpmat.vals = factored.factors[m];

      mat_write(&tmpmat, matfname);
      free(matfname);
    }
  }

  /* cleanup */
  splatt_csf_free(csf, args.opts);
  free(args.opts);

  /* free factor matrix allocations */
  splatt_free_kruskal(&factored);

}



