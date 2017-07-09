/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "stats.h"
#include "sptensor.h"
#include "ftensor.h"
#include "csf.h"
#include "io.h"
#include "reorder.h"
#include "util.h"

#include "adatm_stats.h"


/******************************************************************************
 * SPLATT PRIVATE FUNCTIONS
 *****************************************************************************/
static void p_stats_csf_mode(
  splatt_csf const * const ct)
{
  printf("nmodes: %"SPLATT_PF_IDX" nnz: %"SPLATT_PF_IDX"\n", ct->nmodes,
      ct->nnz);
  printf("dims: %"SPLATT_PF_IDX"", ct->dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX"", ct->dims[m]);
  }
  printf(" (%"SPLATT_PF_IDX"", ct->dim_perm[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("->%"SPLATT_PF_IDX"", ct->dim_perm[m]);
  }
  printf(")\n");
  printf("ntiles: %"SPLATT_PF_IDX" tile dims: %"SPLATT_PF_IDX"", ct->ntiles,
      ct->tile_dims[0]);
  for(idx_t m=1; m < ct->nmodes; ++m) {
    printf("x%"SPLATT_PF_IDX"", ct->tile_dims[m]);
  }

  idx_t empty = 0;
  for(idx_t t=0; t < ct->ntiles; ++t) {
    if(ct->pt[t].vals == NULL) {
      ++empty;
    }
  }

  printf("  empty: %"SPLATT_PF_IDX" (%0.1f%%)\n", empty,
      100. * (double)empty/ (double)ct->ntiles);
}


/******************************************************************************
 * AdaTM PUBLIC FUNCTIONS
 *****************************************************************************/
void stats_csf_adaptive(
  splatt_csf const * const ct,
  idx_t const n_csf)
{
  char * name = (char *)splatt_malloc(100 * sizeof(char));

  for(idx_t g=0; g<n_csf; ++g) {
    sprintf(name, "CSF-%lu", g);
    printf("%s\n", name);
    p_stats_csf_mode(ct + g);
  }

  free(name);
}


void stats_rcsf_adaptive(
  rcsf_seq_adaptive const * const seq_rcsfs,
  idx_t const n_rcsf)
{
  char * name = (char *)splatt_malloc(100 * sizeof(char));

  for(idx_t g=0; g<n_rcsf; ++g) {
    sprintf(name, "vCSF Seq %lu", g);
    printf("%s\n", name);
    for (int i=0; i<(seq_rcsfs+g)->n_imten; ++i) {
      sprintf(name, "   vCSF[%lu]:", i);
      printf("%s\n", name);
      p_stats_csf_mode((seq_rcsfs+g)->rcsfs+i);
    }
  }

  free(name);
}
