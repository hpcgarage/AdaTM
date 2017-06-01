/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sort.h"
#include "tile.h"
#include "io.h"
#include "adatm_rcsf.h"

/******************************************************************************
 * PRIVATE FUNCTIONS from SPLATT
 *****************************************************************************/
/**
* @brief Construct the sparsity structure of the outer-mode of a CSF tensor.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
*/
static void p_mk_outerptr(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr)
{
  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  assert(nnzstart < nnzend);

  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[ct->dim_perm[0]] + nnzstart;

  /* count fibers */
  idx_t nfibs = 1;
  for(idx_t x=1; x < nnz; ++x) {
    assert(ttind[x-1] <= ttind[x]);
    if(ttind[x] != ttind[x-1]) {
      ++nfibs;
    }
  }
  ct->pt[tile_id].nfibs[0] = nfibs;
  assert(nfibs <= ct->dims[ct->dim_perm[0]]);

  /* grab sparsity pattern */
  csf_sparsity * const pt = ct->pt + tile_id;

  pt->fptr[0] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
  // jli: when ntiles == 1, pt->fids[0] is not needed.
  if(ct->ntiles > 1) {
    pt->fids[0] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
  } else {
    pt->fids[0] = NULL;
  }

  idx_t  * const restrict fp = pt->fptr[0];
  idx_t  * const restrict fi = pt->fids[0];
  fp[0] = 0;
  if(fi != NULL) {
    fi[0] = ttind[0];
  }

  idx_t nfound = 1;
  for(idx_t n=1; n < nnz; ++n) {
    /* check for end of outer index */
    if(ttind[n] != ttind[n-1]) {
      if(fi != NULL) {
        fi[nfound] = ttind[n];
      }
      fp[nfound++] = n;
    }
  }

  fp[nfibs] = nnz;

}


/**
* @brief Construct the sparsity structure of any mode but the last. The first
*        (root) mode is handled by p_mk_outerptr and the first is simply a copy
*        of the nonzeros.
*
* @param ct The CSF tensor to construct.
* @param tt The coordinate tensor to construct from. Assumed to be already
*            sorted.
* @param tile_id The ID of the tile to construct.
* @param nnztile_ptr A pointer into 'tt' that marks the start of each tile.
* @param mode Which mode we are constructing.
*/
static void p_mk_fptr(
  splatt_csf * const ct,
  sptensor_t const * const tt,
  idx_t const tile_id,
  idx_t const * const nnztile_ptr,
  idx_t const mode)
{
  assert(mode < ct->nmodes);

  idx_t const nnzstart = nnztile_ptr[tile_id];
  idx_t const nnzend   = nnztile_ptr[tile_id+1];
  idx_t const nnz = nnzend - nnzstart;

  /* outer mode is easy; just look at outer indices */
  if(mode == 0) {
    p_mk_outerptr(ct, tt, tile_id, nnztile_ptr);
    return;
  }
  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[ct->dim_perm[mode]] + nnzstart;

  csf_sparsity * const pt = ct->pt + tile_id;

  /* we will edit this to point to the new fiber idxs instead of nnz */
  idx_t * const restrict fprev = pt->fptr[mode-1];

  /* first count nfibers */
  idx_t nfibs = 0;
  /* foreach 'slice' in the previous dimension */
  for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
    ++nfibs; /* one by default per 'slice' */
    /* count fibers in current hyperplane*/
    for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
      if(ttind[f] != ttind[f-1]) {
        ++nfibs;
      }
    }
  }
  pt->nfibs[mode] = nfibs;


  pt->fptr[mode] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
  pt->fids[mode] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
  idx_t * const restrict fp = pt->fptr[mode];
  idx_t * const restrict fi = pt->fids[mode];
  fp[0] = 0;

  /* now fill in fiber info */
  idx_t nfound = 0;
  for(idx_t s=0; s < pt->nfibs[mode-1]; ++s) {
    idx_t const start = fprev[s]+1;
    idx_t const end = fprev[s+1];

    /* mark start of subtree */
    fprev[s] = nfound;  // jli: modify the previous fptr.
    fi[nfound] = ttind[start-1];
    fp[nfound++] = start-1;

    /* mark fibers in current hyperplane */
    for(idx_t f=start; f < end; ++f) {
      if(ttind[f] != ttind[f-1]) {
        fi[nfound] = ttind[f];
        fp[nfound++] = f;
      }
    }
  }

  /* mark end of last hyperplane */
  fprev[pt->nfibs[mode-1]] = nfibs;
  fp[nfibs] = nnz;
}


/**
* @brief Allocate and fill a CSF tensor from a coordinate tensor without
*        tiling.
*
* @param ct The CSF tensor to fill out.
* @param tt The sparse tensor to start from.
*/
static void p_csf_alloc_untiled(
  splatt_csf * const ct,
  sptensor_t * const tt)
{ 
  idx_t const nmodes = tt->nmodes;
   // jli: sorting according to dim_perm order, when defined dim_perm, 2nd parameter is useless.
  tt_sort(tt, ct->dim_perm[0], ct->dim_perm);

  ct->ntiles = 1;
  for(idx_t m=0; m < nmodes; ++m) {
    ct->tile_dims[m] = 1;
  }
  ct->pt = splatt_malloc(sizeof(*(ct->pt)));

  csf_sparsity * const pt = ct->pt;

  // jli: fptr = NULL for the last mode
  /* last row of fptr is just nonzero inds */
  pt->nfibs[nmodes-1] = ct->nnz;
  pt->fids[nmodes-1] = splatt_malloc(ct->nnz * sizeof(**(pt->fids)));
  pt->vals           = splatt_malloc(ct->nnz * sizeof(*(pt->vals)));
  memcpy(pt->fids[nmodes-1], tt->ind[ct->dim_perm[nmodes-1]],
      ct->nnz * sizeof(**(pt->fids)));
  memcpy(pt->vals, tt->vals, ct->nnz * sizeof(*(pt->vals)));

  /* setup a basic tile ptr for one tile */
  idx_t nnz_ptr[2];
  nnz_ptr[0] = 0;
  nnz_ptr[1] = tt->nnz;

  /* create fptr entries for the rest of the modes, working down from roots.
   * Skip the bottom level (nnz) */
  for(idx_t m=0; m < tt->nmodes-1; ++m) {
    p_mk_fptr(ct, tt, 0, nnz_ptr, m);
  }
    // printf("OK in p_csf_alloc_untiled.\n");
    // fflush(stdout);

}

/******************************************************************************
 * PRIVATE FUNCTIONS for AdaTM
 *****************************************************************************/


static void p_rcsf_alloc_untiled_adaptive (
  rcsf_seq_adaptive * const seq_rcsfs,
  splatt_csf const * const csf,
  idx_t const nfactors,
  double const * const opts)
{
  idx_t nmodes = csf->nmodes;
  idx_t const begin_imten = seq_rcsfs->begin_imten;
  idx_t const n_imten = seq_rcsfs->n_imten;
  splatt_csf * const rcsfs = seq_rcsfs->rcsfs;

  for (int deg=0; deg<n_imten; ++deg) {
    splatt_csf * osubcsf = rcsfs + deg;
    idx_t loc = begin_imten - deg;

    osubcsf->nmodes = loc + 1;
    idx_t nmodes_deg = osubcsf->nmodes;
    // Permute the tensor dims, 
    // csf keeps the same dims with input tensor, but using dim_perm to specify the order 
    for (idx_t m=0; m<nmodes_deg; ++m) {
      osubcsf->dims[m] = csf->dims[csf->dim_perm[m]];  
    }
    osubcsf->dims[nmodes_deg-1] = nfactors;
    for (idx_t m=0; m<nmodes_deg; ++m) {
      // osubcsf->dim_perm[m] = m; // Depth
      osubcsf->dim_perm[m] = csf->dim_perm[m];
    }
    osubcsf->which_tile = csf->which_tile;
    osubcsf->ntiles = csf->ntiles;
    assert (osubcsf->which_tile == SPLATT_NOTILE); //TODO: only support no tile now.
    assert (osubcsf->ntiles == 1);
    for (idx_t m=0; m<nmodes_deg; ++m) {
      osubcsf->tile_dims[m] = csf->tile_dims[csf->dim_perm[m]]; //Permute the tensor's tile_dims
    }

    osubcsf->pt = splatt_malloc(osubcsf->ntiles*sizeof(*(osubcsf->pt)));  //when ntiles = 1
    osubcsf->nnz = csf->pt->nfibs[nmodes_deg-2] * nfactors; //dense fibers.
    for (idx_t t=0; t<osubcsf->ntiles; ++t) {
      csf_sparsity * const pt = csf->pt+t;
      csf_sparsity * const osubpt = osubcsf->pt+t;
      // No need to use dim_perm, since they are ordered from root to leaves.
      for (idx_t m=0; m<nmodes_deg-1; ++m) {
        osubpt->nfibs[m] = pt->nfibs[m];
      }
      osubpt->nfibs[nmodes_deg-1] = osubcsf->nnz;
      // reuse the fids and fptrs for mode-[0-nmodes_deg-3].
      if (nmodes_deg-2 > 0) {
        for (idx_t m=0; m<nmodes_deg-2; ++m) {
          osubpt->fptr[m] = pt->fptr[m]; // Point to CSF fptr and fids
          osubpt->fids[m] = pt->fids[m];
        }
      }
      // No need to store pointers for mode-(nmodes-1). Same with CSF
      osubpt->fptr[nmodes_deg-2] = NULL; 
      osubpt->fids[nmodes_deg-2] = pt->fids[nmodes_deg-2]; // Point to CSF fids
      osubpt->fptr[nmodes_deg-1] = NULL; // pt->fptr[nmodes-1] is also NULL; 
      osubpt->fids[nmodes_deg-1] = NULL; // No need to store indices for mode-(nmodes-1).
      osubpt->vals = splatt_malloc(osubcsf->nnz * sizeof(*(osubpt->vals))); 
      memset (osubpt->vals, 0, osubcsf->nnz * sizeof(*(osubpt->vals)));
    }
  }

}

static void p_mk_rcsf_adaptive(
  rcsf_seq_adaptive * const seq_rcsfs,
  splatt_csf const * const ct,
  idx_t const nfactors,
  double const * const splatt_opts)
{
  assert((splatt_csf_type)splatt_opts[SPLATT_OPTION_TILE] == SPLATT_NOTILE);
  p_rcsf_alloc_untiled_adaptive(seq_rcsfs, ct, nfactors, splatt_opts);
}


static void p_mk_csf_adaptive(
  splatt_csf * const ct,
  sptensor_t * const tt,
  group_properties * const grp_prop,
  idx_t const grp_idx)
{
  ct->nnz = tt->nnz;
  ct->nmodes = tt->nmodes;

  for(idx_t m=0; m < tt->nmodes; ++m) {
    ct->dims[m] = tt->dims[m];
    ct->dim_perm[m] = grp_prop[grp_idx].o_opt[m];
  }

  /* Only use no tiled */
  ct->which_tile = SPLATT_NOTILE;
  p_csf_alloc_untiled(ct, tt);
}

/******************************************************************************
 * AdaTM PUBLIC FUNCTIONS
 *****************************************************************************/
void rcsf_reverse_mode_order(
  idx_t const begin,
  idx_t const mode,
  idx_t * const rdims)
{
  int j = 0;
  for (int i=begin; i>=0; --i) {
    if (i != mode) {
      rdims[j] = i;
      ++ j;
    }
  }
  assert ( j == begin);
}


size_t predict_csf_bytes_adaptive(
  sptensor_t * const tt,
  idx_t const * const nfibs_per_grp)
{
  idx_t const nmodes = tt->nmodes;
  idx_t const nnz = tt->nnz;
  size_t bytes = 0;

  bytes += nnz * sizeof(val_t); /* vals */
  bytes += nnz * sizeof(idx_t); /* fids[nmodes] */
  for(idx_t m=0; m < nmodes-1; ++m) {
    bytes += (nfibs_per_grp[m]+1) * sizeof(idx_t); /* fptr */
    bytes += nfibs_per_grp[m] * sizeof(idx_t); /* fids */
  }

  return bytes;
}

size_t predict_rcsf_bytes_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  idx_t const begin_imten,
  idx_t const n_imten,
  idx_t const * const nfibs_per_grp)
{
  size_t bytes = 0;

  if(n_imten != 0) {
    for (idx_t i=0; i<n_imten; ++i) {
      bytes += nfibs_per_grp[begin_imten-i] * nfactors * sizeof(val_t);
    }
  }
  
  return bytes;
}

size_t predict_csf_ops_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  idx_t const * const nfibs_per_grp)
{
  idx_t const nmodes = tt->nmodes;
  idx_t const nnz = tt->nnz;
  size_t ops = 0;

  for(idx_t m=nmodes; m >= 1; --m) {
    ops += nfibs_per_grp[m-1] * nfactors * 2; /* fids */
  }

  return ops;
}


size_t predict_rcsf_ops_adaptive(
  sptensor_t * const tt,
  idx_t const nfactors,
  idx_t const begin_imten,
  idx_t const n_imten,
  idx_t const num_reMTTKRP,
  idx_t const * const nfibs_per_grp)
{
  size_t ops = 0;
  idx_t end = begin_imten - n_imten + 1;

  if(n_imten != 0) {
    for (idx_t i=1; i<=num_reMTTKRP; ++i) {
      if(i < end) {
        ops += nfibs_per_grp[end-1] * nfactors * 2;
      } else if(i >= end && i < begin_imten) {
        ops += nfibs_per_grp[i] * nfactors * 2;
      } else {
        printf("Wrong case: i: %lu.\n", i);
      }
    }
  }
  
  return ops;
}


/* modified from p_mk_fptr */
void count_nfibs(
  sptensor_t const * const tt,  // tt is sorted as o_opt order.
  idx_t const mode,
  idx_t const * const o_opt,
  idx_t * const fprev,
  idx_t * const nfibs_per_grp,
  idx_t ** const fnext)
{
  idx_t const nmodes = tt->nmodes;
  idx_t const nnz = tt->nnz;
  assert(mode < nmodes);

  idx_t nfibs;
  idx_t nfound;

  if(mode == 0) {
    assert(fprev == NULL);
    idx_t const * const restrict ttind = tt->ind[o_opt[0]];
    nfibs = 1;
    for(idx_t x=1; x < nnz; ++x) {
      assert(ttind[x-1] <= ttind[x]);
      if(ttind[x] != ttind[x-1]) {
        ++ nfibs;
      }    
    }
    nfibs_per_grp[0] = nfibs;
    assert(nfibs <= tt->dims[o_opt[0]]);

    * fnext = splatt_malloc((nfibs+1) * sizeof(idx_t));
    idx_t * tmp_fnext = * fnext;
    tmp_fnext[0] = 0;

    nfound = 1;
    for(idx_t x=1; x < nnz; ++x) {
      /* check for end of outer index */
      if(ttind[x] != ttind[x-1]) {
        tmp_fnext[nfound++] = x;
      }
    }
    assert(nfound == nfibs);
    tmp_fnext[nfibs] = nnz;
    // iprint_array(tmp_fnext, (nfibs+1), "tmp_fnext");
    return;
  }

  assert (fprev != NULL);
  /* the mode after accounting for dim_perm */
  idx_t const * const restrict ttind = tt->ind[o_opt[mode]];

  /* first count nfibers */
  nfibs = 0;
  /* foreach 'slice' in the previous dimension */
  for(idx_t s=0; s < nfibs_per_grp[mode-1]; ++s) {
    ++nfibs; /* one by default per 'slice' */
    /* count fibers in current hyperplane*/
    for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
      if(ttind[f] != ttind[f-1]) {
        ++nfibs;
      }
    }
  }
  nfibs_per_grp[mode] = nfibs;


  * fnext = (idx_t *)splatt_malloc((nfibs+1) * sizeof(idx_t));
  idx_t * tmp_fnext = * fnext;
  tmp_fnext[0] = 0;

  /* now fill in fiber info */
  nfound = 0;
  for(idx_t s=0; s < nfibs_per_grp[mode-1]; ++s) {
    idx_t const start = fprev[s]+1;
    idx_t const end = fprev[s+1];

    /* mark start of subtree */
    fprev[s] = nfound;  // jli: modify the previous fptr.
    tmp_fnext[nfound++] = start-1;

    /* mark fibers in current hyperplane */
    for(idx_t f=start; f < end; ++f) {
      if(ttind[f] != ttind[f-1]) {
        tmp_fnext[nfound++] = f;
      }
    }
  }
  assert(nfound == nfibs);
  // printf("nfound: %lu, nfibs: %lu\n", nfound, nfibs);

  /* mark end of last hyperplane */
  fprev[nfibs_per_grp[mode-1]] = nfibs;
  tmp_fnext[nfibs] = nnz;
  // iprint_array(fprev, (nfibs_per_grp[mode-1]+1), "fprev");
  // iprint_array(tmp_fnext, (nfibs+1), "tmp_fnext");

}


splatt_csf * csf_alloc_adaptive(
  sptensor_t * const tt,
  group_properties * const grp_prop,
  idx_t const n_grp,
  double const * const opts,
  idx_t * n_csf)
{
  splatt_csf * ret = NULL;
  idx_t num_csf = 0;

  if(n_grp == 1) {
    num_csf = 1;
    ret = splatt_malloc(sizeof(*ret));
    p_mk_csf_adaptive(ret, tt, grp_prop, 0);
  } else {
    switch((splatt_csf_type) opts[SPLATT_OPTION_CSF_ALLOC]) {
      case SPLATT_ADAPTIVE_TIME_EFFICIENT:
        /* Allocate space for each recomputed tensor */
        num_csf = n_grp;
        ret = splatt_malloc(num_csf * sizeof(*ret));
        for(idx_t m=0; m < num_csf; ++m) {
          p_mk_csf_adaptive(ret + m, tt, grp_prop, m);
        }
        break;

      case SPLATT_ADAPTIVE_SPACE_EFFICIENT:
        for(idx_t g=0; g<n_grp; ++g) {
          if(grp_prop[g].n_imten != 0) {
            ++ num_csf;
          }
        }
        // assert(num_csf == *n_csf);
        idx_t * mark_grp = (idx_t *)splatt_malloc(num_csf * sizeof(idx_t));
        idx_t tmp_g = 0;
        for(idx_t g=0; g<n_grp; ++g) {
          if(grp_prop[g].n_imten != 0) {
            mark_grp[tmp_g] = g;
            ++ tmp_g;
          }
        }
        assert(tmp_g == num_csf);

        /* Only allocate tensors of memoized MTTKRP, 
            other MTTKRPs use leaf algorithm to compute for space efficiency. */
        ret = splatt_malloc(num_csf * sizeof(*ret));
        for(idx_t m=0; m < num_csf; ++m) {
          p_mk_csf_adaptive(ret + m, tt, grp_prop, mark_grp[m]);
        }
        splatt_free(mark_grp);
        break;
    }
  }

  * n_csf = num_csf;

  return ret;
}


size_t csf_storage_adaptive(
  splatt_csf const * const tensors,
  idx_t const n_csf,
  group_properties * const grp_prop,
  int const n_grp)
{
  size_t bytes = 0;
  for(idx_t m=0; m < n_csf; ++m) {
    splatt_csf const * const ct = tensors + m;
    bytes += ct->nnz * sizeof(*(ct->pt->vals)); /* vals */
    bytes += ct->nnz * sizeof(**(ct->pt->fids)); /* fids[nmodes] */
    bytes += ct->ntiles * sizeof(*(ct->pt)); /* pt */

    for(idx_t t=0; t < ct->ntiles; ++t) {
      csf_sparsity const * const pt = ct->pt + t;

      for(idx_t m=0; m < ct->nmodes-1; ++m) {
        bytes += (pt->nfibs[m]+1) * sizeof(**(pt->fptr)); /* fptr */
        if(pt->fids[m] != NULL) {
          bytes += pt->nfibs[m] * sizeof(**(pt->fids)); /* fids */
        }
      }
    }
  }

  return bytes;
}


rcsf_seq_adaptive * rcsf_alloc_adaptive(
  splatt_csf * const ct,
  idx_t const nfactors,
  group_properties * const grp_prop,
  idx_t const n_grp,
  idx_t const n_csf,
  double const * const opts,
  idx_t * n_rcsf)
{
  rcsf_seq_adaptive * ret = splatt_malloc(n_csf * sizeof(*ret));
  idx_t j = 0;
  for(idx_t g=0; g<n_grp; ++g) {
    if(grp_prop[g].n_imten != 0) {
      ret[j].begin_imten = grp_prop[g].begin_imten;
      ret[j].n_imten = grp_prop[g].n_imten;
      ret[j].rcsfs = splatt_malloc(ret[j].n_imten * sizeof(*(ret[j].rcsfs)));
      p_mk_rcsf_adaptive(ret + j, ct + g, nfactors, opts);
      ++ j;
    }
  }
  assert(j <= n_csf);
  * n_rcsf = j;

  return ret;
}


size_t rcsf_storage_adaptive(
  rcsf_seq_adaptive const * const seq_rcsf,
  idx_t const n_rcsf)
{
  size_t bytes = 0;
  // Basically only add the bytes of values.
  // Loop for rcsf sequences
  for(idx_t m=0; m < n_rcsf; ++m) {
    // Loop for all the stored intermediate tensors
    splatt_csf const * const rcsfs = seq_rcsf[m].rcsfs;
    for(idx_t n=0; n < seq_rcsf[m].n_imten; ++n) {
      splatt_csf const * const ct = rcsfs+n;
      bytes += ct->nnz * sizeof(*(ct->pt->vals)); /* vals */
      assert (ct->pt->fids[ct->nmodes-1] == NULL && ct->pt->fptr[ct->nmodes-1] == NULL && ct->pt->fptr[ct->nmodes-2] == NULL );
      bytes += ct->ntiles * sizeof(*(ct->pt)); /* pt */
      // fptr and fids pointers point to CSF members or are NULL.
    }
  }

  return bytes;
}


