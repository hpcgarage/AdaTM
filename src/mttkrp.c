
/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "base.h"
#include "mttkrp.h"
#include "thd_info.h"
#include "tile.h"
#include "util.h"
#include <omp.h>


#define NLOCKS 1024
static omp_lock_t locks[NLOCKS];


/******************************************************************************
 * API FUNCTIONS
 *****************************************************************************/
int splatt_mttkrp(
    splatt_idx_t const mode,
    splatt_idx_t const ncolumns,
    splatt_csf const * const tensors,
    splatt_val_t ** matrices,
    splatt_val_t * const matout,
    double const * const options)
{
  idx_t const nmodes = tensors->nmodes;

  /* fill matrix pointers  */
  matrix_t * mats[MAX_NMODES+1];
  for(idx_t m=0; m < nmodes; ++m) {
    mats[m] = (matrix_t *) splatt_malloc(sizeof(matrix_t));
    mats[m]->I = tensors->dims[m];
    mats[m]->J = ncolumns,
    mats[m]->rowmajor = 1;
    mats[m]->vals = matrices[m];
  }
  mats[MAX_NMODES] = (matrix_t *) splatt_malloc(sizeof(matrix_t));
  mats[MAX_NMODES]->I = tensors->dims[mode];
  mats[MAX_NMODES]->J = ncolumns;
  mats[MAX_NMODES]->rowmajor = 1;
  mats[MAX_NMODES]->vals = matout;

  /* Setup thread structures. + 64 bytes is to avoid false sharing. */
  idx_t const nthreads = (idx_t) options[SPLATT_OPTION_NTHREADS];
  omp_set_num_threads(nthreads);
  thd_info * thds =  thd_init(nthreads, 3,
    (ncolumns * ncolumns * sizeof(val_t)) + 64,
    0,
    (nmodes * ncolumns * sizeof(val_t)) + 64);

  /* do the MTTKRP */
  mttkrp_csf(tensors, mats, mode, thds, options);

  /* cleanup */
  thd_free(thds, nthreads);
  for(idx_t m=0; m < nmodes; ++m) {
    free(mats[m]);
  }
  free(mats[MAX_NMODES]);

  return SPLATT_SUCCESS;
}



/******************************************************************************
 * PRIVATE FUNCTIONS
 *****************************************************************************/

static inline void p_add_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] += a[f] * b[f];
  }
}


static inline void p_add_hada_clear(
  val_t * const restrict out,
  val_t * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] += a[f] * b[f];
    a[f] = 0;
  }
}


static inline void p_assign_hada(
  val_t * const restrict out,
  val_t const * const restrict a,
  val_t const * const restrict b,
  idx_t const nfactors)
{
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = a[f] * b[f];
  }
}


static inline void p_csf_process_fiber_lock(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const nfactors,
  idx_t const start,
  idx_t const end,
  idx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * nfactors);
    val_t const v = vals[jj];
    omp_set_lock(locks + (inds[jj] % NLOCKS));
    for(idx_t f=0; f < nfactors; ++f) {
      leafrow[f] += v * accumbuf[f];
    }
    omp_unset_lock(locks + (inds[jj] % NLOCKS));
  }
}

static inline void p_csf_process_fiber_nolock(
  val_t * const leafmat,
  val_t const * const restrict accumbuf,
  idx_t const nfactors,
  idx_t const start,
  idx_t const end,
  idx_t const * const restrict inds,
  val_t const * const restrict vals)
{
  for(idx_t jj=start; jj < end; ++jj) {
    val_t * const restrict leafrow = leafmat + (inds[jj] * nfactors);
    val_t const v = vals[jj];
    for(idx_t f=0; f < nfactors; ++f) {
      leafrow[f] += v * accumbuf[f];
    }
  }
}


static inline void p_csf_process_fiber(
  val_t * const restrict accumbuf,
  idx_t const nfactors,
  val_t const * const leafmat,
  idx_t const start,
  idx_t const end,
  idx_t const * const inds,
  val_t const * const vals)
{
  /* foreach nnz in fiber */
  for(idx_t j=start; j < end; ++j) {
    val_t const v = vals[j] ;
    val_t const * const restrict row = leafmat + (nfactors * inds[j]);
    for(idx_t f=0; f < nfactors; ++f) {
      accumbuf[f] += v * row[f];
    }
  }
}



static inline void p_propagate_up_reuse(
  val_t * const out,
  val_t * const * const buf,
  idx_t * const restrict idxstack, // jli: store the start location for each mode
  idx_t const init_depth,
  idx_t const init_idx,
  idx_t const * const * const fp,
  idx_t const * const * const fids,
  val_t const * const restrict vals,
  val_t ** mvals,
  idx_t const nmodes,
  idx_t const nfactors)
{
  /* push initial idx initialize idxstack */
  idx_t const valid_nmodes = nmodes-1;

  idxstack[init_depth] = init_idx;
  assert(init_depth < nmodes-1);
  for(idx_t m=init_depth+1; m < nmodes-1; ++m) {  // Not count the last mode.
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }
  idxstack[nmodes-1] = 0;  // Don't use the last mode

  /* clear out accumulation buffer */
  for(idx_t f=0; f < nfactors; ++f) {
    buf[init_depth+1][f] = 0;
  }

  while(idxstack[init_depth+1] < fp[init_depth][init_idx+1]) {
    /* skip to last internal mode */
    idx_t depth = nmodes - 3; // fp on mode-(nmodes-2) points to mode-(nmodes-1)

    idx_t const start = fp[depth][idxstack[depth]];
    idx_t const end = fp[depth][idxstack[depth]+1];
    // idx_t const start = idxstack[depth+1];
    val_t const * restrict fibrow
        = mvals[depth+1] + (fids[depth+1][start] * nfactors);

    p_assign_hada(buf[depth+1], vals+start*nfactors, fibrow, nfactors);
    for (idx_t ln_idx = start+1; ln_idx < end; ++ln_idx) {
      fibrow = mvals[depth+1] + (fids[depth+1][ln_idx] * nfactors);
      p_add_hada(buf[depth+1], vals+ln_idx*nfactors, fibrow, nfactors);
    }
    idxstack[depth+1] = end;

    if ( init_depth < nmodes-3 ) {
      /* Propagate up until we reach a node with more children to process */
      do {
        /* propagate result up and clear buffer for next sibling */
          val_t const * const restrict fibrow
              = mvals[depth] + (fids[depth][idxstack[depth]] * nfactors);
          p_add_hada_clear(buf[depth], buf[depth+1], fibrow, nfactors);

        ++idxstack[depth];
        --depth;
      } while(depth > init_depth &&
        idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
      
    }
  } /* end DFS */

  /* copy to out */
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = buf[init_depth+1][f];
  }

  return;
}




static inline void p_propagate_up(
  val_t * const out,
  val_t * const * const buf,
  idx_t * const restrict idxstack, // jli: store the start location for each mode
  idx_t const init_depth,
  idx_t const init_idx,
  idx_t const * const * const fp,
  idx_t const * const * const fids,
  val_t const * const restrict vals,
  val_t ** mvals,
  idx_t const nmodes,
  idx_t const nfactors)
{
  /* push initial idx initialize idxstack */
  idxstack[init_depth] = init_idx;
  for(idx_t m=init_depth+1; m < nmodes; ++m) {
    idxstack[m] = fp[m-1][idxstack[m-1]];
  }

  assert(init_depth < nmodes-1);

  /* clear out accumulation buffer */
  for(idx_t f=0; f < nfactors; ++f) {
    buf[init_depth+1][f] = 0;
  }

  while(idxstack[init_depth+1] < fp[init_depth][init_idx+1]) {
    /* skip to last internal mode */
    idx_t depth = nmodes - 2; // fp on mode-(nmodes-2) points to mode-(nmodes-1)

    /* process all nonzeros [start, end) into buf[depth]*/
    idx_t const start = fp[depth][idxstack[depth]];
    idx_t const end   = fp[depth][idxstack[depth]+1];
    p_csf_process_fiber(buf[depth+1], nfactors, mvals[depth+1],
        start, end, fids[depth+1], vals);

    idxstack[depth+1] = end;

    /* exit early if there is no propagation to do... */
    if(init_depth == nmodes-2) {
      for(idx_t f=0; f < nfactors; ++f) {
        out[f] = buf[depth+1][f];
      }
      return;
    }

    /* Propagate up until we reach a node with more children to process */
    do {
      /* propagate result up and clear buffer for next sibling */
      val_t const * const restrict fibrow
          = mvals[depth] + (fids[depth][idxstack[depth]] * nfactors);
      p_add_hada_clear(buf[depth], buf[depth+1], fibrow, nfactors);

      ++idxstack[depth];
      --depth;
    } while(depth > init_depth &&
        idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
  } /* end DFS */

  /* copy to out */
  for(idx_t f=0; f < nfactors; ++f) {
    out[f] = buf[init_depth+1][f];
  }
}



static void p_csf_mttkrp_root_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[1]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t * const restrict mv = ovals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* scale inner products by row of A and update to M */
      val_t const * const restrict av = avals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        mv[r] += accumF[r] * av[r];
      }
    }
  }
}


static void p_csf_mttkrp_root3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[1]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    val_t * const restrict mv = ovals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* scale inner products by row of A and update to M */
      val_t const * const restrict av = avals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        mv[r] += accumF[r] * av[r];
      }
    }
  }
}




static void p_csf_mttkrp_root3_reuse_adaptive(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  rcsf_seq_adaptive * const seq_rcsfs,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  int const degree = seq_rcsfs->n_imten;
  splatt_csf * const rcsf = seq_rcsfs->rcsfs; // zero or one rcsf.
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[1]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const rcsf_vals = (degree == 0) ? NULL : rcsf->pt[tile_id].vals;   
  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];


  if (rcsf_vals != NULL) {
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];
      val_t * const restrict mv = ovals + (fid * nfactors);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        // jli: save intermediate values in rcsf.
        val_t * const restrict rcsf_vv = rcsf_vals + (f * nfactors);
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          rcsf_vv[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            rcsf_vv[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          mv[r] += rcsf_vv[r] * av[r];
        }
      }
    }
  }
  else {  // no rcsf need to store.
    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      idx_t const fid = (sids == NULL) ? s : sids[s];
      val_t * const restrict mv = ovals + (fid * nfactors);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        // jli: save intermediate values in rcsf.
        val_t * const restrict rcsf_vv = rcsf_vals + (f * nfactors);
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
          for(idx_t r=0; r < nfactors; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }
  }
  return;
}


static void p_csf_mttkrp_internal3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* write to fiber row */
      val_t * const restrict ov = ovals  + (fids[f] * nfactors);
      omp_set_lock(locks + (fids[f] % NLOCKS));
      for(idx_t r=0; r < nfactors; ++r) {
        ov[r] += rv[r] * accumF[r];
      }
      omp_unset_lock(locks + (fids[f] % NLOCKS));
    }
  }
}




static void p_csf_mttkrp_internal3_reuse_adaptive(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  rcsf_seq_adaptive const * const seq_rcsfs,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);

  int const degree = seq_rcsfs->n_imten;
  splatt_csf const * const rcsf = seq_rcsfs->rcsfs; // only one rcsf
  assert (degree == 1);

  val_t const * const vals = rcsf->pt[tile_id].vals;

  idx_t const * const restrict sptr = rcsf->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = rcsf->pt[tile_id].fptr[1];  // NULL

  idx_t const * const restrict sids = rcsf->pt[tile_id].fids[0];  // NULL
  idx_t const * const restrict fids = rcsf->pt[tile_id].fids[1];
  idx_t const * const restrict inds = rcsf->pt[tile_id].fids[2];  // NULL
  assert (fptr == NULL && fids == NULL && inds == NULL);

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  idx_t const nslices = rcsf->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* write to fiber row */
      val_t * const restrict ov = ovals  + (fids[f] * nfactors);
      val_t const * const rcsf_vv = vals + (f * nfactors);
      // dprint_array(ov, nfactors, "ov");
      // dprint_array(rv, nfactors, "rv");
      // dprint_array(rcsf_vv, nfactors, "rcsf_vv");

      omp_set_lock(locks + (fids[f] % NLOCKS));
      for(idx_t r=0; r < nfactors; ++r) {
        ov[r] += rv[r] * rcsf_vv[r];
      }
      omp_unset_lock(locks + (fids[f] % NLOCKS));
      // dprint_array(ov, nfactors, "ov");
      // print_mat(mats[MAX_NMODES]);
      // printf("\n");
    }
  }
}


static void p_csf_mttkrp_leaf3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[1]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* fill fiber with hada */
      val_t const * const restrict av = bvals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = rv[r] * av[r];
      }

      /* foreach nnz in fiber, scale with hada and write to ovals */
      for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t * const restrict ov = ovals + (inds[jj] * nfactors);
        omp_set_lock(locks + (inds[jj] % NLOCKS));
        for(idx_t r=0; r < nfactors; ++r) {
          ov[r] += v * accumF[r];
        }
        omp_unset_lock(locks + (inds[jj] % NLOCKS));
      }
    }
  }
}


static void p_csf_mttkrp_root_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root_tiled3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }

  val_t * const ovals = mats[MAX_NMODES]->vals;

  idx_t const nfibs = ct->pt[tile_id].nfibs[0];
  assert(nfibs <= mats[MAX_NMODES]->I);

  for(idx_t s=0; s < nfibs; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    assert(fid < mats[MAX_NMODES]->I);

    p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
        vals, mvals, nmodes, nfactors);

    val_t * const restrict orow = ovals + (fid * nfactors);
    val_t const * const restrict obuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      orow[f] += obuf[f];
    }
  } /* end foreach outer slice */
}



static void p_csf_mttkrp_root(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }

  val_t * const ovals = mats[MAX_NMODES]->vals;

  idx_t const nfibs = ct->pt[tile_id].nfibs[0];
  assert(nfibs <= mats[MAX_NMODES]->I);

  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nfibs; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    assert(fid < mats[MAX_NMODES]->I);

    p_propagate_up(buf[0], buf, idxstack, 0, s, fp, fids,
        vals, mvals, nmodes, nfactors);

    val_t * const restrict orow = ovals + (fid * nfactors);
    val_t const * const restrict obuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      orow[f] += obuf[f];
    }
  } /* end foreach outer slice */
}



static void p_spt2t_add_hada (
  // splatt_csf * const out_rcsf,
  val_t * const out_vals,
  // splatt_csf const * const in_rcsf,
  idx_t const nslices,
  idx_t const * const in_sptr,
  idx_t const * const in_fids,
  val_t const * const in_vals,
  matrix_t const * const mats)
{
  idx_t const nfactors = mats->J;
  // idx_t const in_nmodes = in_rcsf->nmodes;
  // idx_t const out_nmodes = out_rcsf->nmodes;
  // assert (out_nmodes == in_nmodes - 1);
  // assert (in_rcsf->dims[in_nmodes-1] == nfactors);
  // assert (in_rcsf->dims[in_nmodes-2] == mats->I);
  // assert (out_rcsf->nnz == in_rcsf->pt->nfibs[in_nmodes-3] * nfactors);

  // point to slices, slice: mode-(N-2) * mode-(N-1)
  // idx_t const nslices = in_rcsf->pt->nfibs[in_nmodes-3];
  // idx_t const * const in_sptr = in_rcsf->pt->fptr[in_nmodes-3];
  // idx_t const * const in_fids = in_rcsf->pt->fids[in_nmodes-2];
  // val_t const * const in_vals = in_rcsf->pt->vals;
  val_t const * const mat_vals = mats->vals;
  // val_t * const out_vals = out_rcsf->pt->vals;

  // Loop on mode-(N-3), loop slices
  #pragma omp for schedule(dynamic, 16) nowait
  for (idx_t s=0; s<nslices; ++s) {
    val_t * const restrict out_row = out_vals + (nfactors * s);
    // Loop on mode-(N-2), loop fibers
    for ( idx_t f=in_sptr[s]; f<in_sptr[s+1]; ++f) {
      val_t * const restrict in_row = in_vals + (nfactors * f);
      val_t * const restrict mat_row = mat_vals + (nfactors * in_fids[f]);
      for(idx_t r=0; r<nfactors; ++r) {
        out_row[r] += in_row[r] * mat_row[r];
      }
    }
  }

}



static void p_spt2m_add_hada (
  matrix_t * const omats,
  // splatt_csf const * const rcsf,
  idx_t const nslices,
  idx_t const * const sptr,
  idx_t const * const fids,
  val_t const * const rvals,
  matrix_t const * const imats)
{
  idx_t const nfactors = imats->J;
  // idx_t const nmodes = rcsf->nmodes;
  // assert (nmodes == 3);
  assert (omats->J == nfactors);
  // assert (rcsf->dims[nmodes-1] == nfactors);
  // assert (rcsf->dims[nmodes-2] == imats->I);
  assert (rcsf->dims[0] == omats->I);

  // point to slices, slice: mode-(N-2) * mode-(N-1)
  // idx_t const nslices = rcsf->pt->nfibs[0];
  // idx_t const * const sptr = rcsf->pt->fptr[0];
  // idx_t const * const fids = rcsf->pt->fids[1];
  // val_t const * const rvals = rcsf->pt->vals;
  val_t const * const ivals = imats->vals;
  val_t * const ovals = omats->vals;

  /* if mode-0 is sparse, then some rows of omats are 0s. */
  // Loop on mode-(N-3), loop slices
  #pragma omp for schedule(dynamic, 16) nowait
  for (idx_t s=0; s<nslices; ++s) {
    val_t * const restrict orow = ovals + (nfactors * s);
    // Loop on mode-(N-2), loop fibers
    for ( idx_t f=sptr[s]; f<sptr[s+1]; ++f) {
      val_t * const restrict rrow = rvals + (nfactors * f);
      val_t * const restrict irow = ivals + (nfactors * fids[f]);
      for(idx_t r=0; r<nfactors; ++r) {
        orow[r] += rrow[r] * irow[r];
      }
    }
  }

}


static void p_spttm (
  // splatt_csf * const out_rcsf,
  val_t * const out_vals,
  splatt_csf const * const ct,
  matrix_t * const mats,
  idx_t const mode)
{
  assert (mode == nmodes-1);  // TODO: we only support TTM on N-1 now.
  assert (ct->dims[mode] == mats->I);
  idx_t nfactors = mats->J;
  // assert (out_rcsf->nnz == cf->pt->nfibs[mode-1] * nfactors);

  // assert (out_rcsf->pt->fptr[mode-2] == ct_sptr);
  idx_t const nslices = ct->pt->nfibs[mode-2];  // mode-(N-3), slice: mode-(N-2) * mode-(N-1)
  idx_t * const ct_sptr = ct->pt->fptr[mode-2]; // Point to mode-(N-2), slice pointer
  idx_t * const ct_fptr = ct->pt->fptr[mode-1]; // Point to mode-(N-1)
  idx_t * const ct_fids = ct->pt->fids[mode];
  val_t * const ct_vals = ct->pt->vals;
  val_t * const mat_vals = mats->vals;
  // val_t * const out_vals = out_rcsf->pt->vals;
  

  #pragma omp for schedule(dynamic, 16) nowait
  // Loop on mode-(N-3), loop slices. The slices is much more than the first level. 
  for (idx_t s=0; s<nslices; ++s) { 
    // Loop on mode-(N-2), loop fibers
    for ( idx_t f=ct_sptr[s]; f<ct_sptr[s+1]; ++f) {  
      val_t * const restrict out_row = out_vals + (nfactors * f);

      // Loop on mode-(N-1)
      for ( idx_t j=ct_fptr[f]; j<ct_fptr[f+1]; ++j) {
        val_t const v = ct_vals[j];
        val_t * const restrict mat_row = mat_vals + (nfactors * ct_fids[j]);
        for(idx_t r=0; r<nfactors; ++r) {
          out_row[r] += v * mat_row[r];
        }
      }
    }
  }

}





static void p_csf_mttkrp_root_genreuse_adaptive(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  rcsf_seq_adaptive * const seq_rcsfs,
  thd_info * const thds)
{
  idx_t const degree = seq_rcsfs->n_imten;
  idx_t const begin_imten = seq_rcsfs->begin_imten;
  assert (degree == ct->nmodes-2);

  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;

  /* empty tile, just return */
  if(vals == NULL) {
    return;
  }

  if(nmodes == 3) {
    p_csf_mttkrp_root3_reuse_adaptive(ct, tile_id, mats, seq_rcsfs, thds);  // degree = 1 or 0 here
    return;
  }

  assert (degree >= 1);
  splatt_csf * const rcsfs = seq_rcsfs->rcsfs;

  // when nmodes > 3
  // idx_t const * const * const restrict fp
  //     = (idx_t const * const *) ct->pt[tile_id].fptr;
  // idx_t const * const * const restrict fids
  //     = (idx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  if ( degree == nmodes-2) {  // Store all intermediate RCSFs
    sp_timer_t tmptime, spttm_timer, hada_timer;
    // timer_fstart(&tmptime);

    // timer_fstart (&spttm_timer);
    p_spttm ((rcsfs+0)->pt[tile_id].vals, ct, mats[ct->dim_perm[nmodes-1]], nmodes-1);
    // timer_stop (&spttm_timer);
    // printf("spttm_time: %f\n", spttm_timer.seconds);

    // timer_fstart (&hada_timer);
    for (idx_t m=nmodes-2; m>=2; --m) {
      idx_t rloc = nmodes-1-m;  // rcsf location is different from computation order.
      matrix_t * hmat = mats[ct->dim_perm[m]];

      // p_spt2t_add_hada (rcsfs+rloc, rcsfs+rloc-1, hmat);
      // sp_timer_t onehada_timer;
      // timer_fstart (&onehada_timer);
      idx_t const in_nmodes = (rcsfs+rloc-1)->nmodes;
      idx_t nslices = (rcsfs+rloc-1)->pt[tile_id].nfibs[in_nmodes-3];
      idx_t const * in_sptr = (rcsfs+rloc-1)->pt[tile_id].fptr[in_nmodes-3];
      idx_t const * in_fids = (rcsfs+rloc-1)->pt[tile_id].fids[in_nmodes-2];
      val_t const * in_vals = (rcsfs+rloc-1)->pt[tile_id].vals;
      p_spt2t_add_hada ((rcsfs+rloc)->pt[tile_id].vals, nslices, in_sptr, in_fids, in_vals, hmat);
      // timer_stop(&onehada_timer);
      // printf("onehada_time: %f\n", onehada_timer.seconds);
    }

    // p_spt2m_add_hada (mats[MAX_NMODES], rcsfs+nmodes-3, mats[ct->dim_perm[1]]);
    assert ((rcsfs+nmodes-3)->nmodes == 3);
    idx_t nslices = ct->pt[tile_id].nfibs[0];
    idx_t const * in_sptr = ct->pt[tile_id].fptr[0];
    idx_t const * in_fids = ct->pt[tile_id].fids[1];
    p_spt2m_add_hada (mats[MAX_NMODES], nslices, in_sptr, in_fids, (rcsfs+degree-1)->pt[tile_id].vals, mats[ct->dim_perm[1]]);

    // timer_stop (&hada_timer);
    // printf("hada_time: %f\n", hada_timer.seconds);

    // timer_stop(&tmptime);
    // printf("tmptime: %f\n", tmptime.seconds);
  }
  else {  // Only store useful RCSFs
    sp_timer_t tmptime, spttm_timer, hada_timer;
    timer_fstart(&tmptime);

    idx_t const max_nvals = ct->pt[tile_id].nfibs[nmodes-2] * nfactors;
    val_t * tmp_vals = (val_t *)splatt_malloc (max_nvals * sizeof(val_t));
    memset (tmp_vals, 0, max_nvals * sizeof(val_t));
    idx_t const max_nvals_2 = ct->pt[tile_id].nfibs[nmodes-3] * nfactors;
    val_t * tmp_vals_2 = (val_t *)splatt_malloc (max_nvals_2 * sizeof(val_t));
    memset (tmp_vals_2, 0, max_nvals_2 * sizeof(val_t));

    // timer_fstart (&spttm_timer);
    /* mode-(nmodes-1), do SpTTM.
     * Not be saved in RCSF in TWOMODE case.
     */ 
    p_spttm (tmp_vals, ct, mats[ct->dim_perm[nmodes-1]], nmodes-1);
    // timer_stop (&spttm_timer);
    // printf("spttm_time: %f\n", spttm_timer.seconds);
    // printf("mode-%d\n", nmodes-1);
    // dprint_array (tmp_vals, ct->pt[tile_id].nfibs[nmodes-2]*nfactors, "tmp_vals");

    // timer_fstart (&hada_timer);
    /* mode-(nmodes-2), ... , rmodes[0]+1, do a sequence of Hada-Reduction.
     * The resulting values still don't be saved in TWOMODE case.
     */ 
    for (idx_t m=nmodes-2; m>begin_imten; --m) {
      // sp_timer_t onehada_timer, onememcpy_timer;
      // timer_fstart (&onehada_timer);
      memset(tmp_vals_2, 0, ct->pt[tile_id].nfibs[m-1]*nfactors * sizeof(val_t));
      matrix_t * hmat = mats[ct->dim_perm[m]];
      idx_t nslices = ct->pt[tile_id].nfibs[m-1];
      idx_t const * in_sptr = ct->pt[tile_id].fptr[m-1];
      idx_t const * in_fids = ct->pt[tile_id].fids[m];
      p_spt2t_add_hada (tmp_vals_2, nslices, in_sptr, in_fids, tmp_vals, hmat);
      // timer_stop(&onehada_timer);
      // printf("onehada_time: %f\n", onehada_timer.seconds);
      // timer_fstart (&onememcpy_timer);
      memcpy(tmp_vals, tmp_vals_2, ct->pt[tile_id].nfibs[m-1]*nfactors * sizeof(val_t));
      // printf("mode-%d\n", m);
      // dprint_array (tmp_vals, ct->pt[tile_id].nfibs[m-1]*nfactors, "tmp_vals");
      // timer_stop (&onememcpy_timer);
      // printf("onememcpy_time: %f\n", onememcpy_timer.seconds);
    }
    /* mode-rmodes[0], do a Hada-Reduction.
     * The resulting values are saved as RCSF[0].
     */
    matrix_t * hmat = mats[ct->dim_perm[begin_imten]];
    idx_t nslices = ct->pt[tile_id].nfibs[begin_imten-1];
    idx_t const * in_sptr = ct->pt[tile_id].fptr[begin_imten-1];
    idx_t const * in_fids = ct->pt[tile_id].fids[begin_imten];
    p_spt2t_add_hada (rcsfs->pt[tile_id].vals, nslices, in_sptr, in_fids, tmp_vals, hmat);
      // printf("mode-%d\n", begin_imten);
      // dprint_array (rcsfs->pt[tile_id].vals, nslices*nfactors, "rcsfs->pt[tile_id].vals");

    /* mode-(rmodes[1]), ..., (rmodes[degree-1]), do a sequence of Hada-Reduction.
     * The resulting values are saved as RCSF[1], ... , RCSF[degree-1].
     */
    for (idx_t rloc=1; rloc<degree; ++rloc) {
      // idx_t mloc = rmodes[rloc];  // rcsf location is different from computation order.
      idx_t mloc = begin_imten - rloc;
      matrix_t * hmat = mats[ct->dim_perm[mloc]];
      idx_t const in_nmodes = (rcsfs+rloc-1)->nmodes;
      idx_t nslices = (rcsfs+rloc-1)->pt[tile_id].nfibs[in_nmodes-3];
      idx_t const * in_sptr = (rcsfs+rloc-1)->pt[tile_id].fptr[in_nmodes-3];
      idx_t const * in_fids = (rcsfs+rloc-1)->pt[tile_id].fids[in_nmodes-2];
      val_t const * in_vals = (rcsfs+rloc-1)->pt[tile_id].vals;
      p_spt2t_add_hada ((rcsfs+rloc)->pt[tile_id].vals, nslices, in_sptr, in_fids, in_vals, hmat);
      // p_spt2t_add_hada (rcsfs+rloc, rcsfs+rloc-1, hmat);
      // printf("mode-%d\n", mloc);
      // dprint_array ((rcsfs+rloc)->pt[tile_id].vals, nslices*nfactors, "(rcsfs+rloc)->pt[tile_id].vals");
    }

    /* if degree is partial. We cannot store all useful intermediate RCSFs. */
    /* mode-(rmodes[degree-1]-1), do a sequence of Hada-Reduction.
     * The resulting values still don't be saved in TWOMODE case.
     */
    idx_t cur_mode = begin_imten - degree;
    if (cur_mode > 1) {
      hmat = mats[ct->dim_perm[cur_mode]];
      nslices = ct->pt[tile_id].nfibs[cur_mode-1];
      in_sptr = ct->pt[tile_id].fptr[cur_mode-1];
      in_fids = ct->pt[tile_id].fids[cur_mode];
      memset(tmp_vals, 0, nslices*nfactors * sizeof(val_t));
      p_spt2t_add_hada (tmp_vals, nslices, in_sptr, in_fids, (rcsfs+degree-1)->pt[tile_id].vals, hmat);
        // printf("mode-%d\n", cur_mode);
        // dprint_array (tmp_vals, nslices*nfactors, "tmp_vals");

      for (idx_t m=cur_mode-1; m>1; --m) {  // Last one is one mode-2
        // sp_timer_t onehada_timer, onememcpy_timer;
        // timer_fstart (&onehada_timer);
        memset(tmp_vals_2, 0, ct->pt[tile_id].nfibs[m-1]*nfactors * sizeof(val_t));
        matrix_t * hmat = mats[ct->dim_perm[m]];
        idx_t nslices = ct->pt[tile_id].nfibs[m-1];
        idx_t const * in_sptr = ct->pt[tile_id].fptr[m-1];
        idx_t const * in_fids = ct->pt[tile_id].fids[m];
        p_spt2t_add_hada (tmp_vals_2, nslices, in_sptr, in_fids, tmp_vals, hmat);
        // timer_stop(&onehada_timer);
        // printf("onehada_time-2: %f\n", onehada_timer.seconds);
        // timer_fstart (&onememcpy_timer);
        memcpy(tmp_vals, tmp_vals_2, ct->pt[tile_id].nfibs[m-1]*nfactors * sizeof(val_t));
        // printf("mode-%d\n", m);
        // dprint_array (tmp_vals, nslices*nfactors, "tmp_vals");
        // timer_stop (&onememcpy_timer);
        // printf("onememcpy_time-2: %f\n", onememcpy_timer.seconds);
      }

      /* Last Hadamard-reduction */
      // p_spt2m_add_hada (mats[MAX_NMODES], rcsfs+degree-1, mats[ct->dim_perm[1]]);
      nslices = ct->pt[tile_id].nfibs[0];
      in_sptr = ct->pt[tile_id].fptr[0];
      in_fids = ct->pt[tile_id].fids[1];
      p_spt2m_add_hada (mats[MAX_NMODES], nslices, in_sptr, in_fids, tmp_vals, mats[ct->dim_perm[1]]);
        // printf("mode-1\n");
        // dprint_array (mats[MAX_NMODES]->vals, nslices*nfactors, "mats[MAX_NMODES]->vals");
    }
    else {
      /* Last Hadamard-reduction */
      // p_spt2m_add_hada (mats[MAX_NMODES], rcsfs+degree-1, mats[ct->dim_perm[1]]);
      nslices = ct->pt[tile_id].nfibs[0];
      in_sptr = ct->pt[tile_id].fptr[0];
      in_fids = ct->pt[tile_id].fids[1];
      p_spt2m_add_hada (mats[MAX_NMODES], nslices, in_sptr, in_fids, (rcsfs+degree-1)->pt[tile_id].vals, mats[ct->dim_perm[1]]);
    }

    // timer_stop (&hada_timer);
    // printf("hada_time: %f\n", hada_timer.seconds);

    splatt_free(tmp_vals);
    splatt_free(tmp_vals_2);

    // timer_stop(&tmptime);
    // printf("tmptime: %f\n", tmptime.seconds);
  }

}



static void p_csf_mttkrp_leaf_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[1]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* fill fiber with hada */
      val_t const * const restrict av = bvals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = rv[r] * av[r];
      }

      /* foreach nnz in fiber, scale with hada and write to ovals */
      for(idx_t jj=fptr[f]; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t * const restrict ov = ovals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          ov[r] += v * accumF[r];
        }
      }
    }
  }
}




static void p_csf_mttkrp_leaf_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf_tiled3(ct, tile_id, mats, thds);
    return;
  }

  /* extract tensor structures */
  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
  }

  /* foreach outer slice */
  idx_t const nouter = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nouter; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const rootrow = mvals[0] + (fid*nfactors);
    val_t * const rootbuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      rootbuf[f] = rootrow[f];
    }

    idx_t depth = 0;

    idx_t const outer_end = fp[0][s+1];
    while(idxstack[1] < outer_end) {
      /* move down to an nnz node */
      for(; depth < nmodes-2; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      p_csf_process_fiber_nolock(mats[MAX_NMODES]->vals, buf[depth],
          nfactors, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


static void p_csf_mttkrp_leaf(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  /* extract tensor structures */
  val_t const * const vals = ct->pt[tile_id].vals;
  idx_t const nmodes = ct->nmodes;

  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_leaf3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t)); // jli: added
  }

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];
    idxstack[0] = s;

    /* clear out stale data */
    for(idx_t m=1; m < nmodes-1; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    val_t * const rootbuf = buf[0];
    for(idx_t f=0; f < nfactors; ++f) {
      rootbuf[f] = rootrow[f];
    }

    idx_t depth = 0;

    idx_t const outer_end = fp[0][s+1];
    while(idxstack[1] < outer_end) {
      /* move down to an nnz node */
      for(; depth < nmodes-2; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* process all nonzeros [start, end) */
      idx_t const start = fp[depth][idxstack[depth]];
      idx_t const end   = fp[depth][idxstack[depth]+1];
      p_csf_process_fiber_lock(mats[MAX_NMODES]->vals, buf[depth],
          nfactors, start, end, fids[depth+1], vals);

      /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */
}


static void p_csf_mttkrp_internal_tiled3(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  thd_info * const thds)
{
  assert(ct->nmodes == 3);
  val_t const * const vals = ct->pt[tile_id].vals;

  idx_t const * const restrict sptr = ct->pt[tile_id].fptr[0];
  idx_t const * const restrict fptr = ct->pt[tile_id].fptr[1];

  idx_t const * const restrict sids = ct->pt[tile_id].fids[0];
  idx_t const * const restrict fids = ct->pt[tile_id].fids[1];
  idx_t const * const restrict inds = ct->pt[tile_id].fids[2];

  val_t const * const avals = mats[ct->dim_perm[0]]->vals;
  val_t const * const bvals = mats[ct->dim_perm[2]]->vals;
  val_t * const ovals = mats[MAX_NMODES]->vals;
  idx_t const nfactors = mats[MAX_NMODES]->J;

  val_t * const restrict accumF
      = (val_t *) thds[omp_get_thread_num()].scratch[0];

  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (sids == NULL) ? s : sids[s];

    /* root row */
    val_t const * const restrict rv = avals + (fid * nfactors);

    /* foreach fiber in slice */
    for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
      /* first entry of the fiber is used to initialize accumF */
      idx_t const jjfirst  = fptr[f];
      val_t const vfirst   = vals[jjfirst];
      val_t const * const restrict bv = bvals + (inds[jjfirst] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        accumF[r] = vfirst * bv[r];
      }

      /* foreach nnz in fiber */
      for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
        val_t const v = vals[jj];
        val_t const * const restrict bv = bvals + (inds[jj] * nfactors);
        for(idx_t r=0; r < nfactors; ++r) {
          accumF[r] += v * bv[r];
        }
      }

      /* write to fiber row */
      val_t * const restrict ov = ovals  + (fids[f] * nfactors);
      for(idx_t r=0; r < nfactors; ++r) {
        ov[r] += rv[r] * accumF[r];
      }
    }
  }
}


static void p_csf_mttkrp_internal_tiled(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal_tiled3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;

  idx_t const nfactors = mats[0]->J;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    /* push outer slice and fill stack */
    idxstack[0] = s;
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* fill first buf */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buf[0][f] = rootrow[f];
    }

    /* process entire subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][s+1]) {
      /* propagate values down to outdepth-1 */
      for(; depth < outdepth; ++depth) {
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* write to output and clear buf[outdepth] for next subtree */
      idx_t const noderow = fids[outdepth][idxstack[outdepth]];

      /* propagate value up to buf[outdepth] */
      p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
          fp, fids, vals, mvals, nmodes, nfactors);

      val_t * const restrict outbuf = ovals + (noderow * nfactors);
      p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);

      /* backtrack to next unfinished node */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end foreach outer slice */
}


static void p_csf_mttkrp_internal(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes = ct->nmodes;
  val_t const * const vals = ct->pt[tile_id].vals;
  /* pass empty tiles */
  if(vals == NULL) {
    return;
  }
  if(nmodes == 3) {
    p_csf_mttkrp_internal3(ct, tile_id, mats, thds);
    return;
  }

  idx_t const * const * const restrict fp
      = (idx_t const * const *) ct->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) ct->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes);

  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t));
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = ct->pt[tile_id].nfibs[0];
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    /* push outer slice and fill stack */
    idxstack[0] = s;
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* fill first buf */
    val_t const * const restrict rootrow = mvals[0] + (fid*nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buf[0][f] = rootrow[f];
    }

    /* process entire subtree */
    idx_t depth = 0;
    while(idxstack[1] < fp[0][s+1]) {
      /* propagate values down to outdepth-1 */
      for(; depth < outdepth; ++depth) {
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      /* write to output and clear buf[outdepth] for next subtree */
      idx_t const noderow = fids[outdepth][idxstack[outdepth]];

      /* propagate value up to buf[outdepth] */
      p_propagate_up(buf[outdepth], buf, idxstack, outdepth,idxstack[outdepth],
          fp, fids, vals, mvals, nmodes, nfactors);

      val_t * const restrict outbuf = ovals + (noderow * nfactors);
      omp_set_lock(locks + (noderow % NLOCKS));
      p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);
      omp_unset_lock(locks + (noderow % NLOCKS));

      /* backtrack to next unfinished node */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end foreach outer slice */
}






static void p_csf_mttkrp_internal_reuse_adaptive(
  splatt_csf const * const ct,
  idx_t const tile_id,
  matrix_t ** mats,
  rcsf_seq_adaptive const * const seq_rcsfs,
  idx_t const mode,
  thd_info * const thds)
{
  /* extract tensor structures */
  idx_t const nmodes_ct = ct->nmodes;

  /* pass empty tiles */
  if(ct->pt[tile_id].vals == NULL) {
    return;
  }
  if(nmodes_ct == 3) {
    p_csf_mttkrp_internal3_reuse_adaptive(ct, tile_id, mats, seq_rcsfs, thds);
    return;
  }


  // nmodes > 3
  idx_t const degree = seq_rcsfs->n_imten;
  idx_t const begin_imten = seq_rcsfs->begin_imten;
  splatt_csf const * const rcsfs = seq_rcsfs->rcsfs;

  // rdims: saved rcsfs. cur_rdims: needed rcsfs. 
  // Compare the two to get in_deg, reused rcsfs, and the rest modes need to process.
  // Update rdims to cur_rdims.
  // reuse the results rcsfs[0, ..., indeg-1].
  // modes used in the current MTTKRP, using mode location in CSF tree to represent.
  idx_t * cur_rdims = (idx_t *)splatt_malloc( (nmodes_ct-1) * sizeof(idx_t));  
  /* find out which level in the tree this is */
  idx_t outdepth = csf_mode_depth(mode, ct->dim_perm, nmodes_ct);

  rcsf_reverse_mode_order(begin_imten, outdepth, cur_rdims);  //the first entry shows the reused rcsfs.
  idx_t indeg = 0;
  for (idx_t i=0; i<degree; ++i) {
    if ( begin_imten - i == cur_rdims[i] )
      ++ indeg;
  }
  free(cur_rdims);

  // Don't write back to rcsf, write in buf
  splatt_csf const * const reuse_rcsf = rcsfs + indeg - 1;  // The actual reused rcsf, which is the last possible one.
  idx_t const nmodes = reuse_rcsf->nmodes;  // nmodes of reusef_rcsf
  idx_t const valid_nmodes = nmodes-1;  // the last mode is for the vector with the length of nfactors.
  assert (valid_nmodes >= 2);
  /* pass empty tiles */
  if(reuse_rcsf->pt[tile_id].vals == NULL) {  // TODO: may have problem when tiling.
    return;
  }

  val_t const * const vals = reuse_rcsf->pt[tile_id].vals;
  idx_t const * const * const restrict fp
      = (idx_t const * const *) reuse_rcsf->pt[tile_id].fptr;
  idx_t const * const * const restrict fids
      = (idx_t const * const *) reuse_rcsf->pt[tile_id].fids;
  idx_t const nfactors = mats[0]->J;

  // TODO: allocate too much more space.
  val_t * mvals[MAX_NMODES];
  val_t * buf[MAX_NMODES];
  idx_t idxstack[MAX_NMODES];

  int const tid = omp_get_thread_num();
  for(idx_t m=0; m < nmodes_ct; ++m) {
    mvals[m] = mats[ct->dim_perm[m]]->vals;
    /* grab the next row of buf from thds */
    buf[m] = ((val_t *) thds[tid].scratch[2]) + (nfactors * m);
    memset(buf[m], 0, nfactors * sizeof(val_t)); // jli: added
  }
  val_t * const ovals = mats[MAX_NMODES]->vals;

  /* foreach outer slice */
  idx_t const nslices = reuse_rcsf->pt[tile_id].nfibs[0]; // begin from root
  #pragma omp for schedule(dynamic, 16) nowait
  for(idx_t s=0; s < nslices; ++s) {
    idx_t const fid = (fids[0] == NULL) ? s : fids[0][s];

    idxstack[0] = s; 
    for(idx_t m=1; m <= outdepth; ++m) {
      idxstack[m] = fp[m-1][idxstack[m-1]];
    }

    /* first buf will always just be a matrix row */
    val_t const * const restrict rootrow = mvals[0] + (fid * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      buf[0][f] = rootrow[f];
    }

    idx_t depth = 0;  // current depth

    // Each thread executes its own slice.
    while(idxstack[1] < fp[0][s+1]) {
      /* move down to an nnz node */
      for(; depth < outdepth; ++depth) {
        /* propogate buf down */
        val_t const * const restrict drow
            = mvals[depth+1] + (fids[depth+1][idxstack[depth+1]] * nfactors);
        p_assign_hada(buf[depth+1], buf[depth], drow, nfactors);
      }

      // after the loop, depth == outdepth now.
      /* write to output and clear buf[outdepth] for next subtree */
      idx_t const noderow = fids[outdepth][idxstack[outdepth]];
      val_t * const restrict outbuf = ovals + (noderow * nfactors);

      /* propagate value up to buf[outdepth] */
      if (outdepth < nmodes - 2) {
        p_propagate_up_reuse(buf[outdepth], buf, idxstack, outdepth, idxstack[outdepth],
            fp, fids, vals, mvals, nmodes, nfactors);

        omp_set_lock(locks + (noderow % NLOCKS));
        p_add_hada_clear(outbuf, buf[outdepth], buf[outdepth-1], nfactors);
        omp_unset_lock(locks + (noderow % NLOCKS));

      }
      else if (outdepth == nmodes -2) {
        omp_set_lock(locks + (noderow % NLOCKS));
        p_add_hada(outbuf, vals+idxstack[outdepth]*nfactors, buf[outdepth-1], nfactors);
        omp_unset_lock(locks + (noderow % NLOCKS));
      }
      else {
        printf("ERROR for outdepth.\n");
      }

    /* now move back up to the next unprocessed child */
      do {
        ++idxstack[depth];
        --depth;
      } while(depth > 0 && idxstack[depth+1] == fp[depth][idxstack[depth]+1]);
    } /* end DFS */
  } /* end outer slice loop */

}



/* determine which function to call */
static void p_root_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_mttkrp_root(tensor, 0, mats, thds);
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > 0) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          p_csf_mttkrp_root(tensor, t, mats, thds);
          #pragma omp barrier
        }
      } else {
        /* distribute tiles to threads */
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            p_csf_mttkrp_root_tiled(tensor, tid, mats, thds);
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      break;
    case SPLATT_COOPTILE:
      break;
    }
    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */
}



static void p_root_decide_genreuse_adaptive(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    rcsf_seq_adaptive * const seq_rcsfs,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    assert(tensor->which_tile == SPLATT_NOTILE);
    p_csf_mttkrp_root_genreuse_adaptive(tensor, 0, mats, seq_rcsfs, thds);
    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */
}



static void p_leaf_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = nmodes - 1;

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);

    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_mttkrp_leaf(tensor, 0, mats, thds);
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > depth) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          p_csf_mttkrp_leaf(tensor, 0, mats, thds);
        }
      } else {
        // #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            p_csf_mttkrp_leaf_tiled(tensor, tid, mats, thds);
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      break;
    case SPLATT_COOPTILE:
      break;
    }
    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */
}


static void p_intl_decide(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = csf_mode_depth(mode, tensor->dim_perm, nmodes);

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    /* tile id */
    idx_t tid = 0;
    switch(tensor->which_tile) {
    case SPLATT_NOTILE:
      p_csf_mttkrp_internal(tensor, 0, mats, mode, thds);
      break;
    case SPLATT_DENSETILE:
      /* this mode may not be tiled due to minimum tiling depth */
      if(opts[SPLATT_OPTION_TILEDEPTH] > depth) {
        for(idx_t t=0; t < tensor->ntiles; ++t) {
          p_csf_mttkrp_internal(tensor, t, mats, mode, thds);
        }
      } else {
        #pragma omp for schedule(dynamic, 1) nowait
        for(idx_t t=0; t < tensor->tile_dims[mode]; ++t) {
          tid = get_next_tileid(TILE_BEGIN, tensor->tile_dims, nmodes,
              mode, t);
          while(tid != TILE_END) {
            p_csf_mttkrp_internal_tiled(tensor, tid, mats, mode, thds);
            tid = get_next_tileid(tid, tensor->tile_dims, nmodes, mode, t);
          }
        }
      }
      break;

    /* XXX */
    case SPLATT_SYNCTILE:
      break;
    case SPLATT_COOPTILE:
      break;
    }

    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */
}



static void p_intl_decide_reuse_adaptive(
    splatt_csf const * const tensor,
    matrix_t ** mats,
    rcsf_seq_adaptive * const seq_rcsfs,
    idx_t const mode,
    thd_info * const thds,
    double const * const opts)
{
  idx_t const nmodes = tensor->nmodes;
  idx_t const depth = csf_mode_depth(mode, tensor->dim_perm, nmodes);

  #pragma omp parallel
  {
    timer_start(&thds[omp_get_thread_num()].ttime);
    assert(tensor->which_tile == SPLATT_NOTILE); 
    p_csf_mttkrp_internal_reuse_adaptive(tensor, 0, mats, seq_rcsfs, mode, thds);
    timer_stop(&thds[omp_get_thread_num()].ttime);
  } /* end omp parallel */
}





/******************************************************************************
 * PUBLIC FUNCTIONS
 *****************************************************************************/

void mttkrp_csf(
  splatt_csf const * const tensors,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  double const * const opts)
{
  /* clear output matrix */
  matrix_t * const M = mats[MAX_NMODES];
  M->I = tensors[0].dims[mode];
  memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);

  idx_t nmodes = tensors[0].nmodes;
  /* find out which level in the tree this is */
  idx_t outdepth = MAX_NMODES;

  /* choose which MTTKRP function to use */
  splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
  switch(which) {
  case SPLATT_CSF_ONEMODE:
    outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
    if(outdepth == 0) {
      p_root_decide(tensors+0, mats, mode, thds, opts);
    } else if(outdepth == nmodes - 1) {
      p_leaf_decide(tensors+0, mats, mode, thds, opts);
    } else {
      p_intl_decide(tensors+0, mats, mode, thds, opts);
    }
    break;

  case SPLATT_CSF_TWOMODE:
    /* longest mode handled via second tensor's root */
    if(mode == tensors[0].dim_perm[nmodes-1]) {
      p_root_decide(tensors+1, mats, mode, thds, opts);
    /* root and internal modes are handled via first tensor */
    } else {
      outdepth = csf_mode_depth(mode, tensors[0].dim_perm, nmodes);
      if(outdepth == 0) {
        p_root_decide(tensors+0, mats, mode, thds, opts);
      } else {
        p_intl_decide(tensors+0, mats, mode, thds, opts);
      }
    }
    break;

  case SPLATT_CSF_ALLMODE:
    p_root_decide(tensors+mode, mats, mode, thds, opts);
    break;
  }
}



void decide_use_csfs(
  idx_t const nmodes,
  group_properties * const grp_prop,
  int const n_grp,
  idx_t const n_csf,
  idx_t * use_csfs,   // The location of csf in cs.
  idx_t * use_tags) //0: root mttkrp; 1: reuse mttkrp; 2: intern or leaf mttkrp with recompute.
{
  assert(n_csf <= n_grp);
  idx_t * grp_modes = (idx_t *)splatt_malloc(n_grp * sizeof(idx_t));
  memset(use_tags, 0, nmodes * sizeof(idx_t));
  idx_t csf_idx = 0;
  idx_t mode;

  for(idx_t g=0; g<n_grp; ++g) {
    grp_modes[g] = grp_prop[g].memo_mode;
  }

  if(n_csf == n_grp) {
    for(idx_t g=0; g<n_grp-1; ++g) {
      mode = grp_modes[g];
      use_csfs[mode] = g;
      use_tags[mode] = 0;
      for(idx_t rm=mode+1; rm<grp_modes[g+1]; ++rm) {
        use_csfs[rm] = g;
        use_tags[rm] = 1;
      }
    }
    mode = grp_modes[n_grp-1];
    use_csfs[mode] = n_grp-1;
    use_tags[mode] = 0;
    for(idx_t rm=mode+1; rm<nmodes; ++rm) {
      use_csfs[rm] = n_grp-1;
      use_tags[rm] = 1;
    }
  /* n_csf != n_grp */
  } else {
    idx_t * csf_modes = (idx_t *)splatt_malloc(n_csf * sizeof(idx_t));
    for(idx_t g=0; g<n_grp; ++g) {
      if(grp_prop[g].n_imten > 0) {
        csf_modes[csf_idx] = grp_prop[g].memo_mode;
        ++ csf_idx;
      }
    }
    assert(csf_idx == n_csf);

    double mode_loc_csf;
    for(idx_t g=0; g<n_grp-1; ++g) {
      mode = grp_modes[g];
      mode_loc_csf = locate_ind_special(csf_modes, n_csf, mode, nmodes);
      if(mode_loc_csf >= 0) {
        if(mode_loc_csf - (idx_t)mode_loc_csf == 0) {
          use_csfs[mode] = (idx_t)mode_loc_csf;
          use_tags[mode] = 0;
          for(idx_t rm=mode+1; rm<grp_modes[g+1]; ++rm) {
            use_csfs[rm] = use_csfs[mode];
            use_tags[rm] = 1;
          }
        } else {
          use_csfs[mode] = (idx_t)mode_loc_csf;
          use_tags[mode] = 2;
          for(idx_t rm=mode+1; rm<grp_modes[g+1]; ++rm) {
            use_csfs[rm] = use_csfs[mode];
            use_tags[rm] = 2;
          }
        }
      }
      mode = grp_modes[n_grp-1];
      mode_loc_csf = locate_ind_special(csf_modes, n_csf, mode, nmodes);
      if(mode_loc_csf >= 0) {
        if(mode_loc_csf - (idx_t)mode_loc_csf == 0) {
          use_csfs[mode] = (idx_t)mode_loc_csf;
          use_tags[mode] = 0;
          for(idx_t rm=mode+1; rm<nmodes; ++rm) {
            use_csfs[rm] = use_csfs[mode];
            use_tags[rm] = 1;
          }
        } else {
          use_csfs[mode] = (idx_t)mode_loc_csf;
          use_tags[mode] = 2;
          for(idx_t rm=mode+1; rm<nmodes; ++rm) {
            use_csfs[rm] = use_csfs[mode];
            use_tags[rm] = 2;
          }
        }
      }
    } // Loop g

    splatt_free(csf_modes);
  }

  splatt_free(grp_modes);

} // end function



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
  double const * const opts)
{
  // printf("Function \"mttkrp_csf_adaptive\"\n");
  // fflush(stdout);

  /* clear output matrix */
  matrix_t * const M = mats[MAX_NMODES];
  M->I = tensors[0].dims[mode];
  memset(M->vals, 0, M->I * M->J * sizeof(val_t));

  omp_set_num_threads(opts[SPLATT_OPTION_NTHREADS]);

  idx_t nmodes = tensors[0].nmodes;
  /* find out which level in the tree this is */
  idx_t outdepth = MAX_NMODES;

  switch(use_tag) {
  case 0:
    if(grp_prop[use_csf].n_imten != 0)
      p_root_decide_genreuse_adaptive(tensors + use_csf, mats, rs_seq + use_csf, mode, thds, opts);
    else
      p_root_decide(tensors + use_csf, mats, mode, thds, opts);
    break;

  case 1:
    // Reuse intermediate rcsf.
    p_intl_decide_reuse_adaptive(tensors + use_csf, mats, rs_seq + use_csf, mode, thds, opts);
    break;

  case 2:
    outdepth = csf_mode_depth(mode, (tensors + use_csf)->dim_perm, nmodes);
    assert(outdepth > 0);
    if(outdepth == nmodes - 1) {
      p_leaf_decide(tensors + use_csf, mats, mode, thds, opts);
    } else {
      p_intl_decide(tensors + use_csf, mats, mode, thds, opts);
    }
    break;

  default:
    printf("Wrong use_tag.\n");
    return;
  }


}







/******************************************************************************
 * DEPRECATED FUNCTIONS
 *****************************************************************************/








/******************************************************************************
 * SPLATT MTTKRP
 *****************************************************************************/

void mttkrp_splatt(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  if(ft->tiled == SPLATT_SYNCTILE) {
    mttkrp_splatt_sync_tiled(ft, mats, mode, thds, nthreads);
    return;
  }
  if(ft->tiled == SPLATT_COOPTILE) {
    mttkrp_splatt_coop_tiled(ft, mats, mode, thds, nthreads);
    return;
  }

  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];
  idx_t const nslices = ft->dims[mode];
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict sptr = ft->sptr;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  val_t const * const restrict vals = ft->vals;

  sp_timer_t mul_timer, hr_timer;
  double mul_sec, hr_sec;
  timer_reset (&mul_timer);
  timer_reset (&hr_timer);

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 16) nowait
    for(idx_t s=0; s < nslices; ++s) {
      val_t * const restrict mv = mvals + (s * rank);

      /* foreach fiber in slice */
      for(idx_t f=sptr[s]; f < sptr[s+1]; ++f) {
        timer_start(&mul_timer);
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }
        timer_stop(&mul_timer);

        timer_start(&hr_timer);
        /* scale inner products by row of A and update to M */
        val_t const * const restrict av = avals  + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
        timer_stop(&hr_timer);
      }
    }

    mul_sec = mul_timer.seconds;
    hr_sec = hr_timer.seconds;
    printf("mul_sec: %f\n", mul_sec);
    printf("hr_sec: %f\n", hr_sec);

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_sync_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];

  idx_t const nslabs = ft->nslabs;
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr;
  idx_t const * const restrict sids = ft->sids;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    timer_start(&thds[tid].ttime);

    #pragma omp for schedule(dynamic, 1) nowait
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slice */
      for(idx_t f=slabptr[s]; f < slabptr[s+1]; ++f) {
        /* first entry of the fiber is used to initialize accumF */
        idx_t const jjfirst  = fptr[f];
        val_t const vfirst   = vals[jjfirst];
        val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
        for(idx_t r=0; r < rank; ++r) {
          accumF[r] = vfirst * bv[r];
        }

        /* foreach nnz in fiber */
        for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
          val_t const v = vals[jj];
          val_t const * const restrict bv = bvals + (inds[jj] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] += v * bv[r];
          }
        }

        /* scale inner products by row of A and update to M */
        val_t       * const restrict mv = mvals + (sids[f] * rank);
        val_t const * const restrict av = avals + (fids[f] * rank);
        for(idx_t r=0; r < rank; ++r) {
          mv[r] += accumF[r] * av[r];
        }
      }
    }

    timer_stop(&thds[tid].ttime);
  } /* end parallel region */
}


void mttkrp_splatt_coop_tiled(
  ftensor_t const * const ft,
  matrix_t ** mats,
  idx_t const mode,
  thd_info * const thds,
  idx_t const nthreads)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mats[ft->dim_perm[1]];
  matrix_t const * const B = mats[ft->dim_perm[2]];

  idx_t const nslabs = ft->nslabs;
  idx_t const rank = M->J;

  val_t * const mvals = M->vals;
  memset(mvals, 0, ft->dims[mode] * rank * sizeof(val_t));

  val_t const * const avals = A->vals;
  val_t const * const bvals = B->vals;

  idx_t const * const restrict slabptr = ft->slabptr;
  idx_t const * const restrict sptr = ft->sptr;
  idx_t const * const restrict sids = ft->sids;
  idx_t const * const restrict fptr = ft->fptr;
  idx_t const * const restrict fids = ft->fids;
  idx_t const * const restrict inds = ft->inds;
  val_t const * const restrict vals = ft->vals;

  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
    val_t * const localm = (val_t *) thds[tid].scratch[1];
    timer_start(&thds[tid].ttime);

    /* foreach slab */
    for(idx_t s=0; s < nslabs; ++s) {
      /* foreach fiber in slab */
      #pragma omp for schedule(dynamic, 8)
      for(idx_t sl=slabptr[s]; sl < slabptr[s+1]; ++sl) {
        idx_t const slice = sids[sl];
        for(idx_t f=sptr[sl]; f < sptr[sl+1]; ++f) {
          /* first entry of the fiber is used to initialize accumF */
          idx_t const jjfirst  = fptr[f];
          val_t const vfirst   = vals[jjfirst];
          val_t const * const restrict bv = bvals + (inds[jjfirst] * rank);
          for(idx_t r=0; r < rank; ++r) {
            accumF[r] = vfirst * bv[r];
          }

          /* foreach nnz in fiber */
          for(idx_t jj=fptr[f]+1; jj < fptr[f+1]; ++jj) {
            val_t const v = vals[jj];
            val_t const * const restrict bv = bvals + (inds[jj] * rank);
            for(idx_t r=0; r < rank; ++r) {
              accumF[r] += v * bv[r];
            }
          }

          /* scale inner products by row of A and update thread-local M */
          val_t       * const restrict mv = localm + ((slice % TILE_SIZES[0]) * rank);
          val_t const * const restrict av = avals + (fids[f] * rank);
          for(idx_t r=0; r < rank; ++r) {
            mv[r] += accumF[r] * av[r];
          }
        }
      }

      idx_t const start = s * TILE_SIZES[0];
      idx_t const stop  = SS_MIN((s+1) * TILE_SIZES[0], ft->dims[mode]);

      #pragma omp for schedule(static)
      for(idx_t i=start; i < stop; ++i) {
        /* map i back to global slice id */
        idx_t const localrow = i % TILE_SIZES[0];
        for(idx_t t=0; t < nthreads; ++t) {
          val_t * const threadm = (val_t *) thds[t].scratch[1];
          for(idx_t r=0; r < rank; ++r) {
            mvals[r + (i*rank)] += threadm[r + (localrow*rank)];
            threadm[r + (localrow*rank)] = 0.;
          }
        }
      }

    } /* end foreach slab */
    timer_stop(&thds[tid].ttime);
  } /* end omp parallel */
}



/******************************************************************************
 * GIGA MTTKRP
 *****************************************************************************/
void mttkrp_giga(
  spmatrix_t const * const spmat,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = spmat->I;
  idx_t const rank = M->J;

  idx_t const * const restrict rowptr = spmat->rowptr;
  idx_t const * const restrict colind = spmat->colind;
  val_t const * const restrict vals   = spmat->vals;

  #pragma omp parallel
  {
    for(idx_t r=0; r < rank; ++r) {
      val_t       * const restrict mv =  M->vals + (r * I);
      val_t const * const restrict av =  A->vals + (r * A->I);
      val_t const * const restrict bv =  B->vals + (r * B->I);

      /* Joined Hadamard products of X, C, and B */
      #pragma omp for schedule(dynamic, 16)
      for(idx_t i=0; i < I; ++i) {
        for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
          idx_t const a = colind[y] / B->I;
          idx_t const b = colind[y] % B->I;
          scratch[y] = vals[y] * av[a] * bv[b];
        }
      }

      /* now accumulate rows into column of M1 */
      #pragma omp for schedule(dynamic, 16)
      for(idx_t i=0; i < I; ++i) {
        val_t sum = 0;
        for(idx_t y=rowptr[i]; y < rowptr[i+1]; ++y) {
          sum += scratch[y];
        }
        mv[i] = sum;
      }
    }
  }
}


/******************************************************************************
 * TTBOX MTTKRP
 *****************************************************************************/
#if 0
void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  sp_timer_t ttbox_timer, setup_timer, memset_timer, comp_timer, scratch_timer, mv_timer;
  double ttbox_sec, setup_sec, memset_sec, comp_sec, scratch_sec, mv_sec;
  // timer_fstart (&ttbox_timer);

  // timer_fstart (&setup_timer);
  matrix_t       * const M = mats[MAX_NMODES];
  matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
  matrix_t const * const B = mode == 2 ? mats[1] : mats[2];

  idx_t const I = tt->dims[mode];
  idx_t const rank = M->J;

  memset(M->vals, 0, I * rank * sizeof(val_t));

  idx_t const nnz = tt->nnz;
  idx_t const * const restrict indM = tt->ind[mode];
  idx_t const * const restrict indA =
    mode == 0 ? tt->ind[1] : tt->ind[0];
  idx_t const * const restrict indB =
    mode == 2 ? tt->ind[1] : tt->ind[2];

  val_t const * const restrict vals = tt->vals;
  // timer_stop (&setup_timer);
  // setup_sec = setup_timer.seconds;

  // timer_fstart (&comp_timer);
  // timer_reset (&scratch_timer);
  // timer_reset (&mv_timer);
  for(idx_t r=0; r < rank; ++r) {
    val_t       * const restrict mv =  M->vals + (r * I);
    val_t const * const restrict av =  A->vals + (r * A->I);
    val_t const * const restrict bv =  B->vals + (r * B->I);

    /* stretch out columns of A and B */
    // timer_start (&scratch_timer);
    #pragma omp parallel for
    for(idx_t x=0; x < nnz; ++x) {
      scratch[x] = vals[x] * av[indA[x]] * bv[indB[x]];
    }
    // timer_stop (&scratch_timer);
    // scratch_sec = scratch_timer.seconds;

    /* now accumulate into m1 */
    // timer_start (&mv_timer);
    for(idx_t x=0; x < nnz; ++x) {
      mv[indM[x]] += scratch[x];
    }
    // timer_stop (&mv_timer);
    // mv_sec = mv_timer.seconds;
  }
  // timer_stop (&comp_timer);
  // comp_sec = comp_timer.seconds;

  // timer_stop (&ttbox_timer);
  // ttbox_sec = ttbox_timer.seconds;
  // printf("ttbox_sec: %f\n", ttbox_sec);
  // printf("\tsetup_sec: %f\n", setup_sec);
  // printf("\tcomp_sec: %f\n", comp_sec);
  // printf("\t\tscratch_sec: %f\n", scratch_sec);
  // printf("\t\tmv_sec: %f\n", mv_sec);
}
#endif




#if 1
// jli: extend ttbox to high-order tensors.
void mttkrp_ttbox(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode,
  val_t * const scratch)
{
  sp_timer_t ttbox_timer, setup_timer, memset_timer, comp_timer, scratch_timer, mv_timer;
  double ttbox_sec, setup_sec, memset_sec, comp_sec, scratch_sec, mv_sec;
  // timer_fstart (&ttbox_timer);

  // timer_fstart (&setup_timer);
  idx_t const nnz = tt->nnz;
  idx_t ** inds = tt->ind;
  val_t const * const restrict vals = tt->vals;

  matrix_t * const M = mats[MAX_NMODES];
  idx_t const I = tt->dims[mode];
  idx_t const rank = M->J;

  memset(M->vals, 0, I * rank * sizeof(val_t));

  // #pragma omp parallel
  // {
  //   if (omp_get_thread_num() == 0)
  //     printf("omp num_threads: %d\n", omp_get_num_threads());
  // }

  if (tt->type == SPLATT_3MODE)
  {
    /*** For 3rd-order tensor **/
    assert (tt->nmodes == 3);

    //mats is in reserve order.
    matrix_t const * const A = mode == 0 ? mats[1] : mats[0];
    matrix_t const * const B = mode == 2 ? mats[1] : mats[2];
    assert (rank == B->J);

    idx_t const * const indM = tt->ind[mode];
    idx_t const * const indA =
    mode == 0 ? tt->ind[1] : tt->ind[0];
    idx_t const * const indB =
    mode == 2 ? tt->ind[1] : tt->ind[2];
    // timer_stop (&setup_timer);
    // setup_sec = setup_timer.seconds;

    // timer_fstart (&comp_timer);
    // timer_reset (&scratch_timer);
    // timer_reset (&mv_timer);
    for(idx_t r=0; r < rank; ++r) {
      val_t       * const restrict mv =  M->vals + (r * I);
      val_t const * const restrict av =  A->vals + (r * A->I);
      val_t const * const restrict bv =  B->vals + (r * B->I);

      /* stretch out columns of A and B */
      // timer_start (&scratch_timer);
      #pragma omp parallel for
      for(idx_t x=0; x < nnz; ++x) {
        scratch[x] = vals[x] * av[indA[x]] * bv[indB[x]];
      }
      // timer_stop (&scratch_timer);
      // scratch_sec = scratch_timer.seconds;

      /* now accumulate into m1 */
      // timer_start (&mv_timer);
      for(idx_t x=0; x < nnz; ++x) {
        mv[indM[x]] += scratch[x];
      }
      // timer_stop (&mv_timer);
      // mv_sec = mv_timer.seconds;
    }
    // timer_stop (&comp_timer);
    // comp_sec = comp_timer.seconds;
  }
  else if (tt->type == SPLATT_NMODE)
  {
    assert (tt->nmodes > 3);

    val_t * scratch_2 = (val_t*)splatt_malloc(nnz * sizeof(val_t));
    memset(scratch_2, 0, nnz * sizeof(val_t));

    idx_t nmodes = tt->nmodes;
    idx_t nmats = nmodes - 1;
    idx_t * const mats_order = (idx_t *) splatt_malloc (nmats * sizeof(*mats_order));
    idx_t j = 0;
    for (int i=nmodes-1; i>=0; --i) {
      if (i != mode) {
        mats_order[j] = i;
        ++ j;
      }
    }
    assert (j == nmats-1);

    idx_t const * const indM = tt->ind[mode];

    for(idx_t r=0; r < rank; ++r) 
    {
      for (idx_t ii=0; ii<nnz; ++ii)
        scratch[ii] = tt->vals[ii];
      // memcpy (scratch, tt->vals, nnz);
      // val_t       * const mv =  update_mat->vals_ + (r * update_mat->stride_);
      val_t       * const restrict mv =  M->vals + (r * I);
      for (idx_t i=0; i<nmats; i++)
      {
        // set_values<val_t> (scratch, scratch_2, nnz);
        matrix_t *tmp_mat = mats[mats_order[i]];
        assert (rank == tmp_mat->J);
        idx_t *tmp_inds = tt->ind[mats_order[i]];
        val_t const * const av =  tmp_mat->vals + (r * tmp_mat->I);
        #pragma omp parallel for
        for(idx_t x=0; x < nnz; ++x) {
          scratch_2[x] = scratch[x] * av[tmp_inds[x]];
        }
        // memcpy (scratch, scratch_2, nnz);
        for (idx_t ii=0; ii<nnz; ++ii)
          scratch[ii] = scratch_2[ii];
      }
      for(idx_t x=0; x < nnz; ++x) {
        mv[indM[x]] += scratch[x];
      }
    }

    free(mats_order);
    free(scratch_2);
  }
  // print_mat (M);

  // timer_stop (&ttbox_timer);
  // ttbox_sec = ttbox_timer.seconds;

  // printf("ttbox_sec: %f\n", ttbox_sec);
  // printf("\tsetup_sec: %f\n", setup_sec);
  // printf("\tcomp_sec: %f\n", comp_sec);
  // printf("\t\tscratch_sec: %f\n", scratch_sec);
  // printf("\t\tmv_sec: %f\n", mv_sec);

  return;
}
#endif



void mttkrp_stream(
  sptensor_t const * const tt,
  matrix_t ** mats,
  idx_t const mode)
{
  matrix_t * const M = mats[MAX_NMODES];
  idx_t const I = tt->dims[mode];
  idx_t const nfactors = M->J;

  val_t * const outmat = M->vals;
  memset(outmat, 0, I * nfactors * sizeof(val_t));

  idx_t const nmodes = tt->nmodes;

  val_t * accum = (val_t *) splatt_malloc(nfactors * sizeof(val_t));

  val_t * mvals[MAX_NMODES];
  for(idx_t m=0; m < nmodes; ++m) {
    mvals[m] = mats[m]->vals;
  }

  val_t const * const restrict vals = tt->vals;

  /* stream through nnz */
  for(idx_t n=0; n < tt->nnz; ++n) {
    /* initialize with value */
    for(idx_t f=0; f < nfactors; ++f) {
      accum[f] = vals[n];
    }

    for(idx_t m=0; m < nmodes; ++m) {
      if(m == mode) {
        continue;
      }
      val_t const * const restrict inrow = mvals[m] + (tt->ind[m][n] * nfactors);
      for(idx_t f=0; f < nfactors; ++f) {
        accum[f] *= inrow[f];
      }
    }

    /* write to output */
    val_t * const restrict outrow = outmat + (tt->ind[mode][n] * nfactors);
    for(idx_t f=0; f < nfactors; ++f) {
      outrow[f] += accum[f];
    }
  }

  free(accum);
}


