/******************************************************************************
 * INCLUDES
 *****************************************************************************/
#include "sort.h"
#include "timer.h"
#include "adatm_sort.h"



void pair_sort_range(
  pair * const pair_list,
  idx_t const start,
  idx_t const end)
{
  pair pmid;

  size_t i = start+1;
  size_t j = end-1;
  size_t k = start + ((end - start) / 2);

  /* grab pivot */
  pmid.x = pair_list[k].x;
  pair_list[k].x = pair_list[start].x;
  pmid.y = pair_list[k].y;
  pair_list[k].y = pair_list[start].y;

    while(i < j) {
      /* if tt[i] > mid  -> tt[i] is on wrong side */
      if(pmid.y < pair_list[i].y) {
        /* if tt[j] <= mid  -> swap tt[i] and tt[j] */
        if(pmid.y > pair_list[j].y) {
          pair ptmp;
          ptmp.x = pair_list[i].x;
          pair_list[i].x = pair_list[j].x;
          pair_list[j].x = ptmp.x;
          ptmp.y = pair_list[i].y;
          pair_list[i].y = pair_list[j].y;
          pair_list[j].y = ptmp.y;
          ++i;
        }
        --j;
      } else {
        /* if tt[j] > mid  -> tt[j] is on right side */
        if(pmid.y < pair_list[j].y) {
          --j;
        }
        ++i;
      }
    } // End while

    /* if tt[i] > mid */
    if(pmid.y < pair_list[i].y) {
      --i;
    }
    pair_list[start].x = pair_list[i].x;
    pair_list[i].x = pmid.x;
    pair_list[start].y = pair_list[i].y;
    pair_list[i].y = pmid.y;

    if(i > start + 1) {
      pair_sort_range(pair_list, start, i);
    }
    ++i; /* skip the pivot element */
    if(end - i > 1) {
      pair_sort_range(pair_list, i, end);
    }
}


void pair_sort(
  pair * const pair_list,
  idx_t const size)
{
  if(size > 1)
    pair_sort_range(pair_list, 0, size);
}