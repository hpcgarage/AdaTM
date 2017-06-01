
#include "csf.h"
#include "adatm_debug.h"

void iprint_array (
		idx_t const * const array,
		idx_t const len,
		char const * const name)
{
	printf("%s: \n", name);
	for (idx_t i=0; i<len; ++i)
		printf(" %"SPLATT_PF_IDX, array[i]);
	printf("\n");
	fflush(stdout);
}


void print_intarray (
		int const * const array,
		idx_t const len,
		char const * const name)
{
	printf("%s: \n", name);
	for (idx_t i=0; i<len; ++i)
		printf(" %d", array[i]);
	printf("\n");
	fflush(stdout);
}



void dprint_array (
		val_t const * const array,
		idx_t const len,
		char const * const name)
{
	printf("%s: \n", name);
	for (idx_t i=0; i<len; ++i)
		printf(" %f ", array[i]);
	printf("\n");
	fflush(stdout);
}


void print_pair_array (
  pair * const pair_list, 
  idx_t const size, 
  char * const name) 
{
  printf("====== %s =======\n", name);
  for(idx_t i=0; i<size; ++i) {
    printf("[%lu %lu], ", pair_list[i].x, pair_list[i].y);
  }
  printf("\n");
}


void print_group_properties (
  group_properties * const grp_prop, 
  idx_t const size, 
  idx_t const nmodes, 
  char * const name) 
{
  printf("====== %s =======\n", name);
  printf("[memo_mode, begin_imten, n_imten, o_opt]\n");
  for(idx_t i=0; i<size; ++i) {
    printf("%lu, %lu, %lu, \n", grp_prop[i].memo_mode, grp_prop[i].begin_imten, grp_prop[i].n_imten);
    iprint_array(grp_prop[i].o_opt, nmodes, "o_opt");
  }
  printf("\n");
  fflush(stdout);
}

void print_configs (
  configurations_adaptive * const configs, 
  idx_t const begin, 
  idx_t const end,
  idx_t const nmodes, 
  char * const name) 
{
  printf("\n====== %s =======\n", name);
  for(idx_t c=begin; c<end; ++c) {
    printf("\nConfigure %lu\n", c);
    print_group_properties(configs[c].grp_prop, configs[c].n_grp, nmodes, "grp_prop");
    printf("pspace: %lu\npops: %lu\n", configs[c].pspace, configs[c].pops);
  }
  printf("\n");
  fflush(stdout);
}