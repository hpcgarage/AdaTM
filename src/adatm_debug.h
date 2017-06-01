#ifndef ADATM_DEBUG_H
#define ADATM_DEUBG_H

void iprint_array (
		idx_t const * const array,
		idx_t const len,
		char const * const name);

void print_intarray (
		int const * const array,
		idx_t const len,
		char const * const name);

void dprint_array (
		val_t const * const array,
		idx_t const len,
		char const * const name);


void print_pair_array (
  pair * const pair_list, 
  idx_t const size, 
  char * const name);


void print_group_properties (
  group_properties * const grp_prop, 
  idx_t const size, 
  idx_t const nmodes, 
  char * const name);

void print_configs (
  configurations_adaptive * const configs, 
  idx_t const begin, 
  idx_t const end,
  idx_t const nmodes, 
  char * const name);

#endif