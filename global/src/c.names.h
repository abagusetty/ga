#ifndef C_NAMES_H
#define C_NAMES_H

#include "globalp.h"

#define ga_abs_value_                      F77_FUNC_(ga_abs_value, GA_ABS_VALUE)
#define ga_abs_value_patch_                F77_FUNC_(ga_abs_value_patch, GA_ABS_VALUE_PATCH_)
#define ga_acc_                            F77_FUNC_(ga_acc, GA_ACC)
#define ga_access_                         F77_FUNC_(ga_access, GA_ACCESS)
#define ga_add_                            F77_FUNC_(ga_add, GA_ADD)
#define ga_add_constant_                   F77_FUNC_(ga_add_constant, GA_ADD_CONSTANT)
#define ga_add_constant_patch_             F77_FUNC_(ga_add_constant_patch, GA_ADD_CONSTANT_PATCH)
#define ga_add_diagonal_                   F77_FUNC_(ga_add_diagonal, GA_ADD_DIAGONAL)
#define ga_add_patch_                      F77_FUNC_(ga_add_patch, GA_ADD_PATCH)
#define ga_allocate_                       F77_FUNC_(ga_allocate, GA_ALLOCATE)
#define ga_bin_index_                      F77_FUNC_(ga_bin_index, GA_BIN_INDEX)
#define ga_bin_sorter_                     F77_FUNC_(ga_bin_sorter, GA_BIN_SORTER)
#define ga_brdcst_                         F77_FUNC_(ga_brdcst, GA_BRDCST)
#define ga_cadd_                           F77_FUNC_(ga_cadd, GA_CADD)
#define ga_cadd_patch_                     F77_FUNC_(ga_cadd_patch, GA_CADD_PATCH)
#define ga_cfill_                          F77_FUNC_(ga_cfill, GA_CFILL)
#define ga_cfill_patch_                    F77_FUNC_(ga_cfill_patch, GA_CFILL_PATCH)
#define ga_cgemm_                          F77_FUNC_(ga_cgemm, GA_CGEMM)
#define ga_cgop_                           F77_FUNC_(ga_cgop, GA_CGOP)
#define ga_check_handle_                   F77_FUNC_(ga_check_handle, GA_CHECK_HANDLE)
#define ga_cluster_nnodes_                 F77_FUNC_(ga_cluster_nnodes, GA_CLUSTER_NNODES)
#define ga_cluster_nodeid_                 F77_FUNC_(ga_cluster_nodeid, GA_CLUSTER_NODEID)
#define ga_cluster_nprocs_                 F77_FUNC_(ga_cluster_nprocs, GA_CLUSTER_NPROCS)
#define ga_cluster_proc_nodeid_            F77_FUNC_(ga_cluster_proc_nodeid, GA_CLUSTER_PROC_NODEID)
#define ga_cluster_procid_                 F77_FUNC_(ga_cluster_procid, GA_CLUSTER_PROCID)
#define ga_compare_distr_                  F77_FUNC_(ga_compare_distr, GA_COMPARE_DISTR)
#define ga_copy_                           F77_FUNC_(ga_copy, GA_COPY)
#define ga_copy_patch_                     F77_FUNC_(ga_copy_patch, GA_COPY_PATCH)
#define ga_create_                         F77_FUNC_(ga_create, GA_CREATE)
#define ga_create_bin_range_               F77_FUNC_(ga_create_bin_range, GA_CREATE_BIN_RANGE)
#define ga_create_handle_                  F77_FUNC_(ga_create_handle, GA_CREATE_HANDLE)
#define ga_create_irreg_                   F77_FUNC_(ga_create_irreg, GA_CREATE_IRREG)
#define ga_create_mutexes_                 F77_FUNC_(ga_create_mutexes, GA_CREATE_MUTEXES)
#define ga_cscal_                          F77_FUNC_(ga_cscal, GA_CSCAL)
#define ga_cscal_patch_                    F77_FUNC_(ga_cscal_patch, GA_CSCAL_PATCH)
#define ga_dadd_                           F77_FUNC_(ga_dadd, GA_DADD)
#define ga_dadd_patch_                     F77_FUNC_(ga_dadd_patch, GA_DADD_PATCH)
#define ga_ddot_                           F77_FUNC_(ga_ddot, GA_DDOT)
#define ga_ddot_patch_                     F77_FUNC_(ga_ddot_patch, GA_DDOT_PATCH)
#define ga_destroy_                        F77_FUNC_(ga_destroy, GA_DESTROY)
#define ga_destroy_mutexes_                F77_FUNC_(ga_destroy_mutexes, GA_DESTROY_MUTEXES)
#define ga_dfill_                          F77_FUNC_(ga_dfill, GA_DFILL)
#define ga_dfill_patch_                    F77_FUNC_(ga_dfill_patch, GA_DFILL_PATCH)
#define ga_dgemm_                          F77_FUNC_(ga_dgemm, GA_DGEMM)
#define ga_dgop_                           F77_FUNC_(ga_dgop, GA_DGOP)
#define ga_diag_                           F77_FUNC_(ga_diag, GA_DIAG)
#define ga_diag_reuse_                     F77_FUNC_(ga_diag_reuse, GA_DIAG_REUSE)
#define ga_diag_seq_                       F77_FUNC_(ga_diag_seq, GA_DIAG_SEQ)
#define ga_diag_std_                       F77_FUNC_(ga_diag_std, GA_DIAG_STD)
#define ga_diag_std_seq_                   F77_FUNC_(ga_diag_std_seq, GA_DIAG_STD_SEQ)
#define ga_distribution_                   F77_FUNC_(ga_distribution, GA_DISTRIBUTION)
#define ga_dscal_                          F77_FUNC_(ga_dscal, GA_DSCAL)
#define ga_dscal_patch_                    F77_FUNC_(ga_dscal_patch, GA_DSCAL_PATCH)
#define ga_duplicate_                      F77_FUNC_(ga_duplicate, GA_DUPLICATE)
#define ga_elem_divide_                    F77_FUNC_(ga_elem_divide, GA_ELEM_DIVIDE)
#define ga_elem_divide_patch_              F77_FUNC_(ga_elem_divide_patch, GA_ELEM_DIVIDE_PATCH)
#define ga_elem_maximum_                   F77_FUNC_(ga_elem_maximum, GA_ELEM_MAXIMUM)
#define ga_elem_maximum_patch_             F77_FUNC_(ga_elem_maximum_patch, GA_ELEM_MAXIMUM_PATCH)
#define ga_elem_minimum_                   F77_FUNC_(ga_elem_minimum, GA_ELEM_MINIMUM)
#define ga_elem_minimum_patch_             F77_FUNC_(ga_elem_minimum_patch, GA_ELEM_MINIMUM_PATCH)
#define ga_elem_multiply_                  F77_FUNC_(ga_elem_multiply, GA_ELEM_MULTIPLY)
#define ga_elem_multiply_patch_            F77_FUNC_(ga_elem_multiply_patch, GA_ELEM_MULTIPLY_PATCH)
#define ga_error_                          F77_FUNC_(ga_error, GA_ERROR)
#define ga_fast_merge_mirrored_            F77_FUNC_(ga_fast_merge_mirrored, GA_FAST_MERGE_MIRRORED)
#define ga_fence_                          F77_FUNC_(ga_fence, GA_FENCE)
#define ga_fill_                           F77_FUNC_(ga_fill, GA_FILL)
#define ga_randomize_                      F77_FUNC_(ga_randomize, GA_RANDOMIZE)
#define ga_fill_patch_                     F77_FUNC_(ga_fill_patch, GA_FILL_PATCH)
#define ga_gather_                         F77_FUNC_(ga_gather, GA_GATHER)
#define ga_get_                            F77_FUNC_(ga_get, GA_GET)
#define ga_get_block_info_                 F77_FUNC_(ga_get_block_info, GA_GET_BLOCK_INFO)
#define ga_get_debug_                      F77_FUNC_(ga_get_debug, GA_GET_DEBUG)
#define ga_get_diag_                       F77_FUNC_(ga_get_diag, GA_GET_DIAG)
#define ga_get_dimension_                  F77_FUNC_(ga_get_dimension, GA_GET_DIMENSION)
#define ga_get_mirrored_block_             F77_FUNC_(ga_get_mirrored_block, GA_GET_MIRRORED_BLOCK)
#define ga_get_pgroup_                     F77_FUNC_(ga_get_pgroup, GA_GET_PGROUP)
#define ga_get_proc_grid_                  F77_FUNC_(ga_get_proc_grid, GA_GET_PROC_GRID)
#define ga_get_proc_index_                 F77_FUNC_(ga_get_proc_index, GA_GET_PROC_INDEX)
#define ga_ghost_barrier_                  F77_FUNC_(ga_ghost_barrier, GA_GHOST_BARRIER)
#define ga_gop_                            F77_FUNC_(ga_gop, GA_GOP)
#define ga_has_ghosts_                     F77_FUNC_(ga_has_ghosts, GA_HAS_GHOSTS)
#define ga_iadd_                           F77_FUNC_(ga_iadd, GA_IADD)
#define ga_iadd_patch_                     F77_FUNC_(ga_iadd_patch, GA_IADD_PATCH)
#define ga_idot_                           F77_FUNC_(ga_idot, GA_IDOT)
#define ga_idot_patch_                     F77_FUNC_(ga_idot_patch, GA_IDOT_PATCH)
#define ga_ifill_                          F77_FUNC_(ga_ifill, GA_IFILL)
#define ga_ifill_patch_                    F77_FUNC_(ga_ifill_patch, GA_IFILL_PATCH)
#define ga_igop_                           F77_FUNC_(ga_igop, GA_IGOP)
#define ga_init_fence_                     F77_FUNC_(ga_init_fence, GA_INIT_FENCE)
#define ga_initialize_                     F77_FUNC_(ga_initialize, GA_INITIALIZE)
#define ga_initialize_ltd_                 F77_FUNC_(ga_initialize_ltd, GA_INITIALIZE_LTD)
#define ga_inquire_                        F77_FUNC_(ga_inquire, GA_INQUIRE)
#define ga_inquire_internal_               F77_FUNC_(ga_inquire_internal, GA_INQUIRE_INTERNAL)
#define ga_inquire_memory_                 F77_FUNC_(ga_inquire_memory, GA_INQUIRE_MEMORY)
#define ga_inquire_name_                   F77_FUNC_(ga_inquire_name, GA_INQUIRE_NAME)
#define ga_is_mirrored_                    F77_FUNC_(ga_is_mirrored, GA_IS_MIRRORED)
#define ga_iscal_                          F77_FUNC_(ga_iscal, GA_ISCAL)
#define ga_iscal_patch_                    F77_FUNC_(ga_iscal_patch, GA_ISCAL_PATCH)
#define ga_list_data_servers_              F77_FUNC_(ga_list_data_servers, GA_LIST_DATA_SERVERS)
#define ga_list_nodeid_                    F77_FUNC_(ga_list_nodeid, GA_LIST_NODEID)
#define ga_llt_solve_                      F77_FUNC_(ga_llt_solve, GA_LLT_SOLVE)
#define ga_locate_                         F77_FUNC_(ga_locate, GA_LOCATE)
#define ga_locate_region_                  F77_FUNC_(ga_locate_region, GA_LOCATE_REGION)
#define ga_lock_                           F77_FUNC_(ga_lock, GA_LOCK)
#define ga_lu_solve_                       F77_FUNC_(ga_lu_solve, GA_LU_SOLVE)
#define ga_lu_solve_alt_                   F77_FUNC_(ga_lu_solve_alt, GA_LU_SOLVE_ALT)
#define ga_lu_solve_seq_                   F77_FUNC_(ga_lu_solve_seq, GA_LU_SOLVE_SEQ)
#define ga_mask_sync_                      F77_FUNC_(ga_mask_sync, GA_MASK_SYNC)
#define ga_matmul_patch_                   F77_FUNC_(ga_matmul_patch, GA_MATMUL_PATCH)
#define ga_median_                         F77_FUNC_(ga_median, GA_MEDIAN)
#define ga_median_patch_                   F77_FUNC_(ga_median_patch, GA_MEDIAN_PATCH)
#define ga_memory_avail_                   F77_FUNC_(ga_memory_avail, GA_MEMORY_AVAIL)
#define ga_memory_limited_                 F77_FUNC_(ga_memory_limited, GA_MEMORY_LIMITED)
#define ga_merge_mirrored_                 F77_FUNC_(ga_merge_mirrored, GA_MERGE_MIRRORED)
#define ga_nbacc_                          F77_FUNC_(ga_nbacc, GA_NBACC)
#define ga_nbget_                          F77_FUNC_(ga_nbget, GA_NBGET)
#define ga_nbput_                          F77_FUNC_(ga_nbput, GA_NBPUT)
#define ga_nbtest_                         F77_FUNC_(ga_nbtest, GA_NBTEST)
#define ga_nbwait_                         F77_FUNC_(ga_nbwait, GA_NBWAIT)
#define ga_ndim_                           F77_FUNC_(ga_ndim, GA_NDIM)
#define ga_nnodes_                         F77_FUNC_(ga_nnodes, GA_NNODES)
#define ga_nodeid_                         F77_FUNC_(ga_nodeid, GA_NODEID)
#define ga_norm1_                          F77_FUNC_(ga_norm1, GA_NORM1)
#define ga_norm_infinity_                  F77_FUNC_(ga_norm_infinity, GA_NORM_INFINITY)
#define ga_num_data_servers_               F77_FUNC_(ga_num_data_servers, GA_NUM_DATA_SERVERS)
#define ga_num_mirrored_seg_               F77_FUNC_(ga_num_mirrored_seg, GA_NUM_MIRRORED_SEG)
#define ga_pack_                           F77_FUNC_(ga_pack, GA_PACK)
#define ga_patch_enum_                     F77_FUNC_(ga_patch_enum, GA_PATCH_ENUM)
#define ga_pgroup_brdcst_                  F77_FUNC_(ga_pgroup_brdcst, GA_PGROUP_BRDCST)
#define ga_pgroup_cgop_                    F77_FUNC_(ga_pgroup_cgop, GA_PGROUP_CGOP)
#define ga_pgroup_create_                  F77_FUNC_(ga_pgroup_create, GA_PGROUP_CREATE)
#define ga_pgroup_destroy_                 F77_FUNC_(ga_pgroup_destroy, GA_PGROUP_DESTROY)
#define ga_pgroup_dgop_                    F77_FUNC_(ga_pgroup_dgop, GA_PGROUP_DGOP)
#define ga_pgroup_get_default_             F77_FUNC_(ga_pgroup_get_default, GA_PGROUP_GET_DEFAULT)
#define ga_pgroup_get_mirror_              F77_FUNC_(ga_pgroup_get_mirror, GA_PGROUP_GET_MIRROR)
#define ga_pgroup_get_world_               F77_FUNC_(ga_pgroup_get_world, GA_PGROUP_GET_WORLD)
#define ga_pgroup_igop_                    F77_FUNC_(ga_pgroup_igop, GA_PGROUP_IGOP)
#define ga_pgroup_nnodes_                  F77_FUNC_(ga_pgroup_nnodes, GA_PGROUP_NNODES)
#define ga_pgroup_nodeid_                  F77_FUNC_(ga_pgroup_nodeid, GA_PGROUP_NODEID)
#define ga_pgroup_set_default_             F77_FUNC_(ga_pgroup_set_default, GA_PGROUP_SET_DEFAULT)
#define ga_pgroup_sgop_                    F77_FUNC_(ga_pgroup_sgop, GA_PGROUP_SGOP)
#define ga_pgroup_split_                   F77_FUNC_(ga_pgroup_split, GA_PGROUP_SPLIT)
#define ga_pgroup_split_irreg_             F77_FUNC_(ga_pgroup_split_irreg, GA_PGROUP_SPLIT_IRREG)
#define ga_pgroup_sync_                    F77_FUNC_(ga_pgroup_sync, GA_PGROUP_SYNC)
#define ga_pgroup_zgop_                    F77_FUNC_(ga_pgroup_zgop, GA_PGROUP_ZGOP)
#define ga_print_                          F77_FUNC_(ga_print, GA_PRINT)
#define ga_print_distribution_             F77_FUNC_(ga_print_distribution, GA_PRINT_DISTRIBUTION)
#define ga_print_patch_                    F77_FUNC_(ga_print_patch, GA_PRINT_PATCH)
#define ga_print_stats_                    F77_FUNC_(ga_print_stats, GA_PRINT_STATS)
#define ga_proc_topology_                  F77_FUNC_(ga_proc_topology, GA_PROC_TOPOLOGY)
#define ga_put_                            F77_FUNC_(ga_put, GA_PUT)
#define ga_read_inc_                       F77_FUNC_(ga_read_inc, GA_READ_INC)
#define ga_recip_                          F77_FUNC_(ga_recip, GA_RECIP)
#define ga_recip_patch_                    F77_FUNC_(ga_recip_patch, GA_RECIP_PATCH)
#define ga_release_                        F77_FUNC_(ga_release, GA_RELEASE)
#define ga_release_update_                 F77_FUNC_(ga_release_update, GA_RELEASE_UPDATE)
#define ga_sadd_                           F77_FUNC_(ga_sadd, GA_SADD)
#define ga_sadd_patch_                     F77_FUNC_(ga_sadd_patch, GA_SADD_PATCH)
#define ga_scale_                          F77_FUNC_(ga_scale, GA_SCALE)
#define ga_scale_cols_                     F77_FUNC_(ga_scale_cols, GA_SCALE_COLS)
#define ga_scale_patch_                    F77_FUNC_(ga_scale_patch, GA_SCALE_PATCH)
#define ga_scale_rows_                     F77_FUNC_(ga_scale_rows, GA_SCALE_ROWS)
#define ga_scan_add_                       F77_FUNC_(ga_scan_add, GA_SCAN_ADD)
#define ga_scan_copy_                      F77_FUNC_(ga_scan_copy, GA_SCAN_COPY)
#define ga_scatter_                        F77_FUNC_(ga_scatter, GA_SCATTER)
#define ga_scatter_acc_                    F77_FUNC_(ga_scatter_acc, GA_SCATTER_ACC)
#define ga_sdot_                           F77_FUNC_(ga_sdot, GA_SDOT)
#define ga_sdot_patch_                     F77_FUNC_(ga_sdot_patch, GA_SDOT_PATCH)
#define ga_set_array_name_                 F77_FUNC_(ga_set_array_name, GA_SET_ARRAY_NAME)
#define ga_set_block_cyclic_               F77_FUNC_(ga_set_block_cyclic, GA_SET_BLOCK_CYCLIC)
#define ga_set_block_cyclic_proc_grid_     F77_FUNC_(ga_set_block_cyclic_proc_grid, GA_SET_BLOCK_CYCLIC_PROC_GRID)
#define ga_set_chunk_                      F77_FUNC_(ga_set_chunk, GA_SET_CHUNK)
#define ga_set_data_                       F77_FUNC_(ga_set_data, GA_SET_DATA)
#define ga_set_debug_                      F77_FUNC_(ga_set_debug, GA_SET_DEBUG)
#define ga_set_diagonal_                   F77_FUNC_(ga_set_diagonal, GA_SET_DIAGONAL)
#define ga_set_ghost_corner_flag_          F77_FUNC_(ga_set_ghost_corner_flag, GA_SET_GHOST_CORNER_FLAG)
#define ga_set_ghosts_                     F77_FUNC_(ga_set_ghosts, GA_SET_GHOSTS)
#define ga_set_irreg_distr_                F77_FUNC_(ga_set_irreg_distr, GA_SET_IRREG_DISTR)
#define ga_set_irreg_flag_                 F77_FUNC_(ga_set_irreg_flag, GA_SET_IRREG_FLAG)
#define ga_set_memory_limit_               F77_FUNC_(ga_set_memory_limit, GA_SET_MEMORY_LIMIT)
#define ga_set_pgroup_                     F77_FUNC_(ga_set_pgroup, GA_SET_PGROUP)
#define ga_set_restricted_                 F77_FUNC_(ga_set_restricted, GA_SET_RESTRICTED)
#define ga_set_restricted_range_           F77_FUNC_(ga_set_restricted_range, GA_SET_RESTRICTED_RANGE)
#define ga_set_update4_info_               F77_FUNC_(ga_set_update4_info, GA_SET_UPDATE4_INFO)
#define ga_set_update5_info_               F77_FUNC_(ga_set_update5_info, GA_SET_UPDATE5_INFO)
#define ga_sfill_                          F77_FUNC_(ga_sfill, GA_SFILL)
#define ga_sfill_patch_                    F77_FUNC_(ga_sfill_patch, GA_SFILL_PATCH)
#define ga_sgemm_                          F77_FUNC_(ga_sgemm, GA_SGEMM)
#define ga_sgop_                           F77_FUNC_(ga_sgop, GA_SGOP)
#define ga_shift_diagonal_                 F77_FUNC_(ga_shift_diagonal, GA_SHIFT_DIAGONAL)
#define ga_solve_                          F77_FUNC_(ga_solve, GA_SOLVE)
#define ga_sort_permut_                    F77_FUNC_(ga_sort_permut, GA_SORT_PERMUT)
#define ga_spd_invert_                     F77_FUNC_(ga_spd_invert, GA_SPD_INVERT)
#define ga_sscal_                          F77_FUNC_(ga_sscal, GA_SSCAL)
#define ga_sscal_patch_                    F77_FUNC_(ga_sscal_patch, GA_SSCAL_PATCH)
#define ga_step_bound_info_                F77_FUNC_(ga_step_bound_info, GA_STEP_BOUND_INFO)
#define ga_step_bound_info_patch_          F77_FUNC_(ga_step_bound_info_patch, GA_STEP_BOUND_INFO_PATCH)
#define ga_step_max_                       F77_FUNC_(ga_step_max, GA_STEP_MAX)
#define ga_step_max_patch_                 F77_FUNC_(ga_step_max_patch, GA_STEP_MAX_PATCH)
#define ga_summarize_                      F77_FUNC_(ga_summarize, GA_SUMMARIZE)
#define ga_symmetrize_                     F77_FUNC_(ga_symmetrize, GA_SYMMETRIZE)
#define ga_sync_                           F77_FUNC_(ga_sync, GA_SYNC)
#define ga_terminate_                      F77_FUNC_(ga_terminate, GA_TERMINATE)
#define ga_total_blocks_                   F77_FUNC_(ga_total_blocks, GA_TOTAL_BLOCKS)
#define ga_transpose_                      F77_FUNC_(ga_transpose, GA_TRANSPOSE)
#define ga_type_c2f_                       F77_FUNC_(ga_type_c2f, GA_TYPE_C2F)
#define ga_type_f2c_                       F77_FUNC_(ga_type_f2c, GA_TYPE_F2C)
#define ga_unlock_                         F77_FUNC_(ga_unlock, GA_UNLOCK)
#define ga_unpack_                         F77_FUNC_(ga_unpack, GA_UNPACK)
#define ga_update1_ghosts_                 F77_FUNC_(ga_update1_ghosts, GA_UPDATE1_GHOSTS)
#define ga_update2_ghosts_                 F77_FUNC_(ga_update2_ghosts, GA_UPDATE2_GHOSTS)
#define ga_update3_ghosts_                 F77_FUNC_(ga_update3_ghosts, GA_UPDATE3_GHOSTS)
#define ga_update4_ghosts_                 F77_FUNC_(ga_update4_ghosts, GA_UPDATE4_GHOSTS)
#define ga_update5_ghosts_                 F77_FUNC_(ga_update5_ghosts, GA_UPDATE5_GHOSTS)
#define ga_update6_ghosts_                 F77_FUNC_(ga_update6_ghosts, GA_UPDATE6_GHOSTS)
#define ga_update7_ghosts_                 F77_FUNC_(ga_update7_ghosts, GA_UPDATE7_GHOSTS)
#define ga_update_ghosts_                  F77_FUNC_(ga_update_ghosts, GA_UPDATE_GHOSTS)
#define ga_uses_ma_                        F77_FUNC_(ga_uses_ma, GA_USES_MA)
#define ga_uses_proc_grid_                 F77_FUNC_(ga_uses_proc_grid, GA_USES_PROC_GRID)
#define ga_valid_handle_                   F77_FUNC_(ga_valid_handle, GA_VALID_HANDLE)
#define ga_verify_handle_                  F77_FUNC_(ga_verify_handle, GA_VERIFY_HANDLE)
#define ga_wtime_                          F77_FUNC_(ga_wtime, GA_WTIME)
#define ga_zadd_                           F77_FUNC_(ga_zadd, GA_ZADD)
#define ga_zadd_patch_                     F77_FUNC_(ga_zadd_patch, GA_ZADD_PATCH)
#define ga_zero_                           F77_FUNC_(ga_zero, GA_ZERO)
#define ga_zero_diagonal_                  F77_FUNC_(ga_zero_diagonal, GA_ZERO_DIAGONAL)
#define ga_zfill_                          F77_FUNC_(ga_zfill, GA_ZFILL)
#define ga_zfill_patch_                    F77_FUNC_(ga_zfill_patch, GA_ZFILL_PATCH)
#define ga_zgemm_                          F77_FUNC_(ga_zgemm, GA_ZGEMM)
#define ga_zgop_                           F77_FUNC_(ga_zgop, GA_ZGOP)
#define ga_zscal_                          F77_FUNC_(ga_zscal, GA_ZSCAL)
#define ga_zscal_patch_                    F77_FUNC_(ga_zscal_patch, GA_ZSCAL_PATCH)
#define gai_cdot_                          F77_FUNC_(gai_cdot, GAI_CDOT)
#define gai_cdot_patch_                    F77_FUNC_(gai_cdot_patch, GAI_CDOT_PATCH)
#define gai_zdot_                          F77_FUNC_(gai_zdot, GAI_ZDOT)
#define gai_zdot_patch_                    F77_FUNC_(gai_zdot_patch, GAI_ZDOT_PATCH)
#define nga_acc_                           F77_FUNC_(nga_acc, NGA_ACC)
#define nga_access_                        F77_FUNC_(nga_access, NGA_ACCESS)
#define nga_access_block_                  F77_FUNC_(nga_access_block, NGA_ACCESS_BLOCK)
#define nga_access_block_                  F77_FUNC_(nga_access_block, NGA_ACCESS_BLOCK)
#define nga_access_block_segment_          F77_FUNC_(nga_access_block_segment, NGA_ACCESS_BLOCK_SEGMENT)
#define nga_access_ghost_element_          F77_FUNC_(nga_access_ghost_element, NGA_ACCESS_GHOST_ELEMENT)
#define nga_access_ghosts_                 F77_FUNC_(nga_access_ghosts, NGA_ACCESS_GHOSTS)
#define nga_add_patch_                     F77_FUNC_(nga_add_patch, NGA_ADD_PATCH)
#define nga_allocate_                      F77_FUNC_(nga_allocate, NGA_ALLOCATE)
#define nga_compare_distr_                 F77_FUNC_(nga_compare_distr, NGA_COMPARE_DISTR)
#define nga_copy_patch_                    F77_FUNC_(nga_copy_patch, NGA_COPY_PATCH)
#define nga_create_                        F77_FUNC_(nga_create, NGA_CREATE)
#define nga_create_config_                 F77_FUNC_(nga_create_config, NGA_CREATE_CONFIG)
#define nga_create_ghosts_                 F77_FUNC_(nga_create_ghosts, NGA_CREATE_GHOSTS)
#define nga_create_ghosts_config_          F77_FUNC_(nga_create_ghosts_config, NGA_CREATE_GHOSTS_CONFIG)
#define nga_create_ghosts_irreg_           F77_FUNC_(nga_create_ghosts_irreg, NGA_CREATE_GHOSTS_IRREG)
#define nga_create_ghosts_irreg_config_    F77_FUNC_(nga_create_ghosts_irreg_config, NGA_CREATE_GHOSTS_IRREG_CONFIG)
#define nga_create_irreg_                  F77_FUNC_(nga_create_irreg, NGA_CREATE_IRREG)
#define nga_create_irreg_config_           F77_FUNC_(nga_create_irreg_config, NGA_CREATE_IRREG_CONFIG)
#define nga_create_handle_                 F77_FUNC_(nga_create_handle, NGA_CREATE_HANDLE)
#define nga_create_mutexes_                F77_FUNC_(nga_create_mutexes, NGA_CREATE_MUTEXES)
#define nga_destroy_                       F77_FUNC_(nga_destroy, NGA_DESTROY)
#define nga_destroy_mutexes_               F77_FUNC_(nga_destroy_mutexes, NGA_DESTROY_MUTEXES)
#define nga_ddot_patch_                    F77_FUNC_(nga_ddot_patch, NGA_DDOT_PATCH)
#define nga_distribution_                  F77_FUNC_(nga_distribution, NGA_DISTRIBUTION)
#define nga_duplicate_                     F77_FUNC_(nga_duplicate, NGA_DUPLICATE)
#define nga_fill_                          F77_FUNC_(nga_fill, NGA_FILL)
#define nga_fill_patch_                    F77_FUNC_(nga_fill_patch, NGA_FILL_PATCH)
#define nga_gather_                        F77_FUNC_(nga_gather, NGA_GATHER)
#define nga_get_                           F77_FUNC_(nga_get, NGA_GET)
#define nga_get_block_info_                F77_FUNC_(nga_get_block_info, NGA_GET_BLOCK_INFO)
#define nga_get_ghost_block_               F77_FUNC_(nga_get_ghost_block, NGA_GET_GHOST_BLOCK)
#define nga_idot_patch_                    F77_FUNC_(nga_idot_patch, NGA_IDOT_PATCH)
#define nga_inquire_                       F77_FUNC_(nga_inquire, NGA_INQUIRE)
#define nga_inquire_internal_              F77_FUNC_(nga_inquire_internal, NGA_INQUIRE_INTERNAL)
#define nga_locate_                        F77_FUNC_(nga_locate, NGA_LOCATE)
#define nga_locate_num_blocks_             F77_FUNC_(nga_locate_num_blocks, NGA_LOCATE_NUM_BLOCKS)
#define nga_locate_region_                 F77_FUNC_(nga_locate_region, NGA_LOCATE_REGION)
#define nga_matmul_patch_                  F77_FUNC_(nga_matmul_patch, NGA_MATMUL_PATCH)
#define nga_merge_distr_patch_             F77_FUNC_(nga_merge_distr_patch, NGA_MERGE_DISTR_PATCH)
#define nga_merge_distr_patch_             F77_FUNC_(nga_merge_distr_patch, NGA_MERGE_DISTR_PATCH)
#define nga_nbacc_                         F77_FUNC_(nga_nbacc, NGA_NBACC)
#define nga_nbget_                         F77_FUNC_(nga_nbget, NGA_NBGET)
#define nga_nbget_ghost_dir_               F77_FUNC_(nga_nbget_ghost_dir, NGA_NBGET_GHOST_DIR)
#define nga_nbput_                         F77_FUNC_(nga_nbput, NGA_NBPUT)
#define nga_nbtest_                        F77_FUNC_(nga_nbtest, NGA_NBTEST)
#define nga_nbwait_                        F77_FUNC_(nga_nbwait, NGA_NBWAIT)
#define nga_nnodes_                        F77_FUNC_(nga_nnodes, NGA_NNODES)
#define nga_nodeid_                        F77_FUNC_(nga_nodeid, NGA_NODEID)
#define nga_periodic_acc_                  F77_FUNC_(nga_periodic_acc, NGA_PERIODIC_ACC)
#define nga_periodic_get_                  F77_FUNC_(nga_periodic_get, NGA_PERIODIC_GET)
#define nga_periodic_put_                  F77_FUNC_(nga_periodic_put, NGA_PERIODIC_PUT)
#define nga_print_patch_                   F77_FUNC_(nga_print_patch, NGA_PRINT_PATCH)
#define nga_proc_topology_                 F77_FUNC_(nga_proc_topology, NGA_PROC_TOPOLOGY)
#define nga_put_                           F77_FUNC_(nga_put, NGA_PUT)
#define nga_read_inc_                      F77_FUNC_(nga_read_inc, NGA_READ_INC)
#define nga_release_                       F77_FUNC_(nga_release, NGA_RELEASE)
#define nga_release_block_                 F77_FUNC_(nga_release_block, NGA_RELEASE_BLOCK)
#define nga_release_block_grid_            F77_FUNC_(nga_release_block_grid, NGA_RELEASE_BLOCK_GRID)
#define nga_release_block_segment_         F77_FUNC_(nga_release_block_segment, NGA_RELEASE_BLOCK_SEGMENT)
#define nga_release_ghost_element_         F77_FUNC_(nga_release_ghost_element, NGA_RELEASE_GHOST_ELEMENT)
#define nga_release_ghosts_                F77_FUNC_(nga_release_ghosts, NGA_RELEASE_GHOSTS)
#define nga_release_update_                F77_FUNC_(nga_release_update, NGA_RELEASE_UPDATE)
#define nga_release_update_block_          F77_FUNC_(nga_release_update_block, NGA_RELEASE_UPDATE_BLOCK)
#define nga_release_update_block_grid_     F77_FUNC_(nga_release_update_block_grid, NGA_RELEASE_UPDATE_BLOCK_GRID)
#define nga_release_update_block_segment_  F77_FUNC_(nga_release_update_block_segment, NGA_RELEASE_UPDATE_BLOCK_SEGMENT)
#define nga_release_update_ghost_element_  F77_FUNC_(nga_release_update_ghost_element, NGA_RELEASE_UPDATE_GHOST_ELEMENT)
#define nga_release_update_ghosts_         F77_FUNC_(nga_release_update_ghosts, NGA_RELEASE_UPDATE_GHOSTS)
#define nga_scale_patch_                   F77_FUNC_(nga_scale_patch, NGA_SCALE_PATCH)
#define nga_scatter_                       F77_FUNC_(nga_scatter, NGA_SCATTER)
#define nga_scatter_acc_                   F77_FUNC_(nga_scatter_acc, NGA_SCATTER_ACC)
#define nga_sdot_patch_                    F77_FUNC_(nga_sdot_patch, NGA_SDOT_PATCH)
#define nga_select_elem_                   F77_FUNC_(nga_select_elem, NGA_SELECT_ELEM)
#define nga_strided_acc_                   F77_FUNC_(nga_strided_acc, NGA_STRIDED_ACC)
#define nga_strided_get_                   F77_FUNC_(nga_strided_get, NGA_STRIDED_GET)
#define nga_strided_put_                   F77_FUNC_(nga_strided_put, NGA_STRIDED_PUT)
#define nga_update_ghost_dir_              F77_FUNC_(nga_update_ghost_dir, NGA_UPDATE_GHOST_DIR)
#define nga_zero_patch_                    F77_FUNC_(nga_zero_patch, NGA_ZERO_PATCH)
#define ngai_cdot_patch_                   F77_FUNC_(ngai_cdot_patch, NGAI_CDOT_PATCH)
#define ngai_zdot_patch_                   F77_FUNC_(ngai_zdot_patch, NGAI_ZDOT_PATCH)

#endif /* C_NAMES_H */
