#ifndef  C_NAMES_H
#define  C_NAMES_H

#if defined(CRAY) || defined(WIN32) || defined(HITACHI)

#define  ga_acc_                   GA_ACC
#define  ga_access_                GA_ACCESS
#define  ga_add_                   GA_ADD
#define  ga_add_patch_             GA_ADD_PATCH
#define  ga_brdcst_                GA_BRDCST
#define  ga_check_handle_          GA_CHECK_HANDLE
#define  ga_compare_distr_         GA_COMPARE_DISTR
#define  ga_copy_                  GA_COPY
#define  ga_copy_patch_            GA_COPY_PATCH
#define  ga_copy_patch_dp_         GA_COPY_PATCH_DP
#define  ga_create_                GA_CREATE
#define  ga_create_irreg_          GA_CREATE_IRREG
#define  ga_create_mutexes_        GA_CREATE_MUTEXES
#define  ga_ddot_                  GA_DDOT
#define  ga_ddot_patch_dp_         GA_DDOT_PATCH_DP
#define  ga_destroy_               GA_DESTROY
#define  ga_destroy_mutexes_       GA_DESTROY_MUTEXES
#define  ga_dgop_                  GA_DGOP
#define  ga_diag_                  GA_DIAG
#define  ga_diag_reuse_            GA_DIAG_REUSE
#define  ga_diag_seq_              GA_DIAG_SEQ
#define  ga_diag_std_              GA_DIAG_STD
#define  ga_diag_std_seq_          GA_DIAG_STD_SEQ
#define  ga_distribution_          GA_DISTRIBUTION
#define  ga_duplicate_             GA_DUPLICATE
#define  ga_error_                 GA_ERROR
#define  ga_fence_                 GA_FENCE
#define  ga_fill_                  GA_FILL
#define  ga_fill_patch_            GA_FILL_PATCH
#define  ga_gather_                GA_GATHER
#define  ga_get_                   GA_GET
#define  ga_has_ghosts_            GA_HAS_GHOSTS
#define  ga_idot_                  GA_IDOT
#define  ga_igop_                  GA_IGOP
#define  ga_init_fence_            GA_INIT_FENCE
#define  ga_initialize_            GA_INITIALIZE
#define  ga_initialize_ltd_        GA_INITIALIZE_LTD
#define  ga_inquire_               GA_INQUIRE
#define  ga_inquire_memory_        GA_INQUIRE_MEMORY
#define  ga_inquire_name_          GA_INQUIRE_NAME
#define  ga_list_nodeid_           GA_LIST_NODEID
#define  ga_llt_solve_             GA_LLT_SOLVE
#define  ga_locate_                GA_LOCATE
#define  ga_locate_region_         GA_LOCATE_REGION
#define  ga_lock_                  GA_LOCK
#define  ga_lu_solve_alt_          GA_LU_SOLVE_ALT
#define  ga_ma_base_address_       GA_MA_BASE_ADDRESS
#define  ga_ma_diff_               GA_MA_DIFF
#define  ga_ma_get_ptr_            GA_MA_GET_PTR
#define  ga_mask_sync_             GA_MASK_SYNC
#define  ga_ghost_barrier_         GA_GHOST_BARRIER
#define  ga_matmul_patch_          GA_MATMUL_PATCH
#define  ga_memory_avail_          GA_MEMORY_AVAIL
#define  ga_memory_limited_        GA_MEMORY_LIMITED
#define  ga_nblock_                GA_NBLOCK
#define  ga_ndim_                  GA_NDIM
#define  ga_net_nnodes_            GA_NET_NNODES
#define  ga_net_nodeid_            GA_NET_NODEID
#define  ga_nnodes_                GA_NNODES
#define  ga_nodeid_                GA_NODEID
#define  ga_pack_                  GA_PACK
#define  ga_patch_enum_            GA_PATCH_ENUM
#define  ga_print_                 GA_PRINT
#define  ga_print_distribution_    GA_PRINT_DISTRIBUTION
#define  ga_print_patch_           GA_PRINT_PATCH
#define  ga_print_patch_           GA_PRINT_PATCH
#define  ga_print_stats_           GA_PRINT_STATS
#define  ga_proc_topology_         GA_PROC_TOPOLOGY
#define  ga_put_                   GA_PUT
#define  ga_read_inc_              GA_READ_INC
#define  ga_release_               GA_RELEASE
#define  ga_release_update_        GA_RELEASE_UPDATE
#define  ga_scale_                 GA_SCALE
#define  ga_scale_patch_           GA_SCALE_PATCH
#define  ga_scan_add_              GA_SCAN_ADD
#define  ga_scan_copy_             GA_SCAN_COPY
#define  ga_scatter_               GA_SCATTER
#define  ga_scatter_acc_           GA_SCATTER_ACC
#define  ga_sdot_                  GA_SDOT
#define  ga_set_memory_limit_      GA_SET_MEMORY_LIMIT
#define  ga_sgop_                  GA_SGOP
#define  ga_solve_                 GA_SOLVE
#define  ga_sort_scat2_            GA_SORT_SCAT2
#define  ga_spd_invert_            GA_SPD_INVERT
#define  ga_summarize_             GA_SUMMARIZE
#define  ga_symmetrize_            GA_SYMMETRIZE
#define  ga_sync_                  GA_SYNC
#define  ga_terminate_             GA_TERMINATE
#define  ga_transpose_             GA_TRANSPOSE
#define  ga_unlock_                GA_UNLOCK
#define  ga_unpack_                GA_UNPACK
#define  ga_update1_ghosts_        GA_UPDATE1_GHOSTS
#define  ga_update2_ghosts_        GA_UPDATE2_GHOSTS
#define  ga_update3_ghosts_        GA_UPDATE3_GHOSTS
#define  ga_update4_ghosts_        GA_UPDATE4_GHOSTS
#define  ga_update5_ghosts_        GA_UPDATE5_GHOSTS
#define  ga_update6_ghosts_        GA_UPDATE6_GHOSTS
#define  nga_update_ghost_dir_     NGA_UPDATE_GHOST_DIR
#define  ga_update_ghosts_         GA_UPDATE_GHOSTS
#define  ga_uses_ma_               GA_USES_MA
#define  ga_valid_handle_          GA_VALID_HANDLE
#define  ga_verify_handle_         GA_VERIFY_HANDLE
#define  ga_zdot_                  GA_ZDOT
#define  ga_zero_                  GA_ZERO
#define  gai_dot_                  GAI_DOT
#define  gai_dot_patch_            GAI_DOT_PATCH
#define  nga_acc_                  NGA_ACC
#define  nga_access_               NGA_ACCESS
#define  nga_access_ghosts_        NGA_ACCESS_GHOSTS
#define  nga_access_ghost_element_ NGA_ACCESS_GHOST_ELEMENT
#define  nga_add_patch_            NGA_ADD_PATCH
#define  nga_copy_patch_           NGA_COPY_PATCH
#define  nga_create_               NGA_CREATE
#define  nga_create_ghosts_        NGA_CREATE_GHOSTS
#define  nga_create_ghosts_irreg_  NGA_CREATE_GHOSTS_IRREG
#define  nga_create_irreg_         NGA_CREATE_IRREG
#define  nga_ddot_patch_           NGA_DDOT_PATCH
#define  nga_distribution_         NGA_DISTRIBUTION
#define  nga_fill_patch_           NGA_FILL_PATCH
#define  nga_gather_               NGA_GATHER
#define  nga_get_                  NGA_GET
#define  nga_idot_patch_           NGA_IDOT_PATCH
#define  nga_inquire_              NGA_INQUIRE
#define  nga_locate_               NGA_LOCATE
#define  nga_locate_region_        NGA_LOCATE_REGION
#define  nga_matmul_patch_         NGA_MATMUL_PATCH
#define  nga_periodic_acc_         NGA_PERIODIC_ACC
#define  nga_periodic_get_         NGA_PERIODIC_GET
#define  nga_periodic_put_         NGA_PERIODIC_PUT
#define  nga_print_patch_          NGA_PRINT_PATCH
#define  nga_put_                  NGA_PUT
#define  nga_read_inc_             NGA_READ_INC
#define  nga_release_              NGA_RELEASE
#define  nga_release_update_       NGA_RELEASE_UPDATE
#define  nga_scale_patch_          NGA_SCALE_PATCH
#define  nga_scatter_              NGA_SCATTER
#define  nga_scatter_acc_          NGA_SCATTER_ACC
#define  nga_sdot_patch_           NGA_SDOT_PATCH   
#define  nga_zdot_patch_           NGA_ZDOT_PATCH
#define  nga_zero_patch_           NGA_ZERO_PATCH
#define  ngai_dot_patch_           NGAI_DOT_PATCH
#define  ga_abs_value_             GA_ABS_VALUE
#define  ga_add_constant_	   GA_ADD_CONSTANT 
#define  ga_recip_                 GA_RECIP 
#define  ga_elem_multiply_	   GA_ELEM_MULTIPLY  
#define  ga_elem_divide_	   GA_ELEM_DIVIDE  
#define  ga_step_max_	           GA_STEP_MAX  
#define  ga_step_max2_	           GA_STEP_MAX2  
#define  ga_step_max_patch_	   GA_STEP_MAX_PATCH  
#define  ga_step_max2_patch_	   GA_STEP_MAX2_PATCH  
#define  ga_elem_maximum_	   GA_ELEM_MAXIMUM  
#define  ga_elem_minimum_	   GA_ELEM_MINIMUM  
#define  ga_aba_value_patch_       GA_ABS_VALUE_PATCH
#define  ga_add_constant_patch_	   GA_ADD_CONSTANT_PATCH
#define  ga_recip_patch_ 	   GA_RECIP_PATCH
#define  ga_elem_multiply_patch_   GA_ELEM_MULTIPLY_PATCH	
#define  ga_elem_divide_patch_     GA_ELEM_DIVIDE_PATCH	
#define  ga_elem_maxumum_patch_    GA_ELEM_MAXIMUM_PATCH	
#define  ga_elem_minumum_patch_    GA_ELEM_MINIMUM_PATCH	
#define  ga_shift_diagonal_        GA_SHIFT_DIAGONAL
#define  ga_set_diagonal_          GA_SET_DIAGONAL
#define  ga_zero_diagonal_         GA_ZERO_DIAGONAL
#define  ga_add_diagonal_          GA_ADD_DIAGONAL
#define  ga_get_diagonal_          GA_GET_DIAGONAL
#define  ga_scale_rows_            GA_SCALE_ROWS
#define  ga_scale_cols_            GA_SCALE_COLS
#define  ga_norm1_          	   GA_NORM1
#define  ga_norm_infinity_         GA_NORM_INFINITY
#define  ga_median_         	   GA_MEDIAN
#define  ga_median_patch_          GA_MEDIAN_PATCH
#define  ga_cluster_nodeid_        GA_CLUSTER_NODEID 
#define  ga_cluster_nnodes_        GA_CLUSTER_NNODES
#define  ga_cluster_nprocs_        GA_CLUSTER_NPROCS
#define  ga_cluster_procid_        GA_CLUSTER_PROCID
#define  ga_dgemm_                 GA_DGEMM
#define  ga_sgemm_                 GA_SGEMM
#define  ga_zgemm_                 GA_ZGEMM

#elif defined(F2C2_)

#define  ga_acc_                   ga_acc__                 
#define  ga_access_                ga_access__              
#define  ga_add_                   ga_add__                 
#define  ga_add_patch_             ga_add_patch__           
#define  ga_brdcst_                ga_brdcst__              
#define  ga_check_handle_          ga_check_handle__        
#define  ga_compare_distr_         ga_compare_distr__       
#define  ga_copy_                  ga_copy__                
#define  ga_copy_patch_            ga_copy_patch__          
#define  ga_copy_patch_dp_         ga_copy_patch_dp__       
#define  ga_create_                ga_create__              
#define  ga_create_irreg_          ga_create_irreg__        
#define  ga_create_mutexes_        ga_create_mutexes__      
#define  ga_ddot_                  ga_ddot__                
#define  ga_ddot_patch_dp_         ga_ddot_patch_dp__       
#define  ga_destroy_               ga_destroy__             
#define  ga_destroy_mutexes_       ga_destroy_mutexes__     
#define  ga_dgop_                  ga_dgop__                
#define  ga_diag_                  ga_diag__                
#define  ga_diag_reuse_            ga_diag_reuse__          
#define  ga_diag_seq_              ga_diag_seq__
#define  ga_diag_std_              ga_diag_std__
#define  ga_diag_std_seq_          ga_diag_std_seq__
#define  ga_distribution_          ga_distribution__        
#define  ga_duplicate_             ga_duplicate__           
#define  ga_error_                 ga_error__               
#define  ga_fence_                 ga_fence__               
#define  ga_fill_                  ga_fill__                
#define  ga_fill_patch_            ga_fill_patch__          
#define  ga_gather_                ga_gather__              
#define  ga_get_                   ga_get__                 
#define  ga_has_ghosts_            ga_has_ghosts__          
#define  ga_idot_                  ga_idot__                
#define  ga_igop_                  ga_igop__                
#define  ga_init_fence_            ga_init_fence__          
#define  ga_initialize_            ga_initialize__          
#define  ga_initialize_ltd_        ga_initialize_ltd__      
#define  ga_inquire_               ga_inquire__             
#define  ga_inquire_memory_        ga_inquire_memory__      
#define  ga_inquire_name_          ga_inquire_name__        
#define  ga_list_nodeid_           ga_list_nodeid__         
#define  ga_llt_solve_             ga_llt_solve__           
#define  ga_locate_                ga_locate__              
#define  ga_locate_region_         ga_locate_region__       
#define  ga_lock_                  ga_lock__                
#define  ga_lu_solve_alt_          ga_lu_solve_alt__
#define  ga_ma_base_address_       ga_ma_base_address__     
#define  ga_ma_diff_               ga_ma_diff__             
#define  ga_ma_get_ptr_            ga_ma_get_ptr__          
#define  ga_mask_sync_             ga_mask_sync__           
#define  ga_ghost_barrier_         ga_ghost_barrier__       
#define  ga_matmul_patch_          ga_matmul_patch__        
#define  ga_memory_avail_          ga_memory_avail__        
#define  ga_memory_limited_        ga_memory_limited__      
#define  ga_nblock_                ga_nblock__              
#define  ga_ndim_                  ga_ndim__                
#define  ga_net_nnodes_            ga_net_nnodes__          
#define  ga_net_nodeid_            ga_net_nodeid__          
#define  ga_nnodes_                ga_nnodes__              
#define  ga_nodeid_                ga_nodeid__              
#define  ga_pack_                  ga_pack__
#define  ga_patch_enum_            ga_patch_enum__          
#define  ga_print_                 ga_print__               
#define  ga_print_distribution_    ga_print_distribution__  
#define  ga_print_patch_           ga_print_patch__         
#define  ga_print_patch_           ga_print_patch__         
#define  ga_print_stats_           ga_print_stats__         
#define  ga_proc_topology_         ga_proc_topology__       
#define  ga_put_                   ga_put__                 
#define  ga_read_inc_              ga_read_inc__            
#define  ga_reinit_handler_        ga_reinit_handler__      
#define  ga_release_               ga_release__             
#define  ga_release_update_        ga_release_update__      
#define  ga_scan_add_              ga_scan_add__
#define  ga_scan_copy_             ga_scan_copy__
#define  ga_scale_                 ga_scale__               
#define  ga_scale_patch_           ga_scale_patch__         
#define  ga_scatter_               ga_scatter__             
#define  ga_scatter_acc_           ga_scatter_acc__         
#define  ga_sdot_                  ga_sdot__                
#define  ga_set_memory_limit_      ga_set_memory_limit__    
#define  ga_sgop_                  ga_sgop__                
#define  ga_solve_                 ga_solve__
#define  ga_sort_scat2_            ga_sort_scat2__          
#define  ga_spd_invert_            ga_spd_invert__
#define  ga_summarize_             ga_summarize__           
#define  ga_symmetrize_            ga_symmetrize__          
#define  ga_sync_                  ga_sync__                
#define  ga_terminate_             ga_terminate__           
#define  ga_transpose_             ga_transpose__           
#define  ga_unlock_                ga_unlock__              
#define  ga_unpack_                ga_unpack__
#define  ga_update1_ghosts_        ga_update1_ghosts__     
#define  ga_update2_ghosts_        ga_update2_ghosts__      
#define  ga_update3_ghosts_        ga_update3_ghosts__    
#define  ga_update4_ghosts_        ga_update4_ghosts__   
#define  ga_update5_ghosts_        ga_update5_ghosts__  
#define  ga_update6_ghosts_        ga_update6_ghosts__  
#define  nga_update_ghost_dir_     nga_update_ghost_dir__   
#define  ga_update_ghosts_         ga_update_ghosts__     
#define  ga_uses_ma_               ga_uses_ma__             
#define  ga_valid_handle_          ga_valid_handle__        
#define  ga_verify_handle_         ga_verify_handle__    
#define  ga_zdot_                  ga_zdot__                
#define  ga_zero_                  ga_zero__                
#define  gai_dot_                  gai_dot__                
#define  gai_dot_patch_            gai_dot_patch__          
#define  nga_acc_                  nga_acc__                
#define  nga_access_               nga_access__             
#define  nga_access_ghosts_        nga_access_ghosts__      
#define  nga_access_ghost_element_ nga_access_ghost_element__      
#define  nga_add_patch_            nga_add_patch__          
#define  nga_copy_patch_           nga_copy_patch__         
#define  nga_create_               nga_create__             
#define  nga_create_ghosts_        nga_create_ghosts__      
#define  nga_create_ghosts_irreg_  nga_create_ghosts_irreg__
#define  nga_create_irreg_         nga_create_irreg__       
#define  nga_ddot_patch_           nga_ddot_patch__         
#define  nga_distribution_         nga_distribution__       
#define  nga_fill_patch_           nga_fill_patch__         
#define  nga_gather_               nga_gather__             
#define  nga_get_                  nga_get__                
#define  nga_idot_patch_           nga_idot_patch__         
#define  nga_inquire_              nga_inquire__            
#define  nga_locate_               nga_locate__             
#define  nga_locate_region_        nga_locate_region__     
#define  nga_matmul_patch_         nga_matmul_patch__     
#define  nga_periodic_acc_         nga_periodic_acc__    
#define  nga_periodic_get_         nga_periodic_get__   
#define  nga_periodic_put_         nga_periodic_put__  
#define  nga_print_patch_          nga_print_patch__        
#define  nga_put_                  nga_put__                
#define  nga_read_inc_             nga_read_inc__           
#define  nga_release_              nga_release__            
#define  nga_release_update_       nga_release_update__  
#define  nga_scale_patch_          nga_scale_patch__        
#define  nga_scatter_              nga_scatter__            
#define  nga_scatter_acc_          nga_scatter_acc__        
#define  nga_sdot_patch_           nga_sdot_patch__            
#define  nga_select_elem_          nga_select_elem__
#define  nga_zdot_patch_           nga_zdot_patch__         
#define  nga_zero_patch_           nga_zero_patch__         
#define  ngai_dot_patch_           ngai_dot_patch__         
#define  ga_abs_value_             ga_abs_value__
#define  ga_add_constant_	   ga_add_constant__ 
#define  ga_recip_                 ga_recip__ 
#define  ga_elem_multiply_	   ga_elem_multiply__  
#define  ga_elem_divide_	   ga_elem_divide__  
#define  ga_step_max_	           ga_step_max__ 
#define  ga_step_max2_	           ga_step_max2__ 
#define  ga_step_max_patch_	   ga_step_max_patch__ 
#define  ga_step_max2_patch_	   ga_step_max2_patch__ 
#define  ga_elem_maximum_	   ga_elem_maximum__  
#define  ga_elem_minimum_	   ga_elem_minimum__  
#define  ga_aba_value_patch_       ga_abs_value_patch__
#define  ga_add_constant_patch_	   ga_add_constant_patch__
#define  ga_recip_patch_ 	   ga_recip_patch__
#define  ga_elem_multiply_patch_   ga_elem_multiply_patch__	
#define  ga_elem_divide_patch_     ga_elem_divide_patch__	
#define  ga_elem_maxumum_patch_    ga_elem_maximum_patch__	
#define  ga_elem_minumum_patch_    ga_elem_minimum_patch__	
#define  ga_shift_diagonal_        ga_shift_diagonal__
#define  ga_set_diagonal_          ga_set_diagonal__
#define  ga_zero_diagonal_         ga_zero_diagonal__
#define  ga_add_diagonal_          ga_add_diagonal__
#define  ga_get_diagonal_          ga_get_diagonal__
#define  ga_scale_rows_	           ga_scale_rows__		
#define  ga_scale_cols_	           ga_scale_cols__		
#define  ga_norm1_          	   ga_norm1__
#define  ga_norm_infinity_         ga_norm_infinity__
#define  ga_median_         	   ga_median__
#define  ga_median_patch_          ga_median_patch__
#define  ga_cluster_nodeid_        ga_cluster_nodeid__ 
#define  ga_cluster_nnodes_        ga_cluster_nnodes__
#define  ga_cluster_nprocs_        ga_cluster_nprocs__
#define  ga_cluster_procid_        ga_cluster_procid__
#define  ga_dgemm_                 ga_dgemm__
#define  ga_sgemm_                 ga_sgemm__
#define  ga_zgemm_                 ga_zgemm__

#endif


#endif


