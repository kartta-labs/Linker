python step1_pred_pairwise_novgg_withcap.py --preprocess_dir "/data/zekunl/Geolocalizer/google_grouping/" \--map_dir "/data/zekunl/mydata/historical-map-groundtruth-25/" \
--pred_output_dir "predictions_v3_23_usgs_cap/" \
--map_type "usgs" \
--dml_weight_dir "../scripts_v3/weights/" \
--dml_weight_name "23_usgs_textLinking_text_loc_angle_size_cap_nontrainable_diffef1_best.hdf5" 


python step1_pred_pairwise_novgg_withoutcap.py --preprocess_dir "/data/zekunl/Geolocalizer/google_grouping/" \
--map_dir "/data/zekunl/mydata/historical-map-groundtruth-25/" \
--pred_output_dir "predictions_v1_19_usgs_nocap/" \
--map_type "usgs" \
--dml_weight_dir "../scripts/weights/" \
--dml_weight_name "19_textLinking_triploss_clf_noattention_loc_angle_size_knnneg_best.hdf5" 