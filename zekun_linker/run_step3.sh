python step3_generate_phrases.py --input_pred_dir "step2_predictions_v3_23_usgs_cap_p0.5/" \
--IF_WCC \
--output_phrases_dir "phrases_v3_23_usgs_cap_p0.5_wcc/"

python step3_generate_phrases.py --input_pred_dir "step2_predictions_v1_19_usgs_nocap_p0.5/" \
--IF_WCC \
--output_phrases_dir "phrases_v1_19_usgs_nocap_p0.5_wcc/"

###################################################################################

python step3_generate_phrases.py --input_pred_dir "step2_predictions_v3_23_usgs_cap_p0.5/" \
--output_phrases_dir "phrases_v3_23_usgs_cap_p0.5_scc/"

python step3_generate_phrases.py --input_pred_dir "step2_predictions_v1_19_usgs_nocap_p0.5/" \
--output_phrases_dir "phrases_v1_19_usgs_nocap_p0.5_scc/"