datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)

cd ..

## regression results (SVR) - overview for paper
#datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/audio.yaml --name audio --results_base_dir ${datetime_str}_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/demographics_only.yaml --name demographics_only --results_base_dir ${datetime_str}_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/mean_prediction.yaml --name mean_prediction --results_base_dir ${datetime_str}_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/random_sampling_prediction.yaml --name random_sampling_prediction --results_base_dir ${datetime_str}_svr_reduced_feature_sets
#datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/train_test/reduced_features/audio.yaml --name audio --results_base_dir ${datetime_str}_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/train_test/reduced_features/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/train_test/reduced_features/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/train_test/demographics_only.yaml --name demographics_only --results_base_dir ${datetime_str}_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/train_test/mean_prediction.yaml --name mean_prediction --results_base_dir ${datetime_str}_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/train_test/random_sampling_prediction.yaml --name random_sampling_prediction --results_base_dir ${datetime_str}_svr_reduced_feature_sets_train_test
#
## with shap values (for feature importance analysis)
#datetime_str=$(date '+%Y%m%d_%H%M')_$(openssl rand -hex 2)
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/combined_audio_linguistic_shap_and_data.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_svr_reduced_feature_sets_shap_and_data

## SVR journaling, picnicScene, cookieTheft
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/journaling/audio.yaml --name audio --results_base_dir ${datetime_str}_journaling_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/journaling/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_journaling_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/journaling/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_journaling_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/journaling/train_test/audio.yaml --name audio --results_base_dir ${datetime_str}_journaling_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/journaling/train_test/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_journaling_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/journaling/train_test/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_journaling_svr_reduced_feature_sets_train_test
#
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/picnicScene/audio.yaml --name audio --results_base_dir ${datetime_str}_picnicScene_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/picnicScene/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_picnicScene_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/picnicScene/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_picnicScene_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/picnicScene/train_test/audio.yaml --name audio --results_base_dir ${datetime_str}_picnicScene_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/picnicScene/train_test/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_picnicScene_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/picnicScene/train_test/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_picnicScene_svr_reduced_feature_sets_train_test
#
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/cookieTheft/audio.yaml --name audio --results_base_dir ${datetime_str}_cookieTheft_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/cookieTheft/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_cookieTheft_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/cookieTheft/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_cookieTheft_svr_reduced_feature_sets
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/cookieTheft/train_test/audio.yaml --name audio --results_base_dir ${datetime_str}_cookieTheft_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/cookieTheft/train_test/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_cookieTheft_svr_reduced_feature_sets_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/cookieTheft/train_test/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_cookieTheft_svr_reduced_feature_sets_train_test
#
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/journaling/train_test/combined_audio_linguistic_shap_and_data.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_journaling_svr_reduced_feature_sets_train_test_shap_and_data
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/picnicScene/train_test/combined_audio_linguistic_shap_and_data.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_picnicScene_svr_reduced_feature_sets_train_test_shap_and_data
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/other_tasks/cookieTheft/train_test/combined_audio_linguistic_shap_and_data.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_cookieTheft_svr_reduced_feature_sets_train_test_shap_and_data
#
#
## cognitive low performers
#python -u run_multiple.py --config configs/classification/negative_outliers_norm/svc.yaml --name svc --results_base_dir ${datetime_str}_svc_classification_cognitive_low_perf_norm
#python -u run_multiple.py --config configs/classification/negative_outliers_norm/train_test/svc.yaml --name svc --results_base_dir ${datetime_str}_svc_classification_cognitive_low_perf_norm_train_test




# consent filters
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/audio.yaml --name audio --results_base_dir ${datetime_str}_svr_consent_with_audio
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_svr_consent_with_audio
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_svr_consent_with_audio
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/demographics_only.yaml --name demographics_only --results_base_dir ${datetime_str}_svr_consent_with_audio
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/mean_prediction.yaml --name mean_prediction --results_base_dir ${datetime_str}_svr_consent_with_audio
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/random_sampling_prediction.yaml --name random_sampling_prediction --results_base_dir ${datetime_str}_svr_consent_with_audio
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/train_test/audio.yaml --name audio --results_base_dir ${datetime_str}_svr_consent_with_audio_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/train_test/linguistic.yaml --name linguistic --results_base_dir ${datetime_str}_svr_consent_with_audio_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/train_test/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_svr_consent_with_audio_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/train_test/combined_audio_linguistic_and_data.yaml --name combined_audio_linguistic_and_data --results_base_dir ${datetime_str}_svr_consent_with_audio_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/train_test/demographics_only.yaml --name demographics_only --results_base_dir ${datetime_str}_svr_consent_with_audio_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/train_test/mean_prediction.yaml --name mean_prediction --results_base_dir ${datetime_str}_svr_consent_with_audio_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/consent_filter/with_audio/train_test/random_sampling_prediction.yaml --name random_sampling_prediction --results_base_dir ${datetime_str}_svr_consent_with_audio_train_test

# include socioeconomic status as demographic variable
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/with_socioeconomic/combined_audio_linguistic_shap_and_data.yaml --name combined_audio_linguistic_shap_and_data --results_base_dir ${datetime_str}_svr_reduced_feature_sets_with_socioeconomic
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/with_socioeconomic/demographics_only.yaml --name demographics_only --results_base_dir ${datetime_str}_svr_reduced_feature_sets_with_socioeconomic
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/with_socioeconomic/train_test/combined_audio_linguistic.yaml --name combined_audio_linguistic --results_base_dir ${datetime_str}_svr_reduced_feature_sets_with_socioeconomic_train_test
#python -u run_multiple.py --config configs/regression_composite_scores/svr_different_feature_sets/reduced_features/with_socioeconomic/train_test/demographics_only.yaml --name demographics_only --results_base_dir ${datetime_str}_svr_reduced_feature_sets_with_socioeconomic_train_test


python -u run.py --config configs/test.yaml --name test
