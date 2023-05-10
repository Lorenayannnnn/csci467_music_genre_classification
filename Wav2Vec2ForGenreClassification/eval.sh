
CUDA_VISIBLE_DEVICES=1 python3 run_music_genre_classification.py \
  --outputs_dir ./outputs/ensemble_freeze_10_layer_model_biased_model-0.1/ \
  --do_eval \
  --data_dir ../data/genres_original/ \
  --data_split_txt_filepath ../data_split.txt \
  --model_name_or_path ./outputs/ensemble_freeze_10_layer_model_biased_model-0.1/model.ckpt \
  --process_last_hidden_state_method average \
  --normalize_audio_arr \
  --batch_size 32