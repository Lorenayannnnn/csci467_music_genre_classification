python3 run_music_genre_classification.py \
  --outputs_dir ./outputs/hubert_rate_16000_freeze_extractor/ \
  --do_eval \
  --do_test \
  --data_dir ../data/genres_original/ \
  --data_split_txt_filepath ../data_split.txt \
  --batch_size 32 \
  --num_epochs 20 \
  --val_every 10 \
  --sample_rate 16000 \
