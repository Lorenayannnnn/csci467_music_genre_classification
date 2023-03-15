CUDA_VISIBLE_DEVICES=0 python3 run_music_genre_classification.py \
  --outputs_dir ./outputs/wav2vec2-base-has-segments-freeze_feature_extractor/ \
  --do_train \
  --do_eval \
  --do_test \
  --data_dir ../data/genres_original/ \
  --data_split_txt_filepath ../data_split.txt \
  --model_name_or_path facebook/wav2vec2-base \
  --batch_size 32 \
  --num_epochs 100 \
  --val_every 1 \
  --freeze_part feature_extractor

#CUDA_VISIBLE_DEVICES=3 python3 run_music_genre_classification.py \
#  --outputs_dir ./outputs/hubert-base-ls960-18-segments/ \
#  --do_train \
#  --do_eval \
#  --do_test \
#  --data_dir ../data/genres_original/ \
#  --data_split_txt_filepath ../data_split.txt \
#  --feature_extractor_name facebook/hubert-base-ls960 \
#  --model_name_or_path ./outputs/hubert-base-ls960-18-segments/model.ckpt \
#  --batch_size 32 \
#  --num_epochs 20 \
#  --val_every 1

