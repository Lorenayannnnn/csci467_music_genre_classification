# CSCI467 Final Project: Music Genre Classification
I LOVE ROBIN'S BUBBLY PERSONALITY & HEARTWARMING CHUCKLES

Authors: Lorena Yan, Tianhao Wu, Ryan Wang

## Clone the repo & install requirements
- Clone github repo:
    ```
    git clone git@github.com:Lorenayannnnn/csci467_music_genre_classification.git
    cd csci467_music_genre_classification
    ```
- Create virtual environment on local machine:
    ```
    virtualenv -p $(which python3) ./venv
    source ./venv/bin/activate
    ```
    OR with conda if possible (GPU available preferred).
- Install requirements:
    ```
    pip3 install -r requirements.txt
    ```
*Please run this project on GPU if possible. 

## Dataset Preparation
- We use GTZAN from kaggle. Please click the following link [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) and download the dataset.
- Place data under the [data](/data) directory. The project should have the following structure:
```
    └── Baseline_Logistic_Regression
    └── CNN_*
    └── data
        ├── genres_original
        ├── images_original
        ├── features_3_sec.csv
        └── features_30_sec.csv
    └── Wav2Vec2ForGenreClassification
    ...
```

## Baseline
To reproduce the results, run the following under the "Baseline_Logistics_Regression" directory (assuming lr = 0.001, 500 epochs, and batch size of 32). Add the --test flag to evaluate trained model on test set
```
python ImageSoftmax.py  -r 0.001 -b 32 -T 500 --test
```

## CNN
- CNN_midterm and CNN_final folders contain training, testing, model, and some other files used for midterm and final report. 
- Go to CNN_final folder:
    ```
    cd CNN_final
    ```
- To reproduce the results, run (if CUDA is enabled)
    ```
    CUDA_VISIBLE_DEVICE=0 python train.py
    ```
- or (without CUDA)
    ```
    python train.py
    ```
- Assuming all necessary packages and dataset have been installed. Then run
    ```
    python test.py
    ```
    to test the model on the test set, where accuracy and confusion matrix are computed.

## Wav2Vec-based Model
- Go to Wav2Vec2ForGenreClassification folder:
  ```
  cd Wav2Vec2ForGenreClassification
  ```
- Run train:
  ```
  bash train.sh
  ```
  *Values of all hyperparameters and output directory are defined in this [train.sh](./Wav2Vec2ForGenreClassification/train.sh) script. Change the script for experiment purpose.
- Run evaluation:
  - Go to [eval.sh](./Wav2Vec2ForGenreClassification/eval.sh) script
  - Change ```model_name_or_path``` to the model checkpoint that you have saved.
  - Run ```bash eval.sh```