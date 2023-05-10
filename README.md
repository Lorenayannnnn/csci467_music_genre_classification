# csci467_music_genre_classification
I LOVE ROBIN'S BUBBLY PERSONALITY & HEARTWARMING CHUCKLES

# dataset
We use GTZAN from kaggle. Please click the following link.
[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

# Baseline

To install packages, run the following code in the parent directory (csci467_music_genre_classification)

```
pip install -r requirements.txt
```

To reproduce the results, run the following under the "Baseline_Logistics_Regression" directory (assuming lr = 0.001, 500 epochs, and batch size of 32). Add the --test flag to evaluate trained model on test set

```
python ImageSoftmax.py  -r 0.001 -b 32 -T 500 --test
```

# CNN

CNN_midterm and CNN_final folders contain training, testing, model, and some other files used for midterm and final report.

To install packages, go to CNN_final folder and run
```
pip install -r requirements.txt
```
To reproduce the results, run (if CUDA is enabled)
```
CUDA_VISIBLE_DEVICE=0 python train.py
```
or (without CUDA)
```
python train.py
```
assuming all necessary packages and dataset have been installed.

Then run
```
python test.py
```
to test the model on the test set, where accuracy and confusion matrix are computed.