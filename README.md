# csci467_music_genre_classification
I LOVE ROBIN'S BUBBLY PERSONALITY & HEARTWARMING CHUCKLES

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