# Automatic_vectorization_Lausanne
semester project repository of EPFL in DHLAB.

## reposiory Menu

training: folder of semantic segmentation (chapter 3 of the report)
polygon_recovery: folder of polygon recovery (chapter 4 of the report)
report: folder to save reports and LaTeX sources.

## Semantic segmentation
Put images and labels in the folder "training/data0/images", "training/data0/labels", respectively.

1. Run generate_random_dataset.py. It will make patches from images in "training/data0" and save patched dataset in new-created folder "training/data".

2. Run train.py. It will first randomly split dataset to training set, validation set, test set, and then execute the training process with patched dataset, and save trained model in new-created folder
"models"
3. Run predict.py, which predict results of test set.

4. Prediction reconstitution: run "polygon_recovery/patch_and_reconstitute_map.py", it will out whole map predictions from original map. You can add some additional maps without annotation to the folder "additional/images", and run this code to get map inference results.

## Polygon Recovery
The code is located at "polygon_recovery/polygon_recovery.ipynb".

