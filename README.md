# Code Usage Guide

1. **Download the Dataset**: Once you have downloaded the dataset, modify the `train.py` script by setting the `--train-data-path` and `--val-data-path` to the absolute path of the extracted dataset folder.
2. **Download Pre-trained Weights**: Obtain the pre-trained weights required for training.
3. **Set Pre-trained Weights Path**: In the `train.py` script, set the `--weights` parameter to the path where you have saved the pre-trained weights.
4. **Start Training**: With the dataset path `--data-path` and the pre-trained weights path `--weights` correctly set, you can now start training using the `train.py` script. During the training process, a `class_indices.json` file will be automatically generated.
5. **Use Your Own Dataset**: If you are using your own dataset, please arrange it according to the classification structure (i.e., one category corresponds to one folder, and within this, another folder for each package). Also, set the `num_classes` in both the training and prediction scripts to match the number of categories in your dataset.
