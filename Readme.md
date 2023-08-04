-- EDITING IN PROGRESS --

# Using MediaPipe for Dance-Proficiency-Assessment


## Approach


### Machine Learning Process


## Jupyter Notebooks

For more detail on the approach see the Jupyter Notebooks

## Setup

### Install Libraries

`see setup.py`

```shell
pip install -r requirements.txt
```

## Making Predictions

* Example 1

No parameters meaning the script will take all the default parameters

```shell
python 03_pose_predictions.py 
```

* Example 2

## Collecting Pose Data

The script `01_pose_training_data.py` will capture frames from the webcam feed and collect the pose landmarks of interest.  With these values the script will save the x,y values to a CSV file with the specified label.

Once all of the poses are collected you can then use the csv datafile for machine learning

* Example 1

This command line will capture frames and store the landmarks.  It will collect frames for 20 seconds and will wait 10 seconds before starting to collect frames.  The data will be appended to a file called, 'pose_data.csv'

```shell
python 01_pose_training_data.py --class-name Hit --collect-for 20 --start-delay 10 --dry-run
```

## Train Models

The script `02_pose_model_training.py` will run through a number of scikit-learn models to determine which model performs the best and then will save that model to a pickle file.  This file is then used to make predictions

```shell
python 02_pose_model_training.py --file-name my_pose_training.csv --model-name my_pose_model
```


