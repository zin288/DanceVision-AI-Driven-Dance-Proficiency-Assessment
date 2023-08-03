# Using MediaPipe for Dance-Proficiency-Assessment

```shell
git clone https://github.com/youngsoul/mediapipe-ymca.git
cd mediapipe-ymca
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python 03_pose_predictions.py
```

## Approach


### Machine Learning Process

Below is a picture of the general flow of the Machine Learning Process


## Jupyter Notebooks

For more detail on the approach see the Jupyter Notebooks

## Setup

### Install Libraries

`see setup.py`

```shell
pip install -r requirements.txt
```

## Making Predictions

It is always more fun to have the music to dance to.  In the `media` directory I have a short clip of the YMCA song to you dance to.

* Example 1

No parameters meaning the script will take all the default parameters

```shell
python 03_pose_predictions.py 
```

* Example 2

## Collecting Pose Data

The script `01_pose_training_data.py` will capture frames from the webcam feed and collect the pose landmarks of interest.  With these values the script will save the x,y,z values to a CSV file with the specified label.

Once all of the poses are collected you can then use the csv datafile for machine learning

* Example 1

This command line will capture frames and store the landmarks with the 。。。.  It will collect frames for 20 seconds and will wait 10 seconds before starting to collect frames.  The data will be appended to a file called, 'my_pose_data.csv'
```shell
python 01_pose_training_data.py --class-name Y --collect-for 20 --start-delay 10 --file-name my_pose_data.csv
```

* Example 2

This command is similar to the one above but with the addition of --dry-run which will prevent the data from being written to the data file.
```shell
python 01_pose_training_data.py --class-name Y --collect-for 20 --start-delay 10 --dry-run
```

## Train Models

The script `02_pose_model_training.py` will run through a number of scikit-learn models to determine which model performs the best and then will save that model to a pickle file.  This file is then used to make predictions

* Example 1

with no parameters this script will look for a file named, `` and save the model to the name `best_ymca_pose_model.pkl`

```shell
python 02_pose_model_training.py 
```

* Example 2

Parameters allow the defaults to be changed.

```shell
python 02_pose_model_training.py --file-name my_pose_training.csv --model-name my_pose_model
```


