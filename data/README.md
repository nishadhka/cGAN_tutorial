# Data 

## Overview of the data loading workflow

The figure below provides an overview of the data loading workflow: 

![Screenshot 2025-09-19 at 15.31.43](https://hackmd.io/_uploads/SyJ_Y1iigg.png)


The key files within the directories in order of importance are:

1) [data.py](https://github.com/snath-xoc/cGAN_tutorial/blob/main/data/data.py) where:<br>
    a) Individual forecast variables are loaded under `load_fcst`, and all variables are loaded by calling `load_fcst_stack`.<br>
    b) constants (topography and land-sea mask) are loaded under `load_hires_constants`.<br>
    c) truth data is loaded under `load_truth_and_mask`.<br>
    d) Fcst and truth are loaded in for a given date and time in `load_fcst_truth_batch`.<br>
2) [data_generator.py](https://github.com/snath-xoc/cGAN_tutorial/blob/main/data/data_generator.py) which has a ```DataGenerator``` class that calls `load_fcst_truth_batch` at different dates and times through iterative `__getitem__` calls. This is an important function as it does not load in all data to memory at once but (in Nishadh's words) allows streaming.
3) [tfrecords_generator.py](https://github.com/snath-xoc/cGAN_tutorial/blob/main/data/tfrecords_generator.py) which:<br>
    a)  creates tfrecords by calling `write_data`.<br>
    b) during training it loads the batches from the tfrecords via the `create_mixed_dataset` that is called within its own `DataGenerator` function.
    


## Setting up data loading in your own machine

Before starting with anything, you need to check that all data paths are correctly set by going into the [config](https://github.com/snath-xoc/cGAN_tutorial/tree/main/config) sub-directory and adjusting the paths by:
1) Making sure the paths to read-in data (`TRUTH_PATH`, `FCST_PATH` and `CONSTANTS_PATH`) as well as write tfrecords to (`tfrecords_path`) in [data_paths.yaml](https://github.com/snath-xoc/cGAN_tutorial/blob/main/config/data_paths.yaml) are correct. 
2) Making sure that the correct `data_path` option is set in [local_config.yaml](https://github.com/snath-xoc/cGAN_tutorial/blob/main/config/local_config.yaml).

Creating the training data needed to train the cGAN, require the following steps:

1) Creating the forecast normalisation constants consisting of the min, max, mean and standard deviation of each variable considered. This is necessary as for fitting any AI model, values need to be normalised.
2) Setting up the data generator and visualising data to make sure it is loaded in correctly.
3) Creating the tfrecord files

An example notebook is provided under the example_notebooks directory called [create_tfrecords.ipynb](https://github.com/snath-xoc/cGAN_tutorial/blob/main/example_notebooks/create_tfrecords.ipynb) which allows us to follow these steps.
