# Overview of the cGAN code

There are three main parts to getting the cGAN up and running for regional post-processing of global Numerical Weather Prediction (NWP) forecasts. For simplicity these are modularised into sub-directories with individual instructions contained within. When training and running the cGAN, we recommend visiting and following instructions from the sub-directories in the following order:

1) [data](https://github.com/snath-xoc/cGAN_tutorial/tree/main/data): Loading data and creating tfrecords for training.
2) [model](https://github.com/snath-xoc/cGAN_tutorial/tree/main/model): Setting up the model architecture and training the model. 
3) [scripts](https://github.com/snath-xoc/cGAN_tutorial/tree/main/scripts): Generating forecasts.

Additionally, sub-directories [evaluation](https://github.com/snath-xoc/cGAN_tutorial/tree/main/evaluation) and [config](https://github.com/snath-xoc/cGAN_tutorial/tree/main/config) contain evaluation scripts and the necessary configuration files for setting data paths and model architecture. 
