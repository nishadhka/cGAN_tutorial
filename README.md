# Overview of the cGAN code

There are three main parts to getting the cGAN up and running for regional post-processing of global Numerical Weather Prediction (NWP) forecasts. For simplicity these are modularised into sub-directories with individual instructions contained within. When training and running the cGAN, we recommend visiting and following instructions from the sub-directories in the following order:

1) [data](tensorflow-dev-test/data): Loading data and creating tfrecords for training.
2) [model](tensorflow-dev-test/model): Setting up the model architecture and training the model. 
3) [scripts](scripts): Generating forecasts.

Additionally, sub-directories [evaluation](tensorflow-dev-test/evaluation) and [config](tensorflow-dev-test/config) contain evaluation scripts and the necessary configuration files for setting data paths and model architecture. 

## Documentation

See [docs/](docs/README.md) for the documentation index — concept & rationale (cGAN training FAQ, East Africa variable-selection rationale, TF vs PyTorch comparison), planning, and workflow guides.

