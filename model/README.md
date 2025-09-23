# Model

## Model configuration

The cGAN implementation within this code base has several possible configurations which can be set in [config/config.yaml](https://github.com/snath-xoc/cGAN_tutorial/blob/main/config/config.yaml), with the options listed in subsections below. 

### GENERAL

Under the ```GENERAL``` section one can choose the option ```mode``` as:

- ```"det"```: [deterministic](https://github.com/snath-xoc/cGAN_tutorial/blob/main/model/deterministic.py) generator trained without the discriminator
- ```"GAN"```: implements a Wasserstein Conditional Generative Adversarial Network  with Gradient Penalty [(WGANGP)](https://github.com/snath-xoc/cGAN_tutorial/blob/main/model/gan.py).
- ```"VAEGAN"```: [Variational Auto-Encoder WGANGP](https://github.com/snath-xoc/cGAN_tutorial/blob/main/model/vaegantrain.py): Same as above but with an additional auto-encoder term before the cGAN.

As well as the option ```problem_type```:

- ```"normal"```: normal problem of post-processing/downscaling.
- ```"autocoarsen"```: coarsen high resolution forecast input before post-processing/downscaling.

### MODEL

There are different ```MODEL``` options from which the ```architecture``` can be set, with options:
- ```"normal"```: implements three residual blocks from which outputs are concatenated with constant features of topography and land-sea mask and passed through another three residual blocks.
- ```"forceconv"```: adds an initial pass through a 2-D convolutional layer ([see this article for a good explainer on convolutional arithmetic](https://arxiv.org/pdf/1603.07285)) with a 1x1 kernel before passing through the residual blocks.
- ```"forceconv-long"```: Adds an additional 3 residual blocks to the initial 3 residual blocks under ```"forceconv"```.

### GENERATOR

Settings for the generator that can be adjusted to improve cGAN training are:
- ```filters_gen```: generator network width
- ```noise_channels```: number of noise channels to have
- ```learning_rate_gen```: learning steps to take in gradient descent, if training blows up, decrease this.

Discriminator settings under the section```DISCRIMINATOR``` also enable adjustment of the ```filters_disc``` and ```learning_rate_disc```.

### TRAIN, VAL and EVAL

Where most importantly, one can set

- ```train_years``` and ```val_years``` used for training and validation respectively
- ```training_weights```: frequency to sample from each bin used in tfrecord creation
- ```num_samples```: total generator training samples
- ```steps_per_checkpoint```: number of batches per checkpoint save
- ```batch_size```: size of batches used during generator training
- ```ensemble_size```: size of ensemble for content loss; use null to turn off
- ```CL_type```:  type of content loss (additional loss on top of wasserstein loss see Harris et al. (2022)), options are: 
    - ```'CRPS'```: Continuous Ranked Probability Score
    - ```'CRPS_phys'```: CRPS using actual rainfall values by first applying the inverse log transformation
    - ```'ensmeanMSE'```: ensemble mean Mean Squared Error
    - ```'ensmeanMSE_phys'```: ensmeanMSE using actual rainfall values by first applying the inverse log transformation
- ```content_loss_weight```: weighting of content loss when adding it to wasserstein loss

## Default cGAN set up

By default we use the ```GAN``` with ```forceconv-long``` which has an architecture depicted below:

![Screenshot 2025-09-22 at 17.20.11](https://hackmd.io/_uploads/rJ5PDgJ2ex.png)

Where a residual block follows the architecture shown below:

![Screenshot 2025-09-22 at 23.44.04](https://hackmd.io/_uploads/SyuOWL1nxl.png)


Such that the ```filters_gen=128``` and ```filters_disc=512```.

## Step-by-step instructions on training the cGAN

For training and evaluating the cGAN we follow four simple steps:

1) Set up training and evaluation data by calling ```setup_data``` in [setupdata.py](https://github.com/snath-xoc/cGAN_tutorial/blob/main/setupdata.py) which:
    a) Calls ```setup_batch_gen``` to load in the tfrecords and sample them according to the specified ```training_weights```.
    b) Calls ```setup_full_image_dataset``` which loads in full images for ```val_years``` to evaluate the cGAN over.
2) Set up the model according to the configuration from [config/config.yaml](https://github.com/snath-xoc/cGAN_tutorial/blob/main/config/config.yaml) by calling ```setup_model``` from [setupmodel.py](https://github.com/snath-xoc/cGAN_tutorial/blob/main/setupmodel.py).
3) Starts training the model by calling ```train_model``` from [model/train.py](https://github.com/snath-xoc/cGAN_tutorial/blob/main/model/train.py).
4) Evaluates the model across the multiple saved checkpoints by calling ```evaluate_multiple_checkpoints``` from [evaluation/evaluation.py](https://github.com/snath-xoc/cGAN_tutorial/blob/main/evaluation/evaluation.py).

An example notebook is provided under the example_notebooks directory called [train_cgan.ipynb](https://github.com/snath-xoc/cGAN_tutorial/blob/main/example_notebooks/train_cgan.ipynb) which allows us to follow these steps for one training epoch. To fully train the cGAN it is better to follow the command line instructions: 

```
python main.py --config path/to/config_file.yaml
```

There are a number of options you can use at this point. These will 
evaluate your model after it has finished training:

- `--evaluate` to run checkpoint evaluation (CRPS, rank calculations, RMSE, RALSD, etc.)
- `--plot_ranks` will plot rank histograms (requires `--evaluate`)
	   
If you choose to run `--evaluate`, you must also specify if you want
to do this for all model checkpoints or just a selection. Do this using 

- `--eval_full`	  (all model checkpoints)
- `--eval_short`	  (recommended; the final 1/3rd of model checkpoints)
- `--eval_blitz`	  (the final 4 model checkpoints)

Two things to note:
- These three options work well with the 100 checkpoints that we 
have been working with. If this changes, you may want to update
them accordingly.
- Calculating everything, for all model iterations, will take a long 
time. Possibly weeks. You have been warned.

As an example, to train a model and evaluate the last few model
checkpoints, you could run:

```
python main.py --config path/to/config_file.yaml --evaluate --eval_blitz --plot_ranks
```

If you've already trained your model, and you just want to run some 
evaluation, use the --no_train flag, for example:

```
python main.py --config path/to/config_file.yaml --no_train --evaluate --eval_full
```

