from .noise import NoiseGenerator
import sys
sys.path.insert(1,"../")
from evaluation import plot_sequences

import wandb
from wandb.integration.keras import WandbMetricsLogger


def train_model(
    *,
    model=None,
    mode=None,
    batch_gen_train=None,
    data_gen_valid=None,
    noise_channels=None,
    latent_variables=None,
    checkpoint=None,
    steps_per_checkpoint=None,
    num_cases=8,
    plot_fn=None,
    log_wandb=False
):
    for inputs, _ in batch_gen_train.take(1).as_numpy_iterator():
        cond = inputs["lo_res_inputs"]
        img_shape = cond.shape[1:-1]
    batch_size = cond.shape[0]
    del cond
    del inputs

    if mode == "GAN":
        noise_shape = (img_shape[0], img_shape[1], noise_channels)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(
            batch_gen_train, noise_gen, steps_per_checkpoint, training_ratio=2
        )

    elif mode == "VAEGAN":
        noise_shape = (img_shape[0], img_shape[1], latent_variables)
        noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)
        loss_log = model.train(
            batch_gen_train, noise_gen, steps_per_checkpoint, training_ratio=2
        )

    elif mode == "det":
        loss_log = model.train(batch_gen_train, steps_per_checkpoint)

    if log_wandb:
        wandb.login()
        wandb.log({'epochs': checkpoint,
                   'dic_loss': loss_log["disc_loss"],
                   'disc_loss_real':loss_log["disc_loss_real"],
                   'disc_loss_fake': loss_log["disc_loss_fake"],
                   'disc_loss_gp': loss_log["disc_loss_gp"],
                   'gen_loss_total': loss_log["gen_loss_total"],
                   'gen_loss_disc': loss_log["gen_loss_disc"],
                   'fen_loss_ct': loss_log["gen_loss_ct"]})

    plot_sequences(
        model.gen,
        mode,
        data_gen_valid,
        checkpoint,
        noise_channels=noise_channels,
        latent_variables=latent_variables,
        num_cases=num_cases,  # number of examples to plot
        ens_size=4,  # number of ensemble members to draw for each example
        out_fn=plot_fn,
    )

    return loss_log
