from dataclasses import dataclass, replace
from itertools import product


def autoencoder_sweep_suffix(
    latent_dim: int,
    teacher_forcing_dropout_rate: float,
    scheduler_factor: float | None = None,
) -> str:
    teacher_forcing_label = str(teacher_forcing_dropout_rate).replace(".", "p")
    if scheduler_factor is None:
        return f"latent{latent_dim}_tfd{teacher_forcing_label}"
    scheduler_label = str(scheduler_factor).replace(".", "p")
    return f"latent{latent_dim}_tfd{teacher_forcing_label}_sf{scheduler_label}"


@dataclass
class Hyperparameters:
    batch_size: int = 32
    num_epochs: int = 100
    shuffle: bool = True
    dropout: float = 0.1
    patience: int = 10
    lr_patience: int = 3 # probably a good place for starting point before real tuning

@dataclass
class AutoencoderHyperparameters(Hyperparameters):
    learning_rate: float = 1e-3 # do NOT increase this, the highest it should be is 1e-3
    embedding_dim: int = 256
    cnn_out_channels: int = 256
    hidden_dim: int = 512
    latent_dim: int = 512
    kernel_size: int = 5
    num_layers: int = 2
    bidirectional: bool = True
    grad_clip: bool = True # needed when training with 
    condition_decoder_on_latent: bool = True
    teacher_forcing_dropout_rate: float = 0.3 # makes sure we don't totally rely on teacher forcing during training
    scheduler_factor: float = 0.1

@dataclass
class TransformerAutoencoderHyperparameters(Hyperparameters):
    learning_rate: float = 1e-3
    embedding_dim: int = 256
    hidden_dim: int = 256
    latent_dim: int = 128
    num_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    num_layers: int = 2

@dataclass
class AutoencoderSweepConfig:
    latent_dims: tuple[int, ...] = (256,)
    teacher_forcing_dropout_rates: tuple[float, ...] = (0.3, 0.45)
    learning_rates: tuple[float, ...] = (3e-4,)
    lr_patiences: tuple[int, ...] = (5,)
    scheduler_factors: tuple[float, ...] = (0.5,)

    def iter_hyperparameters(
        self,
        base: AutoencoderHyperparameters,
    ) -> list[tuple[AutoencoderHyperparameters, str]]:
        runs = []

        for latent_dim, teacher_forcing_dropout_rate, learning_rate, lr_patience, scheduler_factor in product(
            self.latent_dims,
            self.teacher_forcing_dropout_rates,
            self.learning_rates,
            self.lr_patiences,
            self.scheduler_factors,
        ):
            hyperparams = replace(
                base,
                latent_dim=latent_dim,
                teacher_forcing_dropout_rate=teacher_forcing_dropout_rate,
                learning_rate=learning_rate,
                lr_patience=lr_patience,
                scheduler_factor=scheduler_factor,
            )

            suffix = autoencoder_sweep_suffix(
                latent_dim,
                teacher_forcing_dropout_rate,
                scheduler_factor
            )

            runs.append((hyperparams, suffix))

        return runs
