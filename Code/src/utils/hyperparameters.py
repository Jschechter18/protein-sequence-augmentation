from dataclasses import dataclass, replace
from itertools import product


def autoencoder_sweep_suffix(
    latent_dim: int,
    teacher_forcing_dropout_rate: float,
) -> str:
    teacher_forcing_label = str(teacher_forcing_dropout_rate).replace(".", "p")
    return f"latent{latent_dim}_tfd{teacher_forcing_label}"


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
    latent_dims: tuple[int, ...] = (128, 256, 512)
    teacher_forcing_dropout_rates: tuple[float, ...] = (0.3, 0.45)

    def iter_hyperparameters(
        self,
        base: AutoencoderHyperparameters,
    ) -> list[tuple[AutoencoderHyperparameters, str]]:
        runs = []

        for latent_dim, teacher_forcing_dropout_rate in product(
            self.latent_dims,
            self.teacher_forcing_dropout_rates,
        ):
            hyperparams = replace(
                base,
                latent_dim=latent_dim,
                teacher_forcing_dropout_rate=teacher_forcing_dropout_rate,
            )

            suffix = autoencoder_sweep_suffix(
                latent_dim,
                teacher_forcing_dropout_rate,
            )

            runs.append((hyperparams, suffix))

        return runs
