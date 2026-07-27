from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
from typing import Any, Mapping


@dataclass
class Hyperparameters:
    batch_size: int = 32
    num_epochs: int = 100 # 100 starting to seem too short
    shuffle: bool = True
    dropout: float = 0.1
    patience: int = 10
    lr_patience: int = 3 # probably a good place for starting point before real tuning

@dataclass
class AutoencoderHyperparameters(Hyperparameters):
    learning_rate: float = 1e-4
    layer_type: str = "gru" # gru(+single cnn layer), transformer
    embedding_dim: int = 512
    cnn_out_channels: int = 256
    hidden_dim: int = 512
    latent_dim: int = 256
    kernel_size: int = 5
    num_layers: int = 3
    bidirectional: bool = True
    grad_clip: bool = True
    condition_decoder_on_latent: bool = True
    teacher_forcing_dropout_rate: float = 0.0
    use_decoder_positional_embeddings: bool = False
    max_decoder_positions: int = 1024
    max_encoder_positions: int = 1024
    num_heads: int = 8
    dim_feedforward: int = 2048
    scheduler_factor: float = 0.1
    use_cnn_before_transformer: bool = False
    weight_decay: float = 0.01  # for transformer


GRU_AUTOENCODER_SWEEP_SEARCH_SPACE = {
    "learning_rate": (1e-4, 3e-4),
    "num_layers": (2, 3),
    "hidden_dim": (512, 1024),
}

TRANSFORMER_AUTOENCODER_SWEEP_SEARCH_SPACE = {
    "learning_rate": (1e-4, 3e-4),
    "num_layers": (2, 3),
    "latent_dim": (128, 256),
    "dim_feedforward": (1024, 2048),
    # "teacher_forcing_dropout_rate": (0.0, 0.1),
    "use_cnn_before_transformer": (False, True),
}

AUTOENCODER_SWEEP_SEARCH_SPACES = {
    "gru": GRU_AUTOENCODER_SWEEP_SEARCH_SPACE,
    "transformer": TRANSFORMER_AUTOENCODER_SWEEP_SEARCH_SPACE,
}


def sweep_search_space_for_layer(layer_type: str) -> dict[str, tuple]:
    """Return the sweep search space for the selected autoencoder architecture."""
    try:
        return AUTOENCODER_SWEEP_SEARCH_SPACES[layer_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported layer_type for sweep: {layer_type}") from exc


def describe_sweep_run(
    hyperparams: AutoencoderHyperparameters,
    search_space: Mapping[str, tuple],
) -> str:
    """Format the current values for whichever fields are in the sweep."""
    return ", ".join(
        f"{name}={getattr(hyperparams, name)}"
        for name in search_space
    )


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


SWEEP_FIELD_ALIASES = {
    "latent_dims": "latent_dim",
    "teacher_forcing_dropout_rates": "teacher_forcing_dropout_rate",
    "learning_rates": "learning_rate",
    "lr_patiences": "lr_patience",
    "scheduler_factors": "scheduler_factor",
}

SWEEP_SUFFIX_LABELS = {
    "latent_dim": "latent",
    "teacher_forcing_dropout_rate": "tfd",
    "learning_rate": "lr",
    "lr_patience": "lrp",
    "scheduler_factor": "sf",
}


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, float):
        value_label = f"{value:g}"
    else:
        value_label = str(value)
    return value_label.replace(".", "p").replace("-", "m")


def _generic_sweep_suffix(hyperparameters: Mapping[str, Any]) -> str:
    parts = []
    for name, value in hyperparameters.items():
        label = SWEEP_SUFFIX_LABELS.get(name, name)
        parts.append(f"{label}{_format_sweep_value(value)}")
    return "_".join(parts)

@dataclass
class AutoencoderSweepConfig:
    def __init__(
        self,
        search_space: Mapping[str, tuple[Any, ...]] | None = None,
        **hyperparameter_options: tuple[Any, ...],
    ):
        if search_space is None:
            search_space = hyperparameter_options
        elif hyperparameter_options:
            raise ValueError(
                "Pass sweep values either with search_space or keyword arguments, not both"
            )

        valid_fields = {field.name for field in fields(AutoencoderHyperparameters)}
        normalized_search_space = {}
        for name, values in search_space.items():
            field_name = SWEEP_FIELD_ALIASES.get(name, name)
            if field_name not in valid_fields:
                raise ValueError(
                    f"Unknown autoencoder hyperparameter for sweep: {name}"
                )
            if not values:
                raise ValueError(
                    f"Sweep search space for {name} must contain at least one value"
                )
            normalized_search_space[field_name] = values

        self.search_space = normalized_search_space

    def iter_hyperparameters(
        self,
        base: AutoencoderHyperparameters,
    ) -> list[tuple[AutoencoderHyperparameters, str]]:
        runs = []
        names = tuple(self.search_space.keys())
        value_options = tuple(self.search_space[name] for name in names)

        for values in product(*value_options):
            swept_hyperparameters = dict(zip(names, values))
            hyperparams = replace(base, **swept_hyperparameters)
            suffix = _generic_sweep_suffix(swept_hyperparameters)
            runs.append((hyperparams, suffix))

        return runs
