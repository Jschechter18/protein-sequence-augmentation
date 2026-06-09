from dataclasses import dataclass

@dataclass
class Hyperparameters:
    batch_size: int = 32
    num_epochs: int = 100
    # num_epochs: int = 15
    shuffle: bool = True
    dropout: float = 0.1
    patience: int = 10
    lr_patience: int = 5
    # lr_patience: int = 3 # probably a good place for starting point before real tuning

@dataclass
class AutoencoderHyperparameters(Hyperparameters):
    learning_rate: float = 1e-3 # at 3e-3 it was way off
    # shuffle: bool = True
    embedding_dim: int = 256
    cnn_out_channels: int = 256
    hidden_dim: int = 512
    # latent_dim: int = 256
    latent_dim: int = 512
    kernel_size: int = 5
    num_layers: int = 2
    bidirectional: bool = True
    max_len: int | None = None

@dataclass
class TransformerAutoencoderHyperparameters(Hyperparameters):
    learning_rate: float = 1e-3
    embedding_dim: int = 256
    hidden_dim: int = 256
    latent_dim: int = 128
    num_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 128
    # max_len: int = 512
    # max_len: int = 1024 # picked a fairly large max_len to avoid truncation, but this can be adjusted based on the dataset
    num_layers: int = 2
