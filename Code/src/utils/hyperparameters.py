from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # learning_rate: float = 1e-3
    learning_rate: float = 3e-3
    batch_size: int = 32
    num_epochs: int = 100
    # num_epochs: int = 15
    shuffle: bool = True
    num_layers: int = 2
    dropout: float = 0.1
    patience: int = 10
    lr_patience: int = 5

@dataclass
class AutoencoderHyperparameters(Hyperparameters):
    num_epochs: int = 15
    shuffle: bool = True
    embedding_dim: int = 128
    hidden_dim: int = 512
    latent_dim: int = 512
    # max_len: int | None = 128
    max_len: int | None = None

@dataclass
class TransformerAutoencoderHyperparameters(Hyperparameters):
    embedding_dim: int = 256
    hidden_dim: int = 256
    latent_dim: int = 512
    num_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # max_len: int = 512
    max_len: int = 1024
