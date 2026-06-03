from dataclasses import dataclass

@dataclass
class AutoencoderHyperParameters:
    learning_rate: float = 1e-3
    batch_size: int = 32
    # num_epochs: int = 100
    num_epochs: int = 2
    shuffle: bool = True
    embedding_dim: int = 128
    hidden_dim: int = 256
    # hidden_dim: int = 512
    latent_dim: int = 128
    # latent_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    patience: int = 10
    max_len: int | None = None
