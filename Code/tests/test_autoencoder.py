import torch
from models.autoencoder import ProteinSequenceAutoencoder as AE


def _make_model() -> AE:
    return AE(
        layer_type="gru",
        embedding_dim=64,
        cnn_out_channels=64,
        hidden_dim=128,
        latent_dim=64,
        kernel_size=3,
        num_layers=1,
        dropout=0.0,
        pad_idx=0,
        bos_idx=2,
    )


def test_autoencoder_forward_pass():
    model = _make_model()
    batch_size = 4
    sequence_length = 10
    vocab_size = model.vocab_size
    
    # Create a dummy input tensor with random integers in the range [0, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    # Run the forward pass
    logits = model(input_ids)
    
    # Check the output shape
    assert logits.shape == (batch_size, sequence_length, vocab_size), f"Expected output shape {(batch_size, sequence_length, vocab_size)}, but got {logits.shape}"
    
def test_autoencoder_encoder() -> None:
    model = _make_model()
    batch_size = 4
    sequence_length = 10
    vocab_size = model.vocab_size
    
    # Create a dummy input tensor with random integers in the range [0, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    # Run the encoder
    latent_vectors = model.encode(input_ids)
    
    # Check the output shape
    assert latent_vectors.shape == (batch_size, model.latent_dim), f"Expected latent vector shape {(batch_size, model.latent_dim)}, but got {latent_vectors.shape}"
    

def test_autoencoder_decoder() -> None:
    model = _make_model()
    batch_size = 4
    sequence_length = 10
    
    # Create a dummy latent vector tensor with random values
    latent_vectors = torch.randn(batch_size, model.latent_dim)
    
    # Run the decoder
    output_logits = model.decode(latent_vectors, sequence_length=sequence_length)
    
    # Check the output shape
    assert output_logits.shape == (batch_size, sequence_length, model.vocab_size), f"Expected output shape {(batch_size, sequence_length, model.vocab_size)}, but got {output_logits.shape}"

def test_autoencoder_decoder_can_disable_latent_conditioning() -> None:
    model = _make_model()
    legacy_model = AE(
        layer_type="gru",
        embedding_dim=64,
        cnn_out_channels=64,
        hidden_dim=128,
        latent_dim=64,
        kernel_size=3,
        num_layers=1,
        dropout=0.0,
        pad_idx=0,
        bos_idx=2,
        condition_decoder_on_latent=False,
    )

    assert model.decoder.input_size == model.embedding_dim + model.latent_dim
    assert legacy_model.decoder.input_size == legacy_model.embedding_dim

def test_autoencoder_reconstruct() -> None:
    model = _make_model()
    batch_size = 4
    sequence_length = 10
    vocab_size = model.vocab_size

    # Create a dummy input tensor with random integers in the range [0, vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    # Run the reconstruct method
    reconstructed_ids = model.reconstruct(input_ids)
    
    # Check the output shape
    assert reconstructed_ids.shape == (batch_size, sequence_length), f"Expected output shape {(batch_size, sequence_length)}, but got {reconstructed_ids.shape}"
    assert reconstructed_ids.dtype == torch.long
    assert reconstructed_ids.min() >= 0
    assert reconstructed_ids.max() < model.vocab_size
