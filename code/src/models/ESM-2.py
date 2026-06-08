import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESM2Encoder(nn.Module):
    """ESM-2 protein sequence encoder"""
    
    def __init__(self, model_name: str = "facebook/esm2_t6_8M"):
        super().__init__()
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(model_name)
        except ImportError:
            logger.warning("ESM not installed. Using fallback embedding.")
            self.model = None
            self.alphabet = None
        
        if self.model is not None:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, sequences: list) -> torch.Tensor:
        """
        Encode sequences using ESM-2
        Args:
            sequences: List of protein sequences
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        if self.model is None:
            # Fallback: return random embeddings
            max_len = max(len(seq) for seq in sequences)
            return torch.randn(len(sequences), max_len, 320)
        
        with torch.no_grad():
            batch_converter = self.alphabet.get_batch_converter()
            batch_labels, batch_strs, batch_tokens = batch_converter(
                [(str(i), seq) for i, seq in enumerate(sequences)]
            )
            batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
            
            results = self.model(batch_tokens, repr_layers=[6])
            embeddings = results["representations"][6]
        
        return embeddings


class CNN1DClassifier(nn.Module):
    """1D-CNN classifier for protein sequences"""
    
    def __init__(
        self,
        input_dim: int = 320,
        num_classes: int = 2,
        kernel_sizes: list = None,
        num_filters: int = 64,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers for different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        
        # Pooling and fully connected layers
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected head
        fc_input_dim = num_filters * len(kernel_sizes)
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply convolutions with different kernel sizes
        conv_outputs = []
        for conv, bn in zip(self.conv_layers, self.bns):
            out = conv(x)
            out = bn(out)
            out = torch.relu(out)
            out = self.global_pool(out)
            out = out.squeeze(-1)
            conv_outputs.append(out)
        
        # Concatenate outputs from all convolution branches
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        
        # Fully connected layers
        logits = self.fc(x)
        return logits


class ESM2CNNPipeline:
    """Training pipeline combining ESM-2 encoder with 1D-CNN classifier"""
    
    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        device: str = None,
        esm_model_name: str = "facebook/esm2_t6_8M"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize encoder and classifier
        self.encoder = ESM2Encoder(model_name=esm_model_name).to(self.device)
        self.classifier = CNN1DClassifier(
            input_dim=320,
            num_classes=num_classes
        ).to(self.device)
        
        # Optimizer (only for classifier since encoder is frozen)
        self.optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for sequences, labels in progress_bar:
            labels = labels.to(self.device)
            
            # Encode sequences
            with torch.no_grad():
                embeddings = self.encoder(sequences)
            embeddings = embeddings.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.classifier(embeddings)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': total_loss / (total / labels.size(0)),
                'acc': correct / total
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in tqdm(val_loader, desc="Validation"):
                labels = labels.to(self.device)
                
                # Encode sequences
                embeddings = self.encoder(sequences)
                embeddings = embeddings.to(self.device)
                
                # Forward pass
                logits = self.classifier(embeddings)
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = 5
    ) -> Dict[str, list]:
        """
        Train the model
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return history
    
    def save_checkpoint(self, filepath: str = "best_model.pt"):
        """Save model checkpoint"""
        torch.save(self.classifier.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_checkpoint(self, filepath: str = "best_model.pt"):
        """Load model checkpoint"""
        self.classifier.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"Model loaded from {filepath}")
    
    def predict(self, sequences: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on new sequences
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            predictions (class labels) and probabilities
        """
        self.classifier.eval()
        
        with torch.no_grad():
            embeddings = self.encoder(sequences)
            embeddings = embeddings.to(self.device)
            logits = self.classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = probs.cpu().numpy()
        
        return predictions, probabilities


if __name__ == "__main__":
    # Example usage
    logger.info("ESM-2 + 1D-CNN Classification Pipeline")
    logger.info(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
