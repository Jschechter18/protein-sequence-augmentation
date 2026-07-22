import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import torch

from Code.src.utils.dataloader import create_dataloader
from Code.src.models.classifier import ProteinSequenceClassifier
from Code.src.testing.classification_pipeline import ProteinClassificationTrainingPipeline, save_json
from Code.src.utils.utils import set_random_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: need to implement an experimental sweep
"""
Stage 1: Frozen representation evaluation
for task in ["solubility"]:  # add localization after pipeline validation
    for representation in [
        "direct_sequence_baseline",
        "random_ae_frozen",
        "trained_ae_frozen",
        "esm2_frozen",
        "trained_ae+esm2_frozen",
    ]:
        for head_type in ["linear", "mlp"]:
            for seed in [1, 2, 3, 4, 5]:
                tune hyperparameters on validation split
                train with early stopping
                evaluate selected model on clean test split
                record overall and sequence-length-stratified metrics

Stage 2: Fine-tuning ablations
for selected high-performing representations:
    compare:
        frozen encoder
        partially unfrozen encoder
        fully unfrozen encoder where computationally reasonable

Stage 3: Representation analysis
for each learned representation:
    evaluate:
        linear probe
        k-nearest-neighbor classification
        class separation
        performance by sequence length
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train Protein Sequence Classifier")
    parser.add_argument("--dataset", type=str, default="solubility", choices=["solubility", "localization"])
    parser.add_argument("--data_dir", type=str, default="data/processed/peer")
    parser.add_argument("--results_dir", type=str, default="Code/results/encoder_benchmark")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--embedding_type", type=str, default="esm2", choices=["esm2", "cnn", "autoencoder"])
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-3)
    parser.add_argument("--unfreeze_layers", type=int, default=0)
    parser.add_argument("--unfreeze_all_esm", action="store_true")
    parser.add_argument("--unfreeze_esm", action="store_true")
    parser.add_argument("--esm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--esm_model_name", type=str, default="esm2_t6_8M_UR50D")
    parser.add_argument("--cnn_embedding_dim", type=int, default=128)
    parser.add_argument("--cnn_num_filters", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--autoencoder_checkpoint", type=str, default=None) # default to v5 for now
    parser.add_argument("--autoencoder_embedding_dim", type=int, default=128)
    parser.add_argument("--autoencoder_cnn_channels", type=int, default=128)
    parser.add_argument("--autoencoder_hidden_dim", type=int, default=256)
    parser.add_argument("--autoencoder_latent_dim", type=int, default=128)
    parser.add_argument("--autoencoder_num_layers", type=int, default=1)
    parser.add_argument("--autoencoder_kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--evaluate_test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--run_experiment", action="store_true", default=False, help="Run the experiment with the specified parameters.")

    return parser.parse_args()


def create_run_dir(results_dir: str, dataset: str, args) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.embedding_type == "esm2":
        if not args.unfreeze_esm and not args.unfreeze_all_esm:
            stage = "stage_0_frozen"
        elif args.unfreeze_all_esm:
            stage = "stage_full"
        else:
            stage = f"stage{args.unfreeze_layers}_unfreeze_last{args.unfreeze_layers}"
    elif args.embedding_type == "autoencoder":
        stage = "frozen_pretrained"
    else:
        stage = "trained_from_scratch"

    run_dir = Path(results_dir) / dataset / args.embedding_type / stage / f"{args.embedding_type}_{dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    args = parse_args()
    set_random_seed(args.seed)

    run_dir = create_run_dir(args.results_dir, args.dataset, args)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == "localization" else 2

    data_encoding = "raw" if args.embedding_type == "esm2" else "char"

    # Save run configuration
    config = vars(args)
    config["run_dir"] = str(run_dir)
    config["device"] = device
    config["encoding"] = data_encoding
    save_json(config, run_dir / "config.json")

    # Data Loaders
    train_loader = create_dataloader(
        task=args.dataset, split="train", data_dir=args.data_dir,
        mode="classification", encoding=data_encoding, batch_size=args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        task=args.dataset, split="valid", data_dir=args.data_dir,
        mode="classification", encoding=data_encoding, batch_size=args.batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        task=args.dataset, split="test", data_dir=args.data_dir,
        mode="classification", encoding=data_encoding, batch_size=args.batch_size, shuffle=False
    ) if args.evaluate_test else None
    
    

    if args.run_experiment == False:
        # Model & Pipeline Initialization
        model = ProteinSequenceClassifier(
            embedding_type=args.embedding_type,
            num_classes=args.num_classes,
            esm_model_name=args.esm_model_name,
            autoencoder_checkpoint=args.autoencoder_checkpoint,
            autoencoder_embedding_dim=args.autoencoder_embedding_dim,
            autoencoder_cnn_channels=args.autoencoder_cnn_channels,
            autoencoder_hidden_dim=args.autoencoder_hidden_dim,
            autoencoder_latent_dim=args.autoencoder_latent_dim,
            autoencoder_num_layers=args.autoencoder_num_layers,
            autoencoder_kernel_size=args.autoencoder_kernel_size,
            device=device,
        )

        pipeline = ProteinClassificationTrainingPipeline(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            encoder_learning_rate=args.encoder_learning_rate,
            esm_learning_rate=args.esm_learning_rate,
            unfreeze_layers=args.unfreeze_layers,
            unfreeze_esm=args.unfreeze_esm,
            unfreeze_all_esm=args.unfreeze_all_esm,
            checkpoint_dir=run_dir / args.checkpoint_dir
        )
        
        # Execute training
        start_time = time.time()
        pipeline.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, early_stopping_patience=args.early_stopping_patience)
        
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        config["total_runtime_seconds"] = elapsed_time
        config["total_runtime_formatted"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        save_json(config, run_dir / "config.json")

        if args.evaluate_test and test_loader:
            pipeline.load_checkpoint()
            pipeline.evaluate_test(test_loader)
    else:
        # tasks = ["solubility", "localization"]
        rand_seeds = [42, 43, 44]
        representations = ["random_autoencoder", "trained_autoencoder", "esm2", "trained_autoencoder+esm2"]
        head_types = ["linear", "mlp", "cnn"]
        
        # for task in tasks:
        for seed in rand_seeds:
            for rep in representations:
                for head in head_types:
                    set_random_seed(seed)
                    # Model & Pipeline Initialization
                    model = ProteinSequenceClassifier(
                        embedding_type=rep,
                        num_classes=args.num_classes,
                        esm_model_name=args.esm_model_name,
                        autoencoder_checkpoint=args.autoencoder_checkpoint,
                        autoencoder_embedding_dim=args.autoencoder_embedding_dim,
                        autoencoder_cnn_channels=args.autoencoder_cnn_channels,
                        autoencoder_hidden_dim=args.autoencoder_hidden_dim,
                        autoencoder_latent_dim=args.autoencoder_latent_dim,
                        autoencoder_num_layers=args.autoencoder_num_layers,
                        autoencoder_kernel_size=args.autoencoder_kernel_size,
                        device=device,
                        head_type=head
                    )

                    pipeline = ProteinClassificationTrainingPipeline(
                        model=model,
                        device=device,
                        learning_rate=args.learning_rate,
                        encoder_learning_rate=args.encoder_learning_rate,
                        esm_learning_rate=args.esm_learning_rate,
                        unfreeze_layers=args.unfreeze_layers,
                        unfreeze_esm=args.unfreeze_esm,
                        unfreeze_all_esm=args.unfreeze_all_esm,
                        checkpoint_dir=run_dir / args.checkpoint_dir
                    )
                    
                    # Execute training
                    start_time = time.time()
                    pipeline.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, early_stopping_patience=args.early_stopping_patience)
                    
                    elapsed_time = time.time() - start_time
                    hours, rem = divmod(elapsed_time, 3600)
                    minutes, seconds = divmod(rem, 60)
                    config["total_runtime_seconds"] = elapsed_time
                    config["total_runtime_formatted"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    save_json(config, run_dir / "config.json")
        

    


if __name__ == "__main__":
    main()