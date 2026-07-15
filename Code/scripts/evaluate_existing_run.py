import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from tqdm import tqdm 

from Code.src.models.esm2 import (ESM2Encoder, CNN1DClassifier, RNNClassifier)
from Code.src.utils.dataloader import create_dataloader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run dir", required=True)
    parser.add_argument("--cnn_checkpoint", required=True)
    parser.add_argument("--esm_checkpoint", default=None)
    parser.add_argument("--data_dir", default="data/processed/peer",)
    parser.add_argument("--batch_size", type=int, default=16,)

    return parser.parse_args()

def load_json(path:Path):
    with open(path, "r") as file:
        return json.load(file)
    
def build_classifier(classifier_head: str, num_classes: int,):    
    if classifier_head == "cnn":
        return CNN1DClassifier(
            input_dim= 320,
            num_classes= num_classes,
        )
    
    if classifier_head == "gru":
        return RNNClassifier(
            input_dim= 320,
            hidden_dim=128,
            num_classes=num_classes,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
            rnn_type="gru",
        )
    
    if classifier_head == "lstm":
        return RNNClassifier(
            input_dim=320,
            hidden_dim=128,
            num_classes=num_classes,
            num_layers=1,
            dropout_rate=0.3,
            bidirectional=True,
            rnn_type="lstm"
        )
    
    raise ValueError(f"Unsupported classifier head: {classifier_head}")

def evaluate(encoder,classifier, test_loader,device,num_classes):

    encoder.eval()
    classifier.eval()

    all_sequences = []
    all_true_labels = []
    all_predictions = []
    all_probabilities = []
    all_confidences = []
    all_sequence_lengths = []

    total_loss = 0.0
    criterion= torch.nn.CrossEntropyLoss()

    with torch.inference_mode():
        for batch in tqdm(
            test_loader,
            desc="Evaluating test set",
        ):
            sequences = batch["sequence"]
            labels = batch["label"].to(device).long()

            embeddings = encoder(sequences)
            logits = classifier(embeddings)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = probabilities.argmax(dim=1)
            confidences = probabilities.max(dim=1).values

            all_sequences.extend(sequences)
            all_sequence_lengths.extend(
                [len(sequence) for sequence in sequences]
            )
            all_true_labels.extend(labels.cpu().tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_confidences.extend(
                confidences.cpu().tolist()
            )
            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    labels_array = np.asarray(all_true_labels)
    predictions_array = np.asarray(all_predictions)
    probabilities_array = np.asarray(all_probabilities)

    metrics = {
        "test_loss": total_loss / len(test_loader),
        "test_accuracy": accuracy_score(
            labels_array,
            predictions_array,
        ),
        "test_f1": f1_score(
            labels_array,
            predictions_array,
            average="macro",
            zero_division=0,
        ),
        "test_precision": precision_score(
            labels_array,
            predictions_array,
            average="macro",
            zero_division=0,
        ),
        "test_recall": recall_score(
            labels_array,
            predictions_array,
            average="macro",
            zero_division=0,
        ),
        "num_test_samples": len(labels_array),
    }

    prediction_data = {
        "sample_index": np.arange(
            len(labels_array)
        ),
        "sequence": all_sequences,
        "sequence_length": all_sequence_lengths,
        "true_label": labels_array,
        "predicted_label": predictions_array,
        "confidence": all_confidences,
        "correct": (
            labels_array == predictions_array
        ),
    }

    for class_index in range(num_classes):
        prediction_data[
            f"probability_class_{class_index}"
        ] = probabilities_array[:, class_index]

    predictions_df = pd.DataFrame(
        prediction_data
    )

    return metrics, predictions_df

def main():
    args = parse_args()

    run_dir=Path(args.run_dir)
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"

    if config_path.exists():
        config = load_json(config_path)
    elif history_path.exists():
        config = load_json(history_path)["hyperparameters"]
    else:
        raise FileNotFoundError("run my have config.json and history.json")

    dataset = config["dataset"]
    classifier_head = config["classifier_head"]
    num_classes = int(config["num_classes"])
    esm_model_name = config.get(
        "esm_model_name",
        config.get(
            "esm_model",
            "esm2_t6_8M_UR50D",
        ),
    )

    device = torch.device("cude" if torch.cuda.is_available() else "cpu")    

    encoder = ESM2Encoder(
        model_name=esm_model_name
    ).to(device)

    classifier = build_classifier(
        classifier_head=classifier_head,
        num_classes=num_classes,
    ).to(device)

    classifier.load_state_dict(
        torch.load(
            args.cnn_checkpoint,
            map_location=device,
        )
    )

    if args.esm_checkpoint:
        esm_checkpoint_path = Path(
            args.esm_checkpoint
        )

        if not esm_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing ESM checkpoint: {esm_checkpoint_path}"
            )

        if encoder.model is None:
            raise RuntimeError(
                "ESM checkpoint provided but ESM package is not installed or ESM model could not be initialized."
            )

        encoder.model.load_state_dict(
            torch.load(
                esm_checkpoint_path,
                map_location=device,
            )
        )

    test_loader = create_dataloader(
    task=dataset,
    split="test",
    data_dir=args.data_dir,
    mode="classification",
    encoding="raw",
    batch_size=args.batch_size,
    shuffle=False,
    use_cache=False,
)
    metrics, predictions_df = evaluate(
        encoder=encoder,
        classifier=classifier,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
    )

    with open(
        run_dir / "test_metrics.json",
        "w",
    ) as file:
        json.dump(metrics, file, indent=4)

    predictions_df.to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )

    print(metrics)
    print(
        f"Saved outputs to: {run_dir}"
    )

if __name__ == "__main__":
    main()

#python -m Code.scripts.evaluate_existing_run \
#  --run_dir Code/results/esm2/solubility/cnn/stage_0_frozen/esm2_solubility_<timestamp> \
#  --cnn_checkpoint checkpoints/cnn/esm2_solubility_<timestamp>/best_cnn.pt 
# 
# python -m Code.scripts.evaluate_existing_run \
#  --run_dir Code/results/esm2/solubility/cnn/stage_full_unfreeze/esm2_solubility_<timestamp> \
#  --cnn_checkpoint checkpoints/cnn/esm2_solubility_<timestamp>/best_cnn.pt \
#  --esm_checkpoint checkpoints/esm2/esm2_solubility_<timestamp>/best_esm.pt  



    






