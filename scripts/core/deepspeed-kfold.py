import argparse
import torch
from SequenceDataset import SequenceDataset
from transformers import (
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
from HFDatasetWrapper import HFDatasetWrapper
from CRMTokenClassifier import CRMTokenClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import csv
import pandas as pd


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(np.float32)

    # Flatten for token-level accuracy
    preds_flat = preds.reshape(-1)
    labels_flat = labels.reshape(-1)
    acc = accuracy_score(labels_flat, preds_flat)
    return {"accuracy": acc}


def get_trainer(
        model_name,
        train_dataset,
        eval_dataset,
        output_dir="runs/default_run",
        learning_rate=1e-5,
        num_epochs=3,
    ):

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.TOKEN_CLS,
            target_modules=["attention.self.query", "attention.self.value"],
            modules_to_save=["classifier"] # Keep classifier unfrozen
        )

        model = CRMTokenClassifier(model_name)
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4, 
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            logging_strategy="steps",
            logging_steps=1000,
            report_to=["tensorboard"],
            logging_first_step=True,
        )

        # Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        return trainer

def main(args):

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = SequenceDataset(
        args.fasta,
        args.labels,
        args.model_name,
        2048,
    )

    hf_dataset = HFDatasetWrapper(dataset)

    # Create a subset of the data (using 100% of the data)
    total_indices = torch.randperm(len(dataset)).tolist()
    subset_size = int(len(hf_dataset) * 1.0)  # Use 100% of the data
    subset_indices = total_indices[:subset_size]
    subset = torch.utils.data.Subset(hf_dataset, subset_indices)

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=20, shuffle=False)
    all_metrics = []

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(subset)))):

        # Create train and validation subsets for this fold
        train_subset_fold = torch.utils.data.Subset(subset, train_indices.tolist())
        val_subset_fold = torch.utils.data.Subset(subset, val_indices.tolist())

        # Create trainer with fold-specific output directory
        trainer = get_trainer(
            args.model_name,
            train_subset_fold,
            val_subset_fold,
            output_dir=f"runs/fold_{fold_idx}",
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
        )

        # Train the model for this fold
        trainer.train()

        # Evaluate on validation set
        eval_metrics = trainer.evaluate()
        all_metrics.append(eval_metrics)

        print(f"Fold {fold_idx} metrics:", eval_metrics)

    # Calculate average metrics across all folds
    average_metrics = {}
    for key in all_metrics[0].keys():
        average_metrics[key] = np.mean([m[key] for m in all_metrics])

    print("\nAverage metrics across all folds:")
    for k, v in average_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script with 20-fold CV'
    )

    parser.add_argument('--fasta', type=str, required=True,
                        help='Path to the dataset (FASTA format).')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to the labels file.')
    parser.add_argument('--model_name', type=str,
                        default='InstaDeepAI/nucleotide-transformer-v2-250m-multi-species',
                        help='Pre-trained model name or path.')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum sequence length for the inputs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--run_name', type=str, default='default_run',
                        help='Name for the training run (used in logging).')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for DeepSpeed/DistributedDataParallel.")
    parser.add_argument("--deepspeed", type=str, default=None, 
                        help="Path to the DeepSpeed config file.")

    args = parser.parse_args()
    main(args)