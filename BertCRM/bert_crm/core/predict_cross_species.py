import argparse
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from Trainer import Trainer
from SequenceDataset import SequenceDataset
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
import deepspeed
from HFDatasetWrapper import HFDatasetWrapper
from CRMTokenClassifier import CRMTokenClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from copy import deepcopy
from transformers import TrainerCallback

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
        num_epochs=50,
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

    # Create a subset of the data (currently using 100% of the data)
    total_indices = list(range(len(dataset)))
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Shuffle indices
    total_indices = torch.randperm(len(dataset)).tolist()
    subset_size = int(len(hf_dataset) * 1.0)  # Use 100% of the data
    subset_indices = total_indices[:subset_size]
    subset = torch.utils.data.Subset(hf_dataset, subset_indices)

    # Calculate split sizes
    total_size = len(subset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_indices = subset_indices[:train_size]
    val_indices = subset_indices[train_size:train_size + val_size]
    test_indices = subset_indices[train_size + val_size:]

    train_subset = torch.utils.data.Subset(hf_dataset, train_indices)
    val_subset = torch.utils.data.Subset(hf_dataset, val_indices)
    test_subset = torch.utils.data.Subset(hf_dataset, test_indices)

    trainer = get_trainer(args.model_name, train_subset, val_subset)
    #trainer.add_callback(CustomCallback(trainer)) 
    trainer.train()
    predictions = trainer.predict(test_subset)
    print("Test metrics:", predictions.metrics)
    trainer.save_model("my_final_model")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training script'
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
