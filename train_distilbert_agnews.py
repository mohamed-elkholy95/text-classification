#!/usr/bin/env python3
"""
Step 1: Fine-Tune DistilBERT on AG News for Text Classification
================================================================

This is the EASIEST starting point for learning fine-tuning.
DistilBERT (66M params) fits comfortably in 4GB VRAM with fp16.

What you'll learn:
  - Loading a real dataset from HuggingFace Hub
  - Tokenizing text for transformer input
  - Setting up HuggingFace Trainer with proper args
  - Evaluating with accuracy, F1, precision, recall
  - Saving and loading a fine-tuned model

Dataset: AG News — 120k news articles in 4 categories:
  0: World  |  1: Sports  |  2: Business  |  3: Sci/Tech

Hardware: RTX 3050 Ti (4GB VRAM) — uses ~1.5GB in fp16
Training time: ~15-20 minutes for 3 epochs

Usage:
  python train_distilbert_agnews.py                    # Full training
  python train_distilbert_agnews.py --dry-run           # Test everything loads
  python train_distilbert_agnews.py --epochs 1          # Quick 1-epoch test
  python train_distilbert_agnews.py --batch-size 32     # Larger batches if VRAM allows
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# AG News label names
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Where to save the fine-tuned model
OUTPUT_DIR = Path(__file__).parent / "models" / "distilbert-agnews"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on AG News")
    p.add_argument("--model-name", default="distilbert-base-uncased",
                    help="Base model from HuggingFace Hub (default: distilbert-base-uncased)")
    p.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs (default: 3)")
    p.add_argument("--batch-size", type=int, default=16,
                    help="Per-device batch size (default: 16, reduce to 8 if OOM)")
    p.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate (default: 2e-5)")
    p.add_argument("--max-length", type=int, default=128,
                    help="Max token sequence length (default: 128)")
    p.add_argument("--max-train-samples", type=int, default=None,
                    help="Limit training samples (for quick experiments)")
    p.add_argument("--max-eval-samples", type=int, default=None,
                    help="Limit eval samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                    help="Load everything but skip training (verify setup)")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics — called by Trainer at each evaluation step
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """Compute accuracy, F1, precision, recall from Trainer predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── 1. Check GPU ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    else:
        log.warning("No GPU found — training will be very slow on CPU!")

    # ── 2. Load dataset ───────────────────────────────────────────────────
    # AG News: 120k train, 7.6k test — 4 classes
    log.info("Loading AG News dataset from HuggingFace Hub...")
    dataset = load_dataset("ag_news")
    log.info("Train: %d samples | Test: %d samples", len(dataset["train"]), len(dataset["test"]))
    log.info("Classes: %s", LABEL_NAMES)

    # Optional: limit samples for quick experiments
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
        log.info("Limited training to %d samples", len(dataset["train"]))
    if args.max_eval_samples:
        dataset["test"] = dataset["test"].select(range(min(args.max_eval_samples, len(dataset["test"]))))

    # ── 3. Load tokenizer and tokenize ────────────────────────────────────
    log.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(examples):
        """Tokenize a batch of texts. This runs on the dataset in parallel."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            # We don't pad here — DataCollatorWithPadding does dynamic padding
            # per batch, which is more memory-efficient than padding everything
            # to max_length upfront.
        )

    log.info("Tokenizing dataset (max_length=%d)...", args.max_length)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # HuggingFace Trainer expects the label column to be named "labels"
    tokenized = tokenized.rename_column("label", "labels")

    # ── 4. Load model ─────────────────────────────────────────────────────
    log.info("Loading model: %s (num_labels=%d)", args.model_name, len(LABEL_NAMES))
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_NAMES),
        # Map label IDs to human-readable names (shows up in model config)
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %s total, %s trainable (%.1f%%)",
             f"{total:,}", f"{trainable:,}", 100 * trainable / total)

    if args.dry_run:
        log.info("DRY RUN — everything loaded successfully! Exiting.")
        log.info("Model: %s", args.model_name)
        log.info("Dataset: %d train / %d test", len(tokenized["train"]), len(tokenized["test"]))
        log.info("Run without --dry-run to start training.")
        return

    # ── 5. Training arguments ─────────────────────────────────────────────
    # These are tuned for RTX 3050 Ti (4GB VRAM)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # eval can use larger batch
        learning_rate=args.lr,
        weight_decay=0.01,               # L2 regularization
        warmup_ratio=0.1,                # 10% warmup steps
        lr_scheduler_type="cosine",      # cosine decay — better than linear for fine-tuning
        # Precision — fp16 halves VRAM usage
        fp16=torch.cuda.is_available(),
        # Evaluation
        eval_strategy="epoch",           # evaluate after each epoch
        save_strategy="epoch",           # save checkpoint after each epoch
        load_best_model_at_end=True,     # load the best checkpoint when training ends
        metric_for_best_model="f1",      # pick best model by F1 score
        greater_is_better=True,
        # Logging
        logging_steps=100,               # log every 100 steps
        report_to="none",               # disable wandb/tensorboard
        # Reproducibility
        seed=args.seed,
        # Memory optimization
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    # Dynamic padding — pads each batch to the longest sequence in that batch
    # instead of padding everything to max_length. Saves memory and speeds up training.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 6. Train ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    log.info("Starting training: %d epochs, batch_size=%d, lr=%s",
             args.epochs, args.batch_size, args.lr)
    log.info("Estimated VRAM usage: ~1.5 GB (fp16)")

    train_result = trainer.train()

    # ── 7. Evaluate ───────────────────────────────────────────────────────
    log.info("Evaluating on test set...")
    metrics = trainer.evaluate()
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            log.info("  %s: %.4f", k, v)

    # ── 8. Save ───────────────────────────────────────────────────────────
    save_path = Path(args.output_dir) / "final"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    log.info("Model saved to: %s", save_path)

    # ── 9. Quick inference test ───────────────────────────────────────────
    log.info("\n--- Quick Inference Test ---")
    test_texts = [
        "The stock market surged today as tech companies reported record earnings.",
        "The team won the championship after a thrilling overtime victory.",
        "Scientists discovered a new exoplanet orbiting a nearby star.",
        "World leaders met at the UN to discuss climate change policies.",
    ]
    model.eval()
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        pred = output.logits.argmax(dim=-1).item()
        conf = torch.softmax(output.logits, dim=-1).max().item()
        log.info("  [%s] (%.1f%%) %s", LABEL_NAMES[pred], conf * 100, text[:80])

    log.info("\nTraining complete! Model saved to: %s", save_path)
    log.info("Total training time: %.1f minutes", train_result.metrics["train_runtime"] / 60)


if __name__ == "__main__":
    main()
