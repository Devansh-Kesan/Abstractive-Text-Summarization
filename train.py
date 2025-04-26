import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BartTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from model import EnhancedBartForSummarization
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--use_coverage", action="store_true")
    parser.add_argument("--lambda_coverage", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()

def preprocess_batch(batch, tokenizer, args):
    # Tokenize inputs
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=args.max_input_length,
        return_tensors="pt"
    )
    
    # Tokenize outputs
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(
            batch["highlights"],
            padding="max_length",
            truncation=True,
            max_length=args.max_output_length,
            return_tensors="pt"
        )
    
    batch = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": outputs.input_ids.masked_fill(outputs.input_ids == tokenizer.pad_token_id, -100)
    }
    
    return batch

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    
    # Load dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Initialize model
    model = EnhancedBartForSummarization(
        model_name=args.model_name,
        use_coverage=args.use_coverage,
        lambda_coverage=args.lambda_coverage
    )
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set up mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        # Create DataLoader for training set
        train_dataset = dataset["train"].select(range(min(len(dataset["train"]), 50000)))  # Limit training set size if needed
        
        # Process batches
        for i in tqdm(range(0, len(train_dataset), args.batch_size), desc=f"Epoch {epoch+1}"):
            batch_data = train_dataset[i:i+args.batch_size]
            batch = preprocess_batch(batch_data, tokenizer, args)
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if args.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / args.accumulate_grad_batches
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (i // args.batch_size + 1) % args.accumulate_grad_batches == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                # Standard forward pass
                outputs = model(**batch)
                loss = outputs.loss / args.accumulate_grad_batches
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (i // args.batch_size + 1) % args.accumulate_grad_batches == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            epoch_loss += loss.item() * args.accumulate_grad_batches
            
            # Print progress
            if global_step % 50 == 0:
                print(f"Step {global_step}: Loss {loss.item()}")
        
        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.bart.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        print(f"Epoch {epoch+1} completed. Average loss: {epoch_loss / (len(train_dataset) // args.batch_size)}")
        
        # Validation (simplified)
        model.eval()
        val_dataset = dataset["validation"].select(range(min(len(dataset["validation"]), 1000)))
        val_loss = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(val_dataset), args.batch_size), desc="Validation"):
                batch_data = val_dataset[i:i+args.batch_size]
                batch = preprocess_batch(batch_data, tokenizer, args)
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        val_loss = val_loss / (len(val_dataset) // args.batch_size)
        print(f"Validation loss: {val_loss}")

if __name__ == "__main__":
    main()