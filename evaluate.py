import os
import argparse
import torch
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import evaluate
from model import EnhancedBartForSummarization
from tqdm import tqdm
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import pandas as pd

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# Download necessary NLTK data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enhanced_model_path", type=str, required=True, help="Path to the enhanced model")
    parser.add_argument("--bart_base_model_name", type=str, default="facebook/bart-base", 
                        help="HuggingFace model name for BART base")
    parser.add_argument("--t5_model_name", type=str, default="t5-base", 
                        help="HuggingFace model name for T5")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_output_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--human_eval_size", type=int, default=10, 
                        help="Number of samples for detailed human evaluation")
    parser.add_argument("--output_file", type=str, default="comparison_results_4.txt")
    parser.add_argument("--output_csv", type=str, default="metrics_comparison_4.csv")
    return parser.parse_args()

def generate_lead3_summaries(dataset):
    """Generate Lead-3 baseline summaries (first three sentences)"""
    lead3_summaries = []
    for article in dataset["article"]:
        sentences = sent_tokenize(article)
        summary = " ".join(sentences[:3]) if len(sentences) >= 3 else " ".join(sentences)
        lead3_summaries.append(summary)
    return lead3_summaries

def generate_bart_summaries(model, tokenizer, dataset, args, model_name="model"):
    """Generate summaries using a BART-based model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    generated_summaries = []
    
    print(f"Generating summaries for {model_name}...")
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i:i+args.batch_size]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch["article"],
            padding="max_length",
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summaries
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=args.max_output_length,
                num_beams=args.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode generated summaries
        decoded_summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_summaries.extend(decoded_summaries)
    
    return generated_summaries

def generate_t5_summaries(model, tokenizer, dataset, args):
    """Generate summaries using a T5 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    generated_summaries = []
    
    print("Generating summaries for T5-base...")
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch = dataset[i:i+args.batch_size]
        
        # T5 requires a prefix
        prefix = "summarize: "
        inputs = tokenizer(
            [prefix + article for article in batch["article"]],
            padding="max_length",
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summaries
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=args.max_output_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
        
        # Decode generated summaries
        decoded_summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_summaries.extend(decoded_summaries)
    
    return generated_summaries

def compute_rouge_metrics(generated_summaries, reference_summaries, model_name="model"):
    """Compute ROUGE scores"""
    print(f"Computing ROUGE metrics for {model_name}...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores for each prediction-reference pair
    rouge_scores = []
    for pred, ref in zip(generated_summaries, reference_summaries):
        score = scorer.score(ref, pred)
        rouge_scores.append(score)
    
    # Average the scores
    rouge_dict = {
        f"{model_name}_rouge1": np.mean([score['rouge1'].fmeasure for score in rouge_scores]) * 100,
        f"{model_name}_rouge2": np.mean([score['rouge2'].fmeasure for score in rouge_scores]) * 100,
        f"{model_name}_rougeL": np.mean([score['rougeL'].fmeasure for score in rouge_scores]) * 100,
    }
    
    return rouge_dict

def compute_bert_score(generated_summaries, reference_summaries, model_name="model"):
    """Compute BERTScore"""
    print(f"Computing BERTScore for {model_name}...")
    P, R, F1 = bert_score(generated_summaries, reference_summaries, lang="en", verbose=True)
    bert_score_result = {f"{model_name}_bertscore": F1.mean().item() * 100}
    return bert_score_result

def compute_repetition_scores(summaries, model_name="model"):
    """Compute repetition score for a list of summaries"""
    repetition_scores = []
    
    for summary in summaries:
        words = summary.lower().split()
        if not words:
            repetition_scores.append(0)
            continue
            
        unique_words = set(words)
        # Repetition score: higher means more repetition
        repetition_score = 1 - (len(unique_words) / len(words))
        repetition_scores.append(repetition_score)
    
    return {f"{model_name}_repetition": np.mean(repetition_scores) * 100}

def prepare_human_evaluation(dataset, reference_summaries, model_summaries_dict, args):
    """Prepare data for human evaluation"""
    # Randomly select samples for human evaluation
    indices = np.random.choice(len(dataset), min(args.human_eval_size, len(dataset)), replace=False)
    
    human_eval_data = []
    for idx in indices:
        sample = {
            "article": dataset[int(idx)]["article"],
            "reference": reference_summaries[int(idx)],
        }
        
        # Add summaries from each model
        for model_name, summaries in model_summaries_dict.items():
            sample[model_name] = summaries[int(idx)]
        
        human_eval_data.append(sample)
    
    # Save to file for human evaluation
    with open("human_evaluation_samples.txt", "w") as f:
        f.write("# Samples for Human Evaluation\n\n")
        for i, sample in enumerate(human_eval_data):
            f.write(f"## Sample {i+1}\n\n")
            f.write(f"### Article\n{sample['article'][:800]}...\n\n")
            f.write(f"### Reference Summary\n{sample['reference']}\n\n")
            
            for model_name, summary in sample.items():
                if model_name not in ["article", "reference"]:
                    f.write(f"### {model_name} Summary\n{summary}\n\n")
            
            f.write("### Human Ratings\n")
            f.write("For each summary, rate on a scale of 1-5:\n")
            f.write("1. Coherence: _____\n")
            f.write("2. Relevance: _____\n")
            f.write("3. Overall Quality: _____\n\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Human evaluation samples saved to human_evaluation_samples.txt")
    return human_eval_data

def main():
    args = parse_args()
    
    # Load enhanced model and tokenizer
    print("Loading enhanced model...")
    enhanced_tokenizer = BartTokenizer.from_pretrained(args.enhanced_model_path)
    enhanced_model = EnhancedBartForSummarization.from_pretrained(args.enhanced_model_path)
    
    # Load baseline BART model
    print("Loading BART-base model...")
    bart_tokenizer = BartTokenizer.from_pretrained(args.bart_base_model_name)
    bart_model = BartForConditionalGeneration.from_pretrained(args.bart_base_model_name)
    
    # Load T5 model
    print("Loading T5-base model...")
    t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model_name)
    
    # Load test dataset
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")["test"]
    
    # Limit dataset size if needed
    if args.sample_size > 0:
        dataset = dataset.select(range(min(len(dataset), args.sample_size)))
    
    # Reference summaries
    reference_summaries = dataset["highlights"]
    articles = dataset["article"]
    
    # Generate summaries with all models
    lead3_summaries = generate_lead3_summaries(dataset)
    bart_summaries = generate_bart_summaries(bart_model, bart_tokenizer, dataset, args, model_name="BART-base")
    t5_summaries = generate_t5_summaries(t5_model, t5_tokenizer, dataset, args)
    enhanced_summaries = generate_bart_summaries(enhanced_model, enhanced_tokenizer, dataset, args, model_name="Enhanced-BART")
    
    # Create dictionary of all model summaries
    all_summaries = {
        "Lead-3": lead3_summaries,
        "BART-base": bart_summaries,
        "T5-base": t5_summaries,
        "Enhanced-BART": enhanced_summaries
    }
    
    # Compute metrics for all models
    all_metrics = {}
    
    for model_name, summaries in all_summaries.items():
        # ROUGE metrics
        rouge_metrics = compute_rouge_metrics(summaries, reference_summaries, model_name)
        all_metrics.update(rouge_metrics)
        
        # BERTScore
        bert_metrics = compute_bert_score(summaries, reference_summaries, model_name)
        all_metrics.update(bert_metrics)
        
        # Repetition score
        repetition_metrics = compute_repetition_scores(summaries, model_name)
        all_metrics.update(repetition_metrics)
    
    # Prepare human evaluation
    human_eval_data = prepare_human_evaluation(dataset, reference_summaries, all_summaries, args)
    
    # Calculate improvements relative to BART-base
    improvements = {}
    baseline_metrics = {k: v for k, v in all_metrics.items() if k.startswith("BART-base")}
    enhanced_metrics = {k: v for k, v in all_metrics.items() if k.startswith("Enhanced-BART")}
    
    for baseline_key, baseline_value in baseline_metrics.items():
        metric_name = baseline_key.replace("BART-base_", "")
        enhanced_key = f"Enhanced-BART_{metric_name}"
        if enhanced_key in enhanced_metrics:
            # For repetition, lower is better
            if "repetition" in metric_name:
                improvements[f"improvement_{metric_name}"] = baseline_value - enhanced_metrics[enhanced_key]
            else:
                improvements[f"improvement_{metric_name}"] = enhanced_metrics[enhanced_key] - baseline_value
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for model_name in ["Lead-3", "BART-base", "T5-base", "Enhanced-BART"]:
        print(f"\n{model_name} Metrics:")
        for metric in ["rouge1", "rouge2", "rougeL", "bertscore", "repetition"]:
            key = f"{model_name}_{metric}"
            if key in all_metrics:
                print(f"  {metric}: {all_metrics[key]:.2f}")
    
    print("\nImprovements over BART-base:")
    for metric_name, value in improvements.items():
        print(f"  {metric_name}: {value:.2f} {'✓' if value > 0 else '✗'}")
    
    # Save results to file
    with open(args.output_file, "w") as f:
        f.write("Comprehensive Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name in ["Lead-3", "BART-base", "T5-base", "Enhanced-BART"]:
            f.write(f"\n{model_name} Metrics:\n")
            for metric in ["rouge1", "rouge2", "rougeL", "bertscore", "repetition"]:
                key = f"{model_name}_{metric}"
                if key in all_metrics:
                    f.write(f"  {metric}: {all_metrics[key]:.2f}\n")
        
        f.write("\nImprovements over BART-base:\n")
        for metric_name, value in improvements.items():
            f.write(f"  {metric_name}: {value:.2f} {'✓' if value > 0 else '✗'}\n")
        
        f.write("\nExample Summaries:\n")
        for i in range(min(5, len(enhanced_summaries))):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Article (truncated): {dataset[i]['article'][:300]}...\n")
            f.write(f"Reference: {reference_summaries[i]}\n")
            for model_name, summaries in all_summaries.items():
                f.write(f"{model_name}: {summaries[i]}\n")
            f.write("-" * 80 + "\n")
    
    # Save metrics to CSV for easier analysis and visualization
    metrics_rows = []
    
    # First, add all raw metrics
    for key, value in all_metrics.items():
        model_name, metric = key.split('_', 1)
        metrics_rows.append({
            'Model': model_name,
            'Metric': metric,
            'Value': value
        })
    
    # Then add improvements
    for key, value in improvements.items():
        metrics_rows.append({
            'Model': 'Improvement',
            'Metric': key.replace('improvement_', ''),
            'Value': value
        })
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_csv, index=False)
    
    print(f"\nResults saved to {args.output_file} and {args.output_csv}")
    print(f"Human evaluation samples saved to human_evaluation_samples.txt")

if __name__ == "__main__":
    main()