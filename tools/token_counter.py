import os
import json
import time
import traceback
import concurrent.futures
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import logging
import glob
from pathlib import Path
import yaml
import tabulate

# Import prompt_viewer internals
from tools.prompt_viewer import parse_args as pv_parse_args, Menu, build_model_from_cfg, build_dataset_from_cfg, ICL_PROMPT_TEMPLATES, ICL_RETRIEVERS
from opencompass.registry import ICL_PROMPT_TEMPLATES as _, ICL_RETRIEVERS as __
from opencompass.utils import match_files

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("token_counter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global tokenizer to be initialized once per process
tokenizer = None

def init_tokenizer(model_name):
    """Initialize the tokenizer (called once per process)."""
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info(f"Tokenizer initialized: {model_name}")


def count_tokens(text, model_name):
    """Count the number of tokens in a text using the tokenizer."""
    global tokenizer
    if tokenizer is None:
        init_tokenizer(model_name)
    return len(tokenizer.encode(text))


def estimate_prediction_tokens(dataset, dataset_type, model_name, sample_count=100):
    """
    Estimate the number of tokens in model predictions based on reference answers.
    
    Args:
        dataset: The dataset object
        dataset_type: Type or name of the dataset (for heuristics)
        model_name: Tokenizer model name
        sample_count: Number of samples to analyze for the estimate
    
    Returns:
        avg_tokens: Average tokens per prediction
        total_tokens: Total tokens for all predictions in the dataset
    """
    global tokenizer
    if tokenizer is None:
        init_tokenizer(model_name)
    
    # Get reference answers if available
    total_tokens = 0
    count = 0
    dataset_size = 0
    
    logger.info(f"Estimating prediction tokens for dataset type: {dataset_type}")
    
    # Try to determine dataset size
    try:
        if hasattr(dataset, 'data'):
            dataset_size = len(dataset.data)
            logger.info(f"Dataset size from dataset.data: {dataset_size}")
        elif hasattr(dataset, '__len__'):
            dataset_size = len(dataset)
            logger.info(f"Dataset size from len(dataset): {dataset_size}")
        else:
            logger.warning("Could not determine dataset size, using fallback")
            # Try to guess based on any attribute that might indicate size
            for attr in dir(dataset):
                if attr.lower() in ['size', 'length', 'count', 'num_samples']:
                    try:
                        size_attr = getattr(dataset, attr)
                        if isinstance(size_attr, (int, float)) and size_attr > 0:
                            dataset_size = int(size_attr)
                            logger.info(f"Found dataset size from attribute {attr}: {dataset_size}")
                            break
                    except:
                        pass
    except Exception as e:
        logger.warning(f"Error determining dataset size: {str(e)}")
    
    # Default to 100 if we couldn't determine size
    if dataset_size <= 0:
        dataset_size = 100
        logger.warning(f"Using default dataset size: {dataset_size}")
    
    # Debug dataset structure
    logger.info(f"Dataset type: {type(dataset).__name__}")
    logger.info(f"Dataset attributes: {[attr for attr in dir(dataset) if not attr.startswith('_')]}")
    
    try:
        # Try to access reference answers directly from the dataset
        if hasattr(dataset, 'data'):
            # Get output field name - varies by dataset structure
            output_field = None
            if hasattr(dataset, 'output_column'):
                output_field = dataset.output_column
                logger.info(f"Found output_column: {output_field}")
            elif hasattr(dataset, 'reader_cfg') and hasattr(dataset.reader_cfg, 'output_column'):
                output_field = dataset.reader_cfg.output_column
                logger.info(f"Found reader_cfg.output_column: {output_field}")
            
            # Default output fields to try if not specified
            potential_output_fields = ['answer', 'solution', 'output', 'target', 'response', 'completion']
            
            # Debug first sample to see its structure
            if len(dataset.data) > 0:
                logger.info(f"First sample keys: {list(dataset.data[0].keys()) if isinstance(dataset.data[0], dict) else 'Not a dict'}")
            
            # Sample answers from the dataset
            samples_to_check = min(sample_count, len(dataset.data))
            for i in range(samples_to_check):
                # Try to get answer from the specified output field
                answer = None
                if output_field and isinstance(dataset.data[i], dict) and output_field in dataset.data[i]:
                    answer = dataset.data[i][output_field]
                else:
                    # Try common output field names
                    if isinstance(dataset.data[i], dict):
                        for field in potential_output_fields:
                            if field in dataset.data[i]:
                                answer = dataset.data[i][field]
                                if i == 0:  # Just log for the first sample
                                    logger.info(f"Found answer in field: {field}")
                                break
                
                if answer:
                    tokens = len(tokenizer.encode(str(answer)))
                    total_tokens += tokens
                    count += 1
                    if i < 3:  # Log token counts for first few samples
                        logger.info(f"Sample {i} answer tokens: {tokens}")
        
        # If we got some token counts
        if count > 0:
            avg_tokens = total_tokens / count
            total_est_tokens = avg_tokens * dataset_size
            logger.info(f"Estimated prediction tokens based on {count} samples: {avg_tokens:.1f} tokens/sample")
            logger.info(f"Total estimated prediction tokens: {total_est_tokens}")
            return avg_tokens, total_est_tokens
    
    except Exception as e:
        logger.warning(f"Error estimating prediction tokens from references: {str(e)}")
        logger.warning(traceback.format_exc())
    
    # Fallback to heuristics based on dataset type
    logger.info("Falling back to heuristics for token estimation")
    dataset_type_lower = dataset_type.lower()
    
    # Default heuristics for different dataset types
    if any(x in dataset_type_lower for x in ['math', 'calculation']):
        avg_tokens = 200  # Math solutions tend to be longer
    elif any(x in dataset_type_lower for x in ['code', 'program']):
        avg_tokens = 300  # Code generation is typically longer
    elif any(x in dataset_type_lower for x in ['summarization', 'summary']):
        avg_tokens = 150  # Summaries can be moderate length
    elif any(x in dataset_type_lower for x in ['qa', 'question']):
        avg_tokens = 100  # QA responses vary but are typically moderate
    elif any(x in dataset_type_lower for x in ['choice', 'mcq', 'multiple']):
        avg_tokens = 30   # Multiple choice answers tend to be short
    else:
        avg_tokens = 120  # Default fallback estimate
    
    # Calculate total based on dataset size
    total_est_tokens = avg_tokens * dataset_size
    logger.info(f"Using heuristic for prediction tokens: {avg_tokens:.1f} tokens/sample based on dataset type")
    logger.info(f"Total estimated prediction tokens: {total_est_tokens} (heuristic)")
    
    return avg_tokens, total_est_tokens


def generate_prompts_direct(config_path, count=None, model_name="Qwen/Qwen-7B-Chat", workers=4):
    """
    Directly load the dataset config, build prompts, and return them as a single string.
    Count parameter is now optional - if None, process the entire dataset.
    Returns the combined prompt string, total sample count, and prediction token estimates.
    """
    # Load OpenCompass config
    from mmengine.config import Config
    cfg = Config.fromfile(config_path)

    # Parse model and dataset cfg dictionaries
    model2cfg = {k: v for k, v in getattr(cfg, 'models', {}).items()} or {'None': None}
    dataset2cfg = {}
    if 'datasets' in cfg:
        dataset2cfg = {k: v for k, v in getattr(cfg, 'datasets', {}).items()}
    else:
        for key in cfg.keys():
            if key.endswith('_datasets'):
                if isinstance(getattr(cfg, key), dict):
                    dataset2cfg.update({k: v for k, v in getattr(cfg, key).items()})
                elif isinstance(getattr(cfg, key), list):
                    for i, dataset in enumerate(getattr(cfg, key)):
                        name = getattr(dataset, 'name', f"{key}_{i}")
                        dataset2cfg[name] = dataset

    # Non-interactive selection: pick first entries
    model_abbr, model_cfg = next(iter(model2cfg.items()))
    dataset_abbr, dataset_cfg = next(iter(dataset2cfg.items()))

    # Build model and dataset
    model = build_model_from_cfg(model_cfg) if model_cfg else None
    infer_cfg = dataset_cfg.get('infer_cfg')
    dataset = build_dataset_from_cfg(dataset_cfg)
    
    # Get dataset name for better heuristics
    dataset_name = os.path.splitext(os.path.basename(config_path))[0]
    
    # Estimate prediction tokens
    avg_pred_tokens, total_pred_tokens = estimate_prediction_tokens(
        dataset, dataset_name, model_name, sample_count=min(100, len(dataset) if hasattr(dataset, '__len__') else 100)
    )

    # Build retriever and templates
    ice_template = ICL_PROMPT_TEMPLATES.build(infer_cfg['ice_template']) if infer_cfg.get('ice_template') else None
    prompt_template = ICL_PROMPT_TEMPLATES.build(infer_cfg['prompt_template']) if infer_cfg.get('prompt_template') else None
    retriever = ICL_RETRIEVERS.build({**infer_cfg['retriever'], 'dataset': dataset})

    # Retrieve ICE indices
    ice_idx_list = retriever.retrieve()
    
    # Get the total number of examples to process
    total_examples = len(ice_idx_list) if count is None else min(count, len(ice_idx_list))
    logger.info(f"Processing {total_examples} examples from the dataset")

    # Define a worker function for parallel processing
    def process_example(idx):
        try:
            ice_idx = ice_idx_list[idx]
            ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=infer_cfg['inferencer'].get('gen_field_replace_token', ''),
                ice_template=ice_template,
                prompt_template=prompt_template
            )
            
            # Optionally truncate if beyond model max_seq_len
            max_len = model_cfg.get('max_seq_len') if model_cfg else None
            if max_len is not None:
                token_len = model.get_token_len_from_template(prompt)
                while ice_idx and token_len > max_len:
                    ice_idx = ice_idx[:-1]
                    ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
                    prompt = retriever.generate_prompt_for_generate_task(
                        idx,
                        ice,
                        gen_field_replace_token=infer_cfg['inferencer'].get('gen_field_replace_token', ''),
                        ice_template=ice_template,
                        prompt_template=prompt_template
                    )
                    token_len = model.get_token_len_from_template(prompt)
                
                if idx % 100 == 0 or idx < 5:  # Log less frequently for large datasets
                    logger.info(f"Example {idx}: {len(ice_idx)} ICE entries; tokens={token_len}")
            
            # Parse into final string if model requires
            if model:
                prompt_str = model.parse_template(prompt, mode='gen')
                # If the result is still not a string, convert it
                if not isinstance(prompt_str, str):
                    # Try to convert PromptList to string
                    try:
                        prompt_str = str(prompt_str)
                    except:
                        if hasattr(prompt_str, 'to_string'):
                            prompt_str = prompt_str.to_string()
                        elif hasattr(prompt_str, 'text'):
                            prompt_str = prompt_str.text
                        else:
                            prompt_str = f"[Prompt {idx} - could not convert to string]"
                            logger.warning(f"Could not convert prompt {idx} to string")
            else:
                # If no model to parse, try direct string conversion
                prompt_str = str(prompt)
            
            return idx, prompt_str
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            return idx, f"[ERROR: {str(e)}]"

    # Process examples in parallel using ThreadPoolExecutor
    # Using threads instead of processes since this is I/O bound work
    prompts_text_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_example, idx) for idx in range(total_examples)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Generating prompts"):
            idx, prompt_str = future.result()
            prompts_text_dict[idx] = prompt_str
    
    # Sort the prompts by their original index
    prompts_text = [prompts_text_dict[idx] for idx in sorted(prompts_text_dict.keys())]
    
    logger.info(f"Generated {len(prompts_text)} prompts")
    
    # Join all prompts with separators and return both the prompt and the total sample count
    return "\n\n---\n\n".join(prompts_text), len(prompts_text), avg_pred_tokens, total_pred_tokens



def process_dataset(config_path, model_name, count=None, output_dir="token_counts"):
    """Process a single dataset config and save results with prediction token estimates."""
    try:
        logger.info(f"Processing dataset config: {config_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate dataset name from config path
        dataset_name = os.path.splitext(os.path.basename(config_path))[0]
        
        # Generate full prompt directly
        prompt, total_samples, avg_pred_tokens, total_pred_tokens = generate_prompts_direct(
            config_path, count, model_name
        )
        
        # Save raw prompt
        raw_file = Path(output_dir) / f"{dataset_name}_prompt.txt"
        raw_file.write_text(prompt, encoding='utf-8')
        logger.info(f"Prompt saved to {raw_file}")

        # Count prompt tokens
        logger.info(f"Counting prompt tokens for {dataset_name}...")
        prompt_tokens = count_tokens(prompt, model_name)
        logger.info(f"Total prompt tokens in {dataset_name}: {prompt_tokens}")
        
        # Total tokens (prompt + predictions)
        total_tokens = prompt_tokens + total_pred_tokens

        # Save summary
        summary = {
            "config": config_path,
            "model": model_name,
            "dataset": dataset_name,
            "prompt_length": len(prompt),
            "prompt_tokens": prompt_tokens,
            "total_samples": total_samples,
            "prompt_tokens_per_sample": prompt_tokens / total_samples if total_samples > 0 else 0,
            "estimated_avg_prediction_tokens": avg_pred_tokens,
            "estimated_total_prediction_tokens": total_pred_tokens,
            "estimated_total_tokens": total_tokens,  # Prompt + predictions
            "estimated_tokens_per_sample": total_tokens / total_samples if total_samples > 0 else 0
        }
        summary_file = Path(output_dir) / f"{dataset_name}_summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logger.info(f"Summary saved to {summary_file}")
        
        return summary
    except Exception as e:
        logger.error(f"Error processing {config_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "config": config_path,
            "dataset": os.path.splitext(os.path.basename(config_path))[0],
            "error": str(e),
            "status": "failed"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in prompts generated by prompt_viewer directly"
    )
    parser.add_argument("--config", help="Path to a specific dataset config (optional)")
    parser.add_argument("--pattern", nargs="*", default=["*_gen.py"], 
                        help="Patterns for matching dataset configs, e.g., '*_gen.py'")
    parser.add_argument("--model", default="Qwen/Qwen-7B-Chat", help="Tokenizer model name")
    parser.add_argument("--count", type=int, default=None, 
                        help="Number of prompts to generate per dataset (None for all)")
    parser.add_argument("--output", default="token_counts", 
                        help="Output directory for results")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Maximum number of worker processes (default: 1)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process a single config if specified
    if args.config:
        summary = process_dataset(args.config, args.model, args.count, args.output)
        print(f"Processed {args.config}")
        return

    # Otherwise, find all matching dataset configs
    dataset_configs = []
    for pattern in args.pattern:
        matched = match_files('opencompass/configs/datasets/', pattern, fuzzy=True)
        dataset_configs.extend([path for _, path in matched])
    
    if not dataset_configs:
        logger.error(f"No dataset configs found matching patterns: {args.pattern}")
        return
    
    logger.info(f"Found {len(dataset_configs)} dataset configs to process")
    
    # Display the list of configs to be processed
    table = [['#', 'Dataset Config']]
    for i, config in enumerate(dataset_configs):
        table.append([i+1, config])
    print(tabulate.tabulate(table, headers='firstrow', tablefmt='psql'))
    
    # Process all configs (in parallel if max_workers > 1)
    all_summaries = []
    
    if args.max_workers > 1:
        logger.info(f"Processing datasets in parallel with {args.max_workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_dataset, config, args.model, args.count, args.output)
                for config in dataset_configs
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing datasets"):
                summary = future.result()
                all_summaries.append(summary)
    else:
        logger.info("Processing datasets sequentially")
        for config in tqdm(dataset_configs, desc="Processing datasets"):
            summary = process_dataset(config, args.model, args.count, args.output)
            all_summaries.append(summary)
    
    # Create a combined summary
    combined_summary = {
        "model": args.model,
        "total_datasets": len(dataset_configs),
        "successful_datasets": sum(1 for s in all_summaries if "error" not in s),
        "failed_datasets": sum(1 for s in all_summaries if "error" in s),
        "total_samples": sum(s.get("total_samples", 0) for s in all_summaries if "total_samples" in s),
        "dataset_summaries": all_summaries
    }
    
    # Calculate total tokens
    combined_summary["total_prompt_tokens"] = sum(s.get("prompt_tokens", 0) for s in all_summaries if "prompt_tokens" in s)
    combined_summary["total_prediction_tokens"] = sum(s.get("estimated_total_prediction_tokens", 0) for s in all_summaries if "estimated_total_prediction_tokens" in s)
    combined_summary["grand_total_tokens"] = combined_summary["total_prompt_tokens"] + combined_summary["total_prediction_tokens"]
    
    # Calculate average tokens per sample
    if combined_summary["total_samples"] > 0:
        combined_summary["avg_prompt_tokens_per_sample"] = combined_summary["total_prompt_tokens"] / combined_summary["total_samples"]
        combined_summary["avg_prediction_tokens_per_sample"] = combined_summary["total_prediction_tokens"] / combined_summary["total_samples"]
        combined_summary["avg_total_tokens_per_sample"] = combined_summary["grand_total_tokens"] / combined_summary["total_samples"]
    
    # Save combined summary
    combined_file = Path(args.output) / "combined_summary.json"
    combined_file.write_text(json.dumps(combined_summary, indent=2), encoding='utf-8')
    logger.info(f"Combined summary saved to {combined_file}")
    
    # Create a summary table
    summary_table = [['Dataset', 'Samples', 'Prompt Tokens', 'Est. Pred Tokens', 'Est. Total Tokens', 'Status']]
    for summary in all_summaries:
        dataset_name = summary.get("dataset", "Unknown")
        if "error" in summary:
            status = "Failed"
            samples = "-"
            prompt_tokens = "-"
            pred_tokens = "-"
            total_tokens = "-"
        else:
            status = "Success"
            samples = summary.get("total_samples", 0)
            prompt_tokens = summary.get("prompt_tokens", 0)
            pred_tokens = summary.get("estimated_total_prediction_tokens", 0)
            total_tokens = summary.get("estimated_total_tokens", 0)
        
        summary_table.append([dataset_name, samples, prompt_tokens, pred_tokens, total_tokens, status])
    
    # Add totals row
    summary_table.append([
        "TOTAL",
        combined_summary["total_samples"],
        combined_summary["total_prompt_tokens"],
        combined_summary["total_prediction_tokens"],
        combined_summary["grand_total_tokens"],
        f"{combined_summary['successful_datasets']}/{combined_summary['total_datasets']} succeeded"
    ])
    
    # Print summary table
    print("\nSummary of Token Counts:")
    print(tabulate.tabulate(summary_table, headers='firstrow', tablefmt='psql'))
    
    # Print estimation of costs (using OpenAI API pricing as a reference)
    try:
        # Very rough cost estimates based on typical pricing
        # These are just approximations and will vary by provider
        print("\nEstimated API Costs (approximation):")
        cost_table = [['Model Type', 'Input Cost', 'Output Cost', 'Total Cost']]
        
        # GPT-3.5 Turbo (4K) rate: $0.0015/1K input, $0.002/1K output
        gpt35_input_cost = (combined_summary["total_prompt_tokens"] / 1000) * 0.0015
        gpt35_output_cost = (combined_summary["total_prediction_tokens"] / 1000) * 0.002
        gpt35_total = gpt35_input_cost + gpt35_output_cost
        
        # GPT-4 (8K) rate: $0.03/1K input, $0.06/1K output
        gpt4_input_cost = (combined_summary["total_prompt_tokens"] / 1000) * 0.03
        gpt4_output_cost = (combined_summary["total_prediction_tokens"] / 1000) * 0.06
        gpt4_total = gpt4_input_cost + gpt4_output_cost
        
        # Claude-3 Opus rate: $0.015/1K input, $0.075/1K output
        claude_input_cost = (combined_summary["total_prompt_tokens"] / 1000) * 0.015
        claude_output_cost = (combined_summary["total_prediction_tokens"] / 1000) * 0.075
        claude_total = claude_input_cost + claude_output_cost
        
        # Add to table
        cost_table.append(['GPT-3.5 Turbo', f"${gpt35_input_cost:.2f}", f"${gpt35_output_cost:.2f}", f"${gpt35_total:.2f}"])
        cost_table.append(['GPT-4', f"${gpt4_input_cost:.2f}", f"${gpt4_output_cost:.2f}", f"${gpt4_total:.2f}"])
        cost_table.append(['Claude-3 Opus', f"${claude_input_cost:.2f}", f"${claude_output_cost:.2f}", f"${claude_total:.2f}"])
        
        print(tabulate.tabulate(cost_table, headers='firstrow', tablefmt='psql'))
        print("Note: Cost estimates are approximate and based on typical API pricing.")
    except Exception as e:
        logger.error(f"Error calculating cost estimates: {str(e)}")


if __name__ == "__main__":
    main()