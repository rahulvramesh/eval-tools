import argparse
import os
import sys
from typing import List, Tuple, Union
from mmengine.config import Config

# Ensure the opencompass library is in the Python path
# Adjust this if your script is not in the root of the opencompass project
# Or if opencompass is not installed as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from opencompass.utils.run import get_config_from_arg
from opencompass.utils.build import build_dataset_from_cfg
from opencompass.utils.fileio import JSONToolkit
from opencompass.utils.logging import get_logger
from opencompass.utils.abbr import dataset_abbr_from_cfg

# Initialize a basic logger
logger = get_logger()

def find_dataset_config_from_experiment_cfg(experiment_cfg: Config, target_dataset_name: str) -> Union[Config, None]:
    """
    Searches for a specific dataset configuration within a larger experiment configuration.
    The target_dataset_name can be an abbreviation or a part of it.
    """
    if 'datasets' not in experiment_cfg:
        logger.error("Experiment configuration does not contain a 'datasets' key.")
        return None

    for dataset_cfg_list_item in experiment_cfg['datasets']:
        # experiment_cfg['datasets'] is usually a list of lists of dataset configs
        if not isinstance(dataset_cfg_list_item, list):
            logger.warning(f"Expected a list of dataset configs, but got {type(dataset_cfg_list_item)}. Skipping.")
            continue
        for actual_ds_cfg in dataset_cfg_list_item:
            try:
                current_abbr = dataset_abbr_from_cfg(actual_ds_cfg)
                # Check if the target_dataset_name is the abbreviation or part of it
                if target_dataset_name == current_abbr or \
                   target_dataset_name.startswith(current_abbr.split('_')[0]) or \
                   target_dataset_name in current_abbr: # More flexible matching
                    logger.info(f"Found matching dataset config for '{target_dataset_name}' with abbr '{current_abbr}'.")
                    return actual_ds_cfg
            except Exception as e:
                logger.warning(f"Could not get abbreviation for a dataset config: {e}. Config: {actual_ds_cfg}")
                continue
    logger.error(f"Dataset configuration for '{target_dataset_name}' not found in the loaded experiment config.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Load an OpenCompass dataset and save it to JSONL.")
    parser.add_argument("dataset_name",
                        type=str,
                        help="Name of the dataset to load (e.g., ceval_dev, mmlu_humanities). "
                             "This should match an abbreviation or a name discoverable by OpenCompass.")
    parser.add_argument("output_file",
                        type=str,
                        help="Path to save the output JSONL file.")
    parser.add_argument("--config-dir",
                        default="configs",
                        help="Directory to search for dataset configurations (relative to OpenCompass root or absolute). Default: 'configs'")
    parser.add_argument("--hf-path",
                        default=None, # Provide a dummy default if not used but expected by get_config_from_arg
                        help="Dummy hf-path argument if needed by internal config parsing.")
    parser.add_argument("--custom-dataset-path",
                        default=None,
                        help="Dummy custom-dataset-path argument.")
    parser.add_argument("--models",
                        action="append", # Allows multiple models, will be a list
                        default=None,
                        help="Dummy models argument.")


    cli_args = parser.parse_args()

    # Simulate the args object that get_config_from_arg expects
    # get_config_from_arg is designed for full experiment configs.
    # When args.config is None, it tries to build a config from args.models, args.datasets, etc.
    class ArgsNamespace:
        def __init__(self):
            self.config = None  # We are not loading a full experiment config file
            self.datasets = [cli_args.dataset_name] # This tells get_config_from_arg what to look for
            self.models = cli_args.models # Can be None, get_config_from_arg can handle this
            self.summarizer = None # Not needed for dataset loading
            self.config_dir = cli_args.config_dir
            # Add other attributes that get_config_from_arg might check, with default/None values
            self.hf_path = cli_args.hf_path
            self.custom_dataset_path = cli_args.custom_dataset_path
            self.hf_type = "base"
            self.model_kwargs = {}
            self.tokenizer_path = None
            self.tokenizer_kwargs = {}
            self.peft_path = None
            self.peft_kwargs = {}
            self.generation_kwargs = {}
            self.max_seq_len = None
            self.max_out_len = None
            self.min_out_len = None # Add this
            self.batch_size = None
            self.pad_token_id = None
            self.stop_words = []
            self.hf_num_gpus = 0 # Add this
            self.accelerator = None # Add this

    sim_args = ArgsNamespace()

    try:
        # get_config_from_arg will construct an experiment config.
        # The 'datasets' key in this config will hold a list of lists of dataset configs
        # that matched the names provided in sim_args.datasets.
        experiment_cfg = get_config_from_arg(sim_args)
        if not experiment_cfg.get('datasets'):
             raise ValueError(f"No dataset configurations were found or loaded for '{cli_args.dataset_name}'. "
                              "Please check the dataset name and --config-dir.")

        # Find the specific dataset config from the (potentially multiple) loaded ones
        dataset_cfg = find_dataset_config_from_experiment_cfg(experiment_cfg, cli_args.dataset_name)

        if dataset_cfg is None:
            raise ValueError(f"Could not find a specific dataset configuration matching '{cli_args.dataset_name}' "
                             "within the loaded configurations. Available datasets in loaded config: "
                             f"{[dataset_abbr_from_cfg(ds_list[0]) for ds_list in experiment_cfg.get('datasets', []) if ds_list]}")

    except Exception as e:
        logger.error(f"Error during configuration loading for dataset '{cli_args.dataset_name}': {e}")
        logger.error("Please ensure your dataset name is correct, discoverable by OpenCompass, "
                     "and any necessary dummy arguments are provided if the internal logic requires them.")
        sys.exit(1)

    logger.info(f"Using dataset configuration for: {dataset_abbr_from_cfg(dataset_cfg)}")
    dataset_obj = build_dataset_from_cfg(dataset_cfg)

    # Determine which split to save (prefer 'test', then 'validation', then first available)
    data_to_save = None
    if hasattr(dataset_obj, 'test') and dataset_obj.test is not None:
        data_to_save = dataset_obj.test
        logger.info("Using 'test' split.")
    elif isinstance(dataset_obj.dataset, dict) and 'test' in dataset_obj.dataset and dataset_obj.dataset['test'] is not None:
        data_to_save = dataset_obj.dataset['test']
        logger.info("Using 'test' split from dataset_obj.dataset.")
    elif hasattr(dataset_obj, 'val') and dataset_obj.val is not None: # some datasets use 'val'
        data_to_save = dataset_obj.val
        logger.info("Using 'val' split.")
    elif isinstance(dataset_obj.dataset, dict) and 'validation' in dataset_obj.dataset and dataset_obj.dataset['validation'] is not None:
        data_to_save = dataset_obj.dataset['validation']
        logger.info("Using 'validation' split from dataset_obj.dataset.")
    else:
        try:
            if isinstance(dataset_obj.dataset, dict) and dataset_obj.dataset:
                first_split_key = next(iter(dataset_obj.dataset.keys()))
                data_to_save = dataset_obj.dataset[first_split_key]
                logger.warning(f"Neither 'test' nor 'validation' split found. Using first available split: '{first_split_key}'.")
            elif hasattr(dataset_obj, 'dataset') and not isinstance(dataset_obj.dataset, dict) and dataset_obj.dataset is not None:
                # If dataset_obj.dataset is a HF Dataset directly
                data_to_save = dataset_obj.dataset
                logger.warning("Using dataset_obj.dataset directly as it's not a dict of splits.")
            else:
                 raise AttributeError("Dataset object does not have a 'test' or 'validation' split, and no other splits could be determined.")
        except Exception as e:
             logger.error(f"Error accessing dataset splits: {e}")
             sys.exit(1)

    if data_to_save is None:
        logger.error("Could not find any data to save. Please check the dataset structure.")
        sys.exit(1)

    # Convert Hugging Face Dataset to list of dicts
    try:
        list_of_dicts = [row for row in data_to_save]
    except Exception as e:
        logger.error(f"Error converting dataset split to list of dicts: {e}")
        sys.exit(1)

    if not list_of_dicts:
        logger.warning(f"The selected dataset split for '{cli_args.dataset_name}' is empty.")
    else:
        logger.info(f"Successfully converted dataset split to {len(list_of_dicts)} records.")


    # Ensure output directory exists
    output_dir = os.path.dirname(cli_args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")


    JSONToolkit.save_jsonl(list_of_dicts, cli_args.output_file)
    logger.info(f"Dataset '{cli_args.dataset_name}' loaded and saved {len(list_of_dicts)} records to '{cli_args.output_file}'")

if __name__ == "__main__":
    main()