import argparse
import yaml
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from local_runtime import add_local_arguments, configure_local_runtime, ensure_indexes, export_bird_predictions

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers to use.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    parser.add_argument('--pick_final_sql', type=bool, default=False, help="Pick the final SQL from the generated SQLs.")
    add_local_arguments(parser)
    args = parser.parse_args()
    if args.resume:
        from resume import restore_arguments
        restore_arguments(args)
    configure_local_runtime(args, parser)

    args.run_start_time = datetime.now().isoformat()
    if not isinstance(args.config, dict):
        with open(args.config, 'r') as file:
            args.config=yaml.safe_load(file)
    
    return args

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()
    dataset = load_dataset(args.data_path)

    from runner.run_manager import RunManager

    run_manager = RunManager(args)
    if args.local_model:
        ensure_indexes(dataset, args.db_root_directory)
    run_manager.initialize_tasks(dataset)
    try:
        run_manager.run_tasks()
        run_manager.generate_sql_files()
    finally:
        if args.output_dir:
            export_bird_predictions(dataset, run_manager.result_directory)

if __name__ == '__main__':
    main()
