"""
The datasetup.py script enables you to download and preprocess 
the Eiffel Tower Dataset 
"""

import argparse
import os
import sys
import tqdm
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from collections import namedtuple
from torchvision.datasets.utils import download_and_extract_archive


# Supported datasets
datasets = {"Eiffel-Tower": ["2015", "2016", "2018", "2020"]}

# Named tuple to store batch information
batch = namedtuple("batch", ["url", "dataset"])

# Dictionary to store batch information for each year datasets
batches = {
    "2015": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98240.zip",
        dataset="Eiffel-Tower",
    ),
    "2016": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98289.zip",
        dataset="Eiffel-Tower",
    ),
    "2018": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98314.zip",
        dataset="Eiffel-Tower",
    ),
    "2020": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98356.zip",
        dataset="Eiffel-Tower",
    ),
    "custom": batch(
        url=None,
        dataset="custom",
    ),
}


def setup_dataset(
    dataset_name: str,
    dataset_url: str = None,
    download_root: str = None,
    delete_archive: bool = True,
):
    """
    Downloads and extracts the dataset of interest

    Args:
        dataset_name: name of the dataset or batch of the dataset to
            download (for supported dataset like the Eiffel Tower, specify
            'Eiffel-Tower' to download data for all years or specify
            only the particular year/batch of interest. (e.g. 2015).
            To download from url, specify 'custom')
        dataset_url: url of the dataset. dataset_name must be 'custom'
        delete_archive: deletes archive after extraction if True.
            Keep otherwise.
    """

    # Get specific dataset batch to download from the list of supported batches
    dataset = batches[dataset_name]

    # Set the dataset url for custom data
    if dataset_name == "custom":
        if dataset_url is not None:
            dataset = dataset._replace(url=dataset_url)
        else:
            raise ValueError("custom dataset url cannot be None")

    # Set the download root dir name
    if download_root is None:
        download_root = dataset.dataset

    # Download and extract the archive
    if not delete_archive:
        download_and_extract_archive(dataset.url, download_root)
    else:
        download_and_extract_archive(dataset.url, download_root, remove_finished=True)

def run_command(command: str) -> Optional[str]:
    """
    Runs a command and returns the output.

    Args:
        command: command to run.
    """

    # Run command
    out = subprocess.run(command, capture_output=True, shell=True)

    # Display error if any
    if out.returncode != 0:
        print(f"Error running command: {command}")
        print(out.stderr.decode("utf-8"))
        sys.exit(1)

    # Return output if the is any
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


# Implement datasetup file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        choices=[
            "download",
        ],
        help="setup action to implement",
    )
    parser.add_argument(
        "--dataset",
        choices=list(datasets.keys()) + list(batches.keys()) + ["custom"],
        help="name or url of the dataset or single batch to setup",
        default="custom",
    )
    parser.add_argument(
        "--dataset_url",
        help="for custom data, specify url of the data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--download_root",
        help="name of the directory to download and extract archive to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--delete_archive",
        help="add flag to delete archive after extracting it",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--downscale_factor",
        help="downscale factor (number of times to downscale images)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--images_path",
        help="images directory path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--colmap_model_path",
        help="colmap model directory path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        help="output directory path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sfm_depths_path",
        help="depth maps path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_size",
        help="proportion of the dataset to include in the test split. \
            (must be a float in the range of 0 and 1)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--split_strategy",
        help="technique for splitting the data (one of interleaved or consecutive).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_split_from",
        help="where to start splitting the dataset. \
            (must be a float in the range of 0 and 1)",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    # Implement Module
    base_data_path = os.path.join(os.path.dirname(__file__))

    # 1. Download and extract data
    if args.action == "download":
        print("-" * 75)
        print(f"Downloading dataset")
        print("-" * 75)

        if args.dataset in datasets:
            for batch_name in datasets[args.dataset]:
                #  Define the images and model paths if not provided
                if args.images_path is None:
                    args.images_path = os.path.join(
                        base_data_path, args.dataset, batch_name, "images"
                    )
                if args.colmap_model_path is None:
                    args.colmap_model_path = os.path.join(
                        base_data_path, args.dataset, batch_name, "sfm"
                    )
                if args.output_path is None:
                    args.output_path = colmap_model_path = os.path.join(
                        base_data_path, args.dataset, batch_name
                    )

                # Run setup_dataset function for each data batch in dataset
                setup_dataset(
                    batch_name,
                    args.dataset_url,
                    args.download_root,
                    args.delete_archive,
                )

        else:
            batch_name = batches[args.dataset]
            #  Define the images and model paths if not provided
            if args.images_path is None:
                args.images_path = os.path.join(
                    base_data_path, batch_name.dataset, args.dataset, "images"
                )
            if args.colmap_model_path is None:
                args.colmap_model_path = os.path.join(
                    base_data_path, batch_name.dataset, args.dataset, "sfm"
                )
            if args.output_path is None:
                args.output_path = colmap_model_path = os.path.join(
                    base_data_path, batch_name.dataset, args.dataset
                )

            # Run setup_dataset function on single dataset
            setup_dataset(
                args.dataset, args.dataset_url, args.download_root, args.delete_archive
            )

        print("Done!!")
        
