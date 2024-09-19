import sys
import logging
import datetime
import argparse

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def setup_logging(prefix: str):
    """ Sets up logging for the program.

    :param prefix: A prefix for the log file.
    :return:
    """
    log_name = f"{prefix}.log"
    # Configure logging to write to a file and stderr.
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename=log_name),
            logging.StreamHandler(sys.stderr)
        ],
        format="%(asctime)s [%(levelname)s] (%(filename)s) %(message)s"
    )
    # Capture warnings in the log file.
    logging.captureWarnings(True)
    return log_name


def parse_args():
    """ Parses arguments from the command line and sets up logging for the script.
    :return: None
    """
    # Get start time for the script.
    start_time = datetime.datetime.now()

    # A condensed date string for use in output files.
    start_time_str = start_time.strftime('%Y%m%d_%H%M%S')

    # Create argument parser
    parser = argparse.ArgumentParser()

    # Require the either VCF or TRAW input files.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--vcf", required=False, help="Path to a VCF file to read.")
    input_group.add_argument("--traw", required=False, help="Path to a .traw file to read.")

    # Other options
    parser.add_argument("--output_prefix", help="Path/prefix for the output files.")
    parser.add_argument("--max_freq", type=float, required=True, default=0.1, help="Variants with MAF > max_freq will be excluded from the RAS calculation.")
    parser.add_argument("--gens", type=int, required=True, default=30, help="Number of bootstrap iterations to perform")
    parser.add_argument("--num_vars", type=int, required=True, default=500, help="At each bootstrap iteration, sample until we have similarity scores for n_variants.")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--optimized", action="store_true", help="If this flag is passed, use a faster RAS computation which is numerically identical to the original implementation.")
    parser.add_argument("--no_sample", action="store_true", help="If this flag is passed, calculate RAS across all variants which pass filtering without random sampling. No effect unless using optimized (--optimized) algorithm,")
    parser.add_argument("--matrix_mem_limit", type=float, help="When using --optimized, this parameter controls the memory limit in bytes imposed for each matrix block computation. Total program memory consumption will exceed this limit. Default is 4GB" )
    parsed_args = parser.parse_args()

    # If output prefix is None, use input filename as output prefix.
    if parsed_args.output_prefix is None:
        input_filename = parsed_args.traw or parsed_args.vcf
        parsed_args.output_prefix = f"{input_filename.rsplit('.', maxsplit=1)[0]}"
    # Append a datetime string to the prefix to avoid overwriting results.
    parsed_args.output_prefix += f"_{start_time_str}"

    # Setup logging
    log_name = setup_logging(prefix=parsed_args.output_prefix)

    # Log the script name.
    log.info("RAS Python Script v1.0")
    log.info(f"Current Time: {start_time.strftime('%c')}")
    log.info(f"Logging to: {log_name}")
    # Print program configuration to log.
    parsed_args_d = parsed_args.__dict__
    max_key_len = max([len(k) for k in parsed_args_d.keys()])
    max_val_len = max([len(str(v)) for v in parsed_args_d.values()])

    header_len = max_key_len + max_val_len + 2

    log.info("Args".center(header_len, "="))
    for arg_key, arg_val in parsed_args_d.items():
        # Print a nicely formatted string for the memory limit.
        if arg_key == "matrix_mem_limit" and arg_val is not None:
            log.info(f"{arg_key.rjust(max_key_len)}: {memory_str(arg_val)}")
        else:
            log.info(f"{arg_key.rjust(max_key_len)}: {arg_val}")

    log.info("="*header_len)

    return parsed_args


def write_results(output_prefix: str, ras_matrix: pd.DataFrame, ras_pairs: pd.DataFrame = None):
    """ Takes results from RAS computation and writes to file(s).

    :param output_prefix: The output prefix which will be the base of the output filenames.
    :param ras_matrix: DataFrame (n_samples, n_samples) of RAS values.
    :param ras_pairs:  DataFrame (n_samples^2, num_vars) containing multiple RAS estimates
                       for each sample pair. These estimates are averaged to create `ras_matrix`.
                       This DataFrame may be None if the --no_sample flag was set.
    :return: Writes two files '*_rasPairs.csv' and '*_rasMatrix.csv'
    """

    # Write RAS pairs (if computed)
    if ras_pairs is not None:
        ras_pairs_filename = f"{output_prefix}_rasPairs.csv"
        ras_pairs.to_csv(ras_pairs_filename)
        log.info(f"Wrote RAS pairs to: '{ras_pairs_filename}'")

    # Write RAS matrix
    ras_matrix_filename = f"{output_prefix}_rasMatrix.csv"
    ras_matrix.to_csv(ras_matrix_filename)
    log.info(f"Wrote RAS matrix to: '{ras_matrix_filename}'")


def progress_bar(width=80, progress: float = None, n_of_n: tuple[int, int] = None, per_sec: float = None):
    """ Generate a simple progress bar showing the progress of some operation.

    :param progress: Fractional progress [0.0, 1.0] of the operation.
    :param width: Width in terminal characters of the bar. defaults to 80.
    :param n_of_n: Optional tuple containing fraction of tasks finished and total tasks
                   (n_finished, n_total). If provided, the bar will display this information.
    :param per_sec: Optional number of operations which occur every second. If provided, print how many iterations/tasks
                    are completed each second.

    :return: A string representation of a progress bar.
    """
    if progress is None and n_of_n is None:
        raise RuntimeError("Must pass one of 'progress' or 'n_of_n'!")

    if n_of_n is not None:
        n_complete, n_total = n_of_n
        progress = n_complete/n_total if progress is None else progress
        n_total_len = len(str(n_total))
        n_of_n_msg = f"%0{n_total_len}d / %0{n_total_len}d" % (n_complete, n_total)
    else:
        n_of_n_msg = ""

    if per_sec is not None:
        per_sec_msg = f", {per_sec:06.2f} it/s"
    else:
        per_sec_msg = ""

    arrow = ("=" * int(progress*width)) + ">"
    blanks = " "*(width-len(arrow))

    print(f"[{arrow}{blanks}] ({n_of_n_msg}{per_sec_msg})", end="\r" if progress == 0.0 else "\n" if progress == 1.0 else "\r")


def memory_str(n_bytes: int):
    if n_bytes < 0:
        raise ValueError("Negative bytes!")

    log_bytes = np.log10(n_bytes)

    bytes_log_thresh = np.array([3, 6, 9, 12])
    bytes_label = ["B", "KB", "MB", "GB"]

    bytes_idx = max(np.searchsorted(bytes_log_thresh, log_bytes, side="right"), 0)

    return f"{np.power(10, 3 + log_bytes - bytes_log_thresh[bytes_idx]):0.1f}{bytes_label[bytes_idx]}"
