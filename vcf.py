import logging

import numpy as np
import os
import time
import subprocess
import pandas as pd
from collections import defaultdict
from functools import cache
from utils import memory_str

log = logging.getLogger(__name__)


@cache
def _get_format_dictionary(format_str: str):
    return {v: i for i, v in enumerate(format_str.split(":"))}


@cache
def _get_format_key_index(format_str: str, key_str: str):

    format_dict = _get_format_dictionary(format_str)

    if key_str not in format_dict:
        raise RuntimeError(f"FORMAT string does not contain a '{key_str}' field! Got: '{format_str}'.")

    return format_dict[key_str]


def _process_sample_entry(entry: str, format_str: str):
    fields = entry.split(":")
    gt_field_index = _get_format_key_index(format_str=format_str, key_str="GT")
    gt_field = fields[gt_field_index]

    gt_vals = None
    for sep in ["/", "|"]:
        if sep in gt_field:
            gt_vals = gt_field.split(sep)
            break
    if gt_vals is None or len(gt_vals) != 2:
        # log.warning(f"Encountered unexpected value '{gt_field}' for GT tag, will be coded as NaN.")
        gt_vals = (".", ".")
        # raise RuntimeError(f"Error when parsing VCF, got unexpected value '{gt_field}' for GT tag.")

    return np.sum([np.float32(np.nan if v == "." else v) for v in gt_vals])


def _calculate_frequency(elems):
    return np.nanmean(elems) / 2


def _process_data_rows(row_gen, max_freq: float = None):
    # Iterate over the row generator.
    for i, row in enumerate(row_gen):
        # Extract columns.
        columns = row.rstrip("\n").split("\t")
        meta = columns[0:9]
        samples = columns[9:]
        # meta[8] should be a format string, which tells us what VCF values are in what fields.
        # We need to extract the GT field for allele counts.
        processed_row = [_process_sample_entry(entry=sample, format_str=meta[8]) for sample in samples]
        # Calculate the minor allele frequency of this variant
        var_freq = _calculate_frequency(processed_row)
        # If the max_freq filter is disabled (None) or frequency of
        # this variant is <= max_freq, we yield the row.
        # Rows which don't pass the filter are discarded.
        if (max_freq is None) or (var_freq <= max_freq):
            if (i+1) % 1000 == 0:
                log.info(f"Processed {i+1} variants...")
            yield meta + processed_row


def read_vcf(path: str, max_freq: float = None):
    with open(path, "r") as vcf_f:
        start = time.perf_counter()
        # Skip all header lines which begin with a double hash.
        post_header_rows = (row for row in vcf_f if not row.startswith("##"))
        # The column label row will have a single hash.
        col_labels = next(post_header_rows).strip("#").rstrip("\n").split("\t")
        # Data rows are everything following the column label row.
        data_rows = list(_process_data_rows(post_header_rows, max_freq=max_freq))
        # Create a dataframe output
        geno_df = pd.DataFrame.from_records(data_rows, columns=col_labels, index=col_labels[:9], )
        elapsed = time.perf_counter() - start
        # Print information about the VCF.
        log.info(f"VCF loaded in {elapsed:0.2f} seconds.")
        log.info(f"After variant filtering, matrix has {geno_df.shape[0]} variants and {geno_df.shape[1]} samples.")
        log.info(f"DataFrame size: {memory_str(geno_df.memory_usage().sum())}")
        return geno_df


def read_traw(path: str, max_freq: float):
    start = time.perf_counter()
    # Parse all columns except the 6 fixed metadata columns as float32.
    col_dtypes = {
        "CHR": "str",
        "SNP": "str",
        "(C)M": "str",
        "POS": "str",
        "COUNTED": "str",
        "ALT": "str"
    }
    # We don't know what the column names of the file will be, but the
    # defaultdict will return `np.float32` for every entry which is not in the
    # dict of key/dtypes above.
    col_dtypes_default = defaultdict(np.float32, col_dtypes)
    # Read in .traw file, which is tab-delimited
    traw_df = pd.read_csv(path, sep="\t", index_col=list(range(6)), dtype=col_dtypes_default)
    # PLINK2 .traw files hold the sum of the REF allele, so flip the count.
    traw_df = 2 - traw_df
    # Calculate allele frequencies.
    freqs = traw_df.apply(_calculate_frequency, axis=1)
    # Filter on allele frequencies.
    traw_df = traw_df.loc[freqs <= max_freq]
    # Caculate elapsed time.
    elapsed = time.perf_counter() - start
    # Print information about the VCF.
    log.info(f".traw file loaded in {elapsed:0.2f} seconds.")
    log.info(f"After variant filtering, matrix has {traw_df.shape[0]} variants and {traw_df.shape[1]} samples.")
    log.info(f"DataFrame size: {memory_str(traw_df.memory_usage().sum())}")
    return traw_df


def make_traw_from_vcf(vcf_path: str):
    if not os.path.exists(vcf_path):
        raise RuntimeError(f"File '{vcf_path}' does not exist!")

    traw_fname = vcf_path.rstrip(".vcf.gz").rstrip(".vcf")

    arg_str = ["plink2", "--vcf", vcf_path, "--export", "Av", "--out", traw_fname]

    result = subprocess.check_output(arg_str)

    log.info(result)
