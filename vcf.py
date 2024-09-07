import numpy as np
import os
import subprocess
import pandas as pd

from enum import Enum

class VCFHeaderField(Enum):
    CHROM = 0
    POS = 1
    ID = 2
    REF = 3
    ALT = 4
    QUAL = 5
    FILTER = 6
    INFO = 7
    FORMAT = 8


def _process_sample_entry(entry: str):
    fields = entry.split(":")
    gt_field = fields[0]

    gt_vals = None
    for sep in ["/", "|"]:
        if sep in gt_field:
            gt_vals = gt_field.split(sep)
            break
    if gt_vals is None or len(gt_vals) != 2:
        raise RuntimeError(f"Error when parsing VCF, got unexpected value '{gt_field}' for GT tag.")

    return sum([np.nan if v == "." else int(v) for v in gt_vals])


def _process_data_row(row: str):
    columns = row.rstrip("\n").split("\t")
    meta = columns[0:9]
    samples = columns[9:]
    # Data rows need to be processed to extract the GT field, but metadata fields are passed through unchanged.
    return meta + [_process_sample_entry(sample) for sample in samples]


def read_vcf(path: str):
    with open(path, "r") as vcf_f:
        # Skip all header lines which begin with a double hash.
        post_header_rows = (row for row in vcf_f if not row.startswith("##"))
        # The column label row will have a single hash.
        col_labels = next(post_header_rows).strip("#").split("\t")
        # Data rows are everything following the column label row.
        data_rows = [_process_data_row(row=row) for row in post_header_rows]
        # Create a dataframe output
        geno_df = pd.DataFrame.from_records(data_rows, columns=col_labels, index=col_labels[:9])
        return geno_df


def make_traw_from_vcf(vcf_path: str):
    if not os.path.exists(vcf_path):
        raise RuntimeError(f"File '{vcf_path}' does not exist!")

    traw_fname = vcf_path.rstrip(".vcf.gz").rstrip(".vcf")

    arg_str = ["plink2", "--vcf", vcf_path, "--export", "Av", "--out", traw_fname]

    result = subprocess.check_output(arg_str)

    print(result)




