import argparse
import numpy as np
import pandas as pd

from vcf import read_vcf
from utils import progress_bar


def compute_ras_original(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None):
    # Option to set seed for reproducibility
    rng = np.random.default_rng(seed=random_seed)
    # Get an array view on the DataFrame
    df_arr = df.to_numpy(copy=False)
    # Holds results for RAS
    ras_pairs = {}
    # Iterate over all samples (even identity pairs) and calculate RAS
    n_pairs = len(df.columns) ** 2
    for sample1_ind, sample1_lbl in enumerate(df.columns):
        for sample2_ind, sample2_lbl in enumerate(df.columns):
            ras_pairs[(sample1_lbl, sample2_lbl)] = calculate_ras_pair(matrix=df_arr, id1=sample1_ind, id2=sample2_ind,
                                                                       rng=rng, gens=gens, num_vars=num_vars)
            progress_bar(n_of_n=(len(ras_pairs), n_pairs))
    return ras_pairs


def calculate_ras_pair(matrix, id1, id2, rng, gens: int, num_vars: int):
    score_means = []
    for i in range(gens):
        scores = []
        while len(scores) < num_vars:
            index = rng.choice(len(matrix))
            sum1 = matrix[index, id1]
            sum2 = matrix[index, id2]
            # if np.any(np.isnan([allele1, allele2, allele3, allele4])) | ((allele1 == 0) & (allele2 == 0)):
            #     pass
            if np.any(np.isnan([sum1, sum2])) | (sum1 == 0):
                pass
            else:

                if sum1 == 0:
                    raise ValueError
                if sum2 == 0:
                    similarity = 0
                elif sum1 == sum2:
                    similarity = 1
                else:
                    similarity = 0.5
                # Python version of Perl script code below.
                # Compared to the version above, we precompute the allele sums
                # when we read the VCF, so we don't need to recalculate
                # sum1 and sum2 multiple times unnecessarily.
                # I also simplified the logic for similarity.

                # allele1, allele2 = matrix[index, id1]
                # allele3, allele4 = matrix[index, id2]
                # sum1 = allele1 + allele2
                # sum2 = allele3 + allele4
                # if sum1 == 0:
                #     raise ValueError
                # elif sum1 == 1:
                #     if sum2 == 0:
                #         similarity=0
                #     elif sum2 == 1:
                #         similarity=1
                #     elif sum2 == 2:
                #         similarity=0.5
                # elif sum1 == 2:
                #     if sum2 == 0:
                #         similarity=0
                #     elif sum2 == 1:
                #         similarity=0.5
                #     elif sum2 == 2:
                #         similarity=1
                scores.append(similarity)
        score_means.append(np.mean(scores))
    return score_means


def compute_ras(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None, go_fast: bool = False):
    if go_fast:
        pass
    else:
        return compute_ras_original(df=df, gens=gens, num_vars=num_vars, random_seed=random_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=False, help="Path to a VCF file to read.")
    parser.add_argument("--traw", required=False, help="Path to a .traw file to read.")
    parser.add_argument("--output_prefix", required=True, help="Path/prefix for the output files.")
    parser.add_argument("--max_freq", type=float, required=True, default=0.1,
                        help="Variants with MAF > max_freq will be excluded from the RAS calculation.")
    parser.add_argument("--gens", type=int, required=True, default=30, help="Number of bootstrap iterations to perform")
    parser.add_argument("--n_variants", type=int, required=True, default=500,
                        help="At each bootstrap iteration, sample until we have similarity scores for n_variants.")
    parser.add_argument("--go_fast", default=True, action="store_true",
                        help="If this flag is passed, use a faster RAS computation which is numerically identical to the original implementation.")

    args = parser.parse_args()

    vcf_df = read_vcf(args.vcf)

    ras_result = compute_ras(vcf_df, gens=args.gens, num_vars=args.n_variants, go_fast=False)

    print("Done!")
