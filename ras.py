import argparse
import logging
import datetime
import sys

import numpy as np
import pandas as pd

from vcf import read_vcf, read_traw
from utils import progress_bar, setup_logging, memory_str

log = logging.getLogger(__name__)


def compute_ras_optimized(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None, no_sample: bool = False):
    # Option to set seed for reproducibility
    rng = np.random.default_rng(seed=random_seed)
    # Get an array view on the DataFrame
    df_arr = df.to_numpy(copy=False)

    # Dividing pairwise min/max is a simpler but equivalent way of writing the logic used in compute_ras_original.
    # There are 9 cases, but 3 are skipped (where sample 1 count == 0)
    #
    #   --------------------------------
    #   | sample 1 | sample 2 | allele |
    #   |  count   |  count   |  sim   |
    #   --------------------------------
    #   |    0     |     0    |  <NA>  |
    #   |    0     |     1    |  <NA>  |
    #   |    0     |     2    |  <NA>  |
    #   |    1     |     0    |    0   |
    #   |    1     |     1    |    1   |
    #   |    1     |     2    |   0.5  |
    #   |    2     |     0    |    0   |
    #   |    2     |     1    |   0.5  |
    #   |    2     |     2    |    1   |
    #   --------------------------------

    # Since this broadcast operation is O(n^2) in memory, we will
    # perform the operation in chunks if the matrix size is too big.

    # We calculate a blocksize that allows us to stay within a reasonable memory limit (4GB)
    size_thresh = 4e9
    n_variants = df_arr.shape[0]
    n_samples = df_arr.shape[1]

    block_size_float = size_thresh/(n_variants*n_samples*4)
    block_size_int = max(int(block_size_float), 1)

    if block_size_float < 1:
        # Assumes 32-bit float (4 bytes)
        alloc_size = memory_str(int(n_variants * n_samples * 4))
        log.warning(f"Calculating similarity for single pair requires allocating ({n_variants}, {n_samples}) matrix ({alloc_size}). You may experience a crash if your computer doesn't have enough free memory!")

    block_start_idx = np.arange(0,n_samples,block_size_int)
    block_end_idx = np.concatenate([block_start_idx[1:], [n_samples]])
    n_blocks = block_start_idx.shape[0]

    matrix_exp1 = np.expand_dims(df_arr, 1).copy()
    matrix_exp1[matrix_exp1 == 0] = np.nan

    full_pairs = []
    full_matrix = []

    for i, (start_idx, end_idx) in enumerate(zip(block_start_idx, block_end_idx)):
        log.info(f"Processing block {i+1}/{n_blocks}...")
        # This is a little unintuitive, but here we do the following:
        # Create two numpy arrays with expanded dimensions, to allow for a pairwise
        # computation using broadcasting.
        # df_arr = (n_variants, n_samples)
        # df_arr1 = (n_variants, 1, n_samples)
        # When broadcasted, the resulting array will be of shape:
        # ras_matrix_full = (n_variants, n_samples, n_samples)
        # or a matrix of (n_samples, n_samples) for each variant.
        # df_arr2 = (n_variants, n_samples, 1)
        chunk_exp2 = np.expand_dims(df_arr[:, start_idx:end_idx], 2)

        ras_matrix_chunk = np.minimum(matrix_exp1, chunk_exp2) / np.maximum(matrix_exp1, chunk_exp2)

        if no_sample:
            ras_matrix_chunk_mean = np.nanmean(ras_matrix_chunk, axis=0)
            chunk_matrix_df = pd.DataFrame(ras_matrix_chunk_mean, index=df.columns, columns=df.columns)
            chunk_matrix_df.index.name = "to_id"
            chunk_matrix_df.columns.name = "from_id"
            # Above we compute the full similarity across all non-missing genotypes for all samples.
            # If we don't care about bootstrap sampling, we can skip the sampling process.
        else:
            # Get indices where the RAS matrix is not NaN (i.e. locations with valid genotypes)
            idx_var, idx_sample1, idx_sample2 = np.where(~np.isnan(ras_matrix_chunk))
            # Construct a dataframe of the indices of valid genotypes.
            idx_df = pd.DataFrame({"idx_var": idx_var, "idx_sample1": idx_sample1, "idx_sample2": idx_sample2})
            # Collect the valid variant indices for each combination of sample1, sample2.
            # This lets us avoid wasting time sampling from indices which we know are missing/invalid.
            valid_idcs = idx_df.groupby(["idx_sample1", "idx_sample2"]).idx_var.apply(np.array)

            # A dictionary to hold output.
            chunk_pairs = {}

            # Iterate over each pair (sample1, sample2) and sample similarity across (num_vars, gens) locations.
            for i, (idx_pair, idx_list) in enumerate(zip(valid_idcs.index, valid_idcs)):
                # Randomly choose (with replacement) from the valid variant locations for this pair (sample1, sample2)
                random_choice_idcs = rng.choice(idx_list, size=(num_vars, gens), replace=True)
                # Extract the values of the random choice locations for this pair (sample1, sample2)
                ras_samples = ras_matrix_chunk[random_choice_idcs, idx_pair[0], idx_pair[1]]
                # ras_samples = np.take(tmp_matrix, random_choice_idcs)
                # Take the mean across the `num_vars` dimension to generate `gens` estimates of the mean RAS.
                chunk_pairs[df.columns[idx_pair[0]], df.columns[idx_pair[1]]] = ras_samples.mean(axis=0)
                # Print a progress bar.
                progress_bar(n_of_n=(i + 1, len(valid_idcs)))

            # Construct a DataFrame of the output pairs (the equivalent to the .rareAlleleSharingPairs file from the original script).
            chunk_pair_df = pd.DataFrame.from_dict(chunk_pairs, orient="index")
            chunk_pair_df.set_index(pd.MultiIndex.from_tuples(chunk_pair_df.index, names=["to_id", "from_id"]), inplace=True)
            full_pairs.append(chunk_pair_df)

            # Construct a DataFrame of the output matrix (the equivalent to the .rareAlleleSharingMatrix file from the original script).
            chunk_matrix_df = chunk_pair_df.mean(axis=1).unstack(level="to_id")
        full_matrix.append(chunk_matrix_df)

    if len(full_pairs) != 0:
        full_pairs_df = pd.concat(full_pairs, axis=0)

    full_matrix_df = pd.concat(full_matrix, axis=1)
    # Return results
    return full_matrix_df, full_pairs_df


def compute_ras_original(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None):
    # Option to set seed for reproducibility
    rng = np.random.default_rng(seed=random_seed)
    # Get an array view on the DataFrame
    df_arr = df.to_numpy(copy=False)
    # Holds results for RAS
    ras_pairs = {}
    # Iterate over all samples (even identity pairs) and calculate RAS
    n_pairs = len(df.columns) ** 2
    # Print log message
    log.info("Starting RAS calculation with original method.")
    # df columns
    sample_list = df.columns
    # Iterate samples
    for sample1_ind, sample1_lbl in enumerate(sample_list):
        # Iterate samples again
        for sample2_ind, sample2_lbl in enumerate(sample_list):
            score_means = []

            sample1_vec = df_arr[:, sample1_ind]
            sample2_vec = df_arr[:, sample2_ind]

            # Calculate which indices are valid (i.e. not NaN for either sample, and not 0 for sample1
            valid_idcs, = np.where(
                np.logical_and(
                    np.logical_and(~np.isnan(sample1_vec), ~np.isnan(sample2_vec)),
                    sample1_vec != 0
                )
            )
            # Iterate generations
            for i in range(gens):
                scores = []
                # Iterate and sample variants until we've computed `num_vars` valid scores
                for index in rng.choice(valid_idcs, size=500):
                   # index = rng.choice(len(df_arr))
                    sum1 = df_arr[index, sample1_ind]
                    sum2 = df_arr[index, sample2_ind]
                    # if np.any(np.isnan([allele1, allele2, allele3, allele4])) | ((allele1 == 0) & (allele2 == 0)):
                    #     pass
                    if (np.isnan(sum1) | np.isnan(sum2)) | (sum1 == 0):
                        raise RuntimeError("Shouldn't encounter NaN sums, or sum1==0 since valid indices have been precomputed!")
                    else:
                        if sum1 == 0:
                            raise ValueError
                        # Simplified version of the similarity computation logic.
                        # Original version (translated to Python) is available as a comment below.
                        # Behavior is identical to original version.
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
            ras_pairs[(sample1_lbl, sample2_lbl)] = score_means
            progress_bar(n_of_n=(len(ras_pairs), n_pairs))

    out_pair_df = pd.DataFrame.from_dict(ras_pairs, orient="index")
    out_pair_df.set_index(pd.MultiIndex.from_tuples(out_pair_df.index, names=["from_id", "to_id"]), inplace=True)

    out_matrix_df = out_pair_df.mean(axis=1).unstack(level="to_id")

    return out_matrix_df, out_pair_df


def compute_ras(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None, use_optimized_method: bool = False, no_sample: bool = False):
    if use_optimized_method:
        ras_result = compute_ras_optimized(df=df, gens=gens, num_vars=num_vars, random_seed=random_seed, no_sample=no_sample)
    else:
        ras_result = compute_ras_original(df=df, gens=gens, num_vars=num_vars, random_seed=random_seed)
    return ras_result


def parse_args():
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
    parser.add_argument("--no-sample", action="store_true", help="If this flag is passed, calculate RAS across all variants which pass filtering without random sampling.")

    parsed_args = parser.parse_args()

    # If output prefix is None, use input filename as output prefix.
    if parsed_args.output_prefix is None:
        input_filename = parsed_args.traw or parsed_args.vcf
        parsed_args.output_prefix = f"{input_filename.rsplit('.', maxsplit=1)[0]}"
    # Append a datetime string to the prefix to avoid overwriting results.
    parsed_args.output_prefix += f"_{start_time_str}"

    # Setup logging
    setup_logging(prefix=parsed_args.output_prefix)

    # Log the script name.
    log.info("RAS Python Script v1.0")
    log.info(f"Current Time: {start_time.strftime('%c')}")

    # Print program configuration to log.
    parsed_args_d = parsed_args.__dict__
    max_key_len = max([len(k) for k in parsed_args_d.keys()])
    max_val_len = max([len(str(v)) for v in parsed_args_d.values()])

    header_len = max_key_len + max_val_len + 2

    log.info("Args".center(header_len, "="))
    for arg_key, arg_val in parsed_args_d.items():
        log.info(f"{arg_key.rjust(max_key_len)}: {arg_val}")
    log.info("="*header_len)

    return parsed_args


def write_results(output_prefix: str, results: tuple[pd.DataFrame, pd.DataFrame]):
    ras_matrix, ras_pairs = results
    # Write results to file
    # Write RAS pairs (if computed)
    if ras_pairs is not None:
        ras_pairs_filename = f"{output_prefix}_rasPairs.csv"
        ras_pairs.to_csv(ras_pairs_filename)
        log.info(f"Wrote RAS pairs to: '{ras_pairs_filename}'")

        # Write RAS matrix
    ras_matrix_filename = f"{output_prefix}_rasMatrix.csv"
    ras_matrix.to_csv(ras_matrix_filename)
    log.info(f"Wrote RAS matrix to: '{ras_matrix_filename}'")


if __name__ == "__main__":
    # Parse arguments.
    args = parse_args()

    # Read the VCF or .traw file into a DataFrame.
    if args.vcf is not None:
        geno_df = read_vcf(path=args.vcf, max_freq=args.max_freq)
    elif args.traw is not None:
        geno_df = read_traw(path=args.traw, max_freq=args.max_freq)
    else:
        raise RuntimeError("Either --vcf or --traw input must be specified!")

    geno_df = pd.concat([geno_df]*10, axis=1)
    geno_df.columns = [f"sample_{i}" for i in range(geno_df.shape[1])]

    # Compute RAS
    ras_results = compute_ras(df=geno_df, gens=args.gens, num_vars=args.num_vars, use_optimized_method=args.optimized, random_seed=args.random_seed, no_sample=args.no_sample)

    # Write results
    write_results(results=ras_results, output_prefix=args.output_prefix)

    log.info("Done!")
