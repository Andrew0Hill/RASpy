import time
import logging
import numpy as np
import pandas as pd

from vcf import read_vcf, read_traw
from utils import parse_args, write_results, progress_bar, memory_str

log = logging.getLogger(__name__)


def compute_ras_original(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None):
    """ Compute RAS pairs and matrix in a similar way to the original Perl script.

    In order to make the computation more feasible in Python I made three minor adjustments, which are also
    explained inline in the code.

        1. Precompute the allele sums for each sample/variant prior to iteration. This saves time (and memory) since
           in the original script each allele sum is calculated n_samples times.
        2. Instead of sampling random variant indexes, and iterating the loop if the variant index is invalid,
           I instead precompute the set of valid variant indexes for each pair of samples, and sample num_vars
           elements from this set. This behavior is equivalent to the original code, but much faster.
        3. Simplified the similarity code logic to make it easier to understand.

    :param df: An input DataFrame (n_variants, n_samples) containing a loaded genotype file.
    :param gens: The number of generations to sample for.
                 This value corresponds to the number of output samples in the '*_rasPairs.csv' file.
    :param num_vars: The number of variants to sample for each of the 'gens' iterations.
    :param random_seed: An integer random seed to allow for reproducibility (if all other arguments remain identical)
    :return: A tuple containing two DataFrames:
                1. A DataFrame containing the (n_samples, n_samples) RAS matrix.
                2. A DataFrame containing the (n_samples^2, gens) RAS estimates for each pair of samples.
                   These `gens` estimates are averaged to obtain the value in the RAS matrix.
    """
    # Option to set seed for reproducibility
    rng = np.random.default_rng(seed=random_seed)
    # Get an array view on the DataFrame
    df_arr = df.to_numpy(copy=False)
    # Holds results for RAS
    ras_pair_dict = {}
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
            # Vector of all variants for sample1
            sample1_vec = df_arr[:, sample1_ind]
            # Vector of all variants for sample2
            sample2_vec = df_arr[:, sample2_ind]

            # Calculate which indices are valid (i.e. not NaN for either sample, and not 0 for sample1)
            # This saves us time inside the two inner loops, since we don't need to spend loop iterations sampling
            # from indices which are invalid.
            valid_idcs, = np.where(
                np.logical_and(
                    np.logical_and(~np.isnan(sample1_vec), ~np.isnan(sample2_vec)),
                    sample1_vec != 0
                )
            )
            # Iterate generations
            for i in range(gens):
                if len(valid_idcs) == 0:
                    scores = [0]*30
                else:
                    scores = []
                    # Since we know all indices we are sampling are valid, we can directly sample the exact number of variants
                    # we want (with replacement) for identical behavior to the original script.
                    for index in rng.choice(valid_idcs, size=num_vars):
                        # index = rng.choice(len(df_arr))
                        # Allele sums for each sample are precomputed
                        sum1 = df_arr[index, sample1_ind]
                        sum2 = df_arr[index, sample2_ind]
                        # if np.any(np.isnan([allele1, allele2, allele3, allele4])) | ((allele1 == 0) & (allele2 == 0)):
                        #     pass
                        if (np.isnan(sum1) | np.isnan(sum2)) | (sum1 == 0):
                            raise RuntimeError(
                                "Shouldn't encounter NaN sums, or sum1==0 since valid indices have been precomputed!")
                        else:
                            if sum1 == 0:
                                raise ValueError
                            # Python version of the similarity computation.
                            # Compared to original, we precompute allele sums, and
                            # simplify similarity comparison logic.
                            # Behavior is identical to original version.
                            if sum2 == 0:
                                similarity = 0
                            elif sum1 == sum2:
                                similarity = 1
                            else:
                                similarity = 0.5
                            # Construct a list of all scores for this generation
                            scores.append(similarity)
                # Compute the estimate for this generation by averaging the num_vars scores.
                score_means.append(np.mean(scores))
            # Save the num_vars scores for this pair of samples in a dictionary.
            ras_pair_dict[(sample1_lbl, sample2_lbl)] = score_means
            # Print progress
            progress_bar(n_of_n=(len(ras_pair_dict), n_pairs))
    # Construct a dataframe of the pair samples.
    out_pair_df = pd.DataFrame.from_dict(ras_pair_dict, orient="index")
    # Fix the index, so it properly takes up two columns.
    out_pair_df.set_index(pd.MultiIndex.from_tuples(out_pair_df.index, names=["from_id", "to_id"]), inplace=True)
    # Compute the matrix by averaging the num_vars samples for each sample pair, then unstack to create a matrix.
    out_matrix_df = out_pair_df.mean(axis=1).unstack(level="to_id")
    # Return the matrix and pair dataframes.
    return out_matrix_df, out_pair_df


def compute_ras_optimized(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None, no_sample: bool = False, matrix_mem_limit: int = None):
    """ Compute RAS pairs and matrix in an optimized way.

    This function should run much quicker than `compute_ras_original` but uses NumPy specific features to speed up
    the computation.

    :param df: An input DataFrame (n_variants, n_samples) containing a loaded genotype file.
    :param gens: The number of generations to sample for.
                 This value corresponds to the number of output samples in the '*_rasPairs.csv' file.
    :param num_vars: The number of variants to sample for each of the 'gens' iterations.
    :param random_seed: An integer random seed to allow for reproducibility (if all other arguments remain identical)
    :param no_sample: A boolean flag to disable the bootstrap sampling, and instead compute RAS over all variants.
    :param matrix_mem_limit: Limit the matrix blocks we generate to be <= matrix_mem_limit bytes. Default is 4GB but
                             higher values will enable quicker execution.
    :return: A tuple containing two DataFrames:
                1. A DataFrame containing the (n_samples, n_samples) RAS matrix.
                2. A DataFrame containing the (n_samples^2, gens) RAS estimates for each pair of samples.
                   These `gens` estimates are averaged to obtain the value in the RAS matrix.
    """
    # Option to set seed for reproducibility
    rng = np.random.default_rng(seed=random_seed)
    # Get an array view on the DataFrame
    df_arr = df.to_numpy(copy=False)

    # We calculate a blocksize that allows us to stay within a reasonable memory limit (4GB)
    # TODO: Make this configurable, or at least provide a log message about this limit.
    size_thresh = matrix_mem_limit or 4e9
    n_variants = df_arr.shape[0]
    n_samples = df_arr.shape[1]

    log.info(f"Using a block size memory threshold of {memory_str(size_thresh)}.")
    block_size_float = size_thresh/(n_variants*n_samples*4)
    block_size_int = min(max(int(block_size_float), 1), n_samples)

    block_start_idx = np.arange(0, n_samples, block_size_int)
    block_end_idx = np.concatenate([block_start_idx[1:], [n_samples]])
    n_blocks = block_start_idx.shape[0]

    if block_size_float < 1:
        # Assumes 32-bit float (4 bytes)
        alloc_size = memory_str(int(n_variants * n_samples * 4))
        log.warning(f"Calculating similarity for single pair requires allocating ({n_variants}, {n_samples}) matrix ({alloc_size}). You may experience a crash if your computer doesn't have enough free memory!")
    else:
        log.info(f"Will calculate RAS in blocks of ({block_size_int}, {n_samples}) ({n_blocks} blocks).")

    # Here we take advantage of NumPy broadcasting to allow us to quickly compute
    # pairwise similarity between samples across *all* variants.
    # df_arr = (n_variants, n_samples)
    # matrix_exp1 = (n_variants, 1, n_samples)
    # chunk_exp2 =  (n_variants, chunk_size, 1)
    #
    # When broadcasted, the resulting array will be of shape:
    # ras_matrix_chunk = (n_variants, chunk_size, n_samples)
    #
    # Taking the np.nanmean across the first dimension (n_variants)
    # will reduce this to a (chunk_size, n_samples) matrix of RAS values.
    # We can concatenate all chunks along the first dimension to construct
    # the (n_samples, n_samples) matrix without allocating too much memory at once.

    matrix_exp1 = np.expand_dims(df_arr, 1).copy()
    matrix_exp1[matrix_exp1 == 0] = np.nan

    full_pairs = []
    full_matrix = []

    for i, (start_idx, end_idx) in enumerate(zip(block_start_idx, block_end_idx)):
        log.info(f"Processing block {i+1}/{n_blocks}...")
        chunk_exp2 = np.expand_dims(df_arr[:, start_idx:end_idx], 2)

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
        ras_matrix_chunk = np.minimum(matrix_exp1, chunk_exp2) / np.maximum(matrix_exp1, chunk_exp2)

        if no_sample:
            ras_matrix_chunk_mean = np.nanmean(ras_matrix_chunk, axis=0).transpose()
            chunk_matrix_df = pd.DataFrame(ras_matrix_chunk_mean, index=df.columns, columns=df.columns[start_idx: end_idx])
            chunk_matrix_df.index.name = "from_id"
            chunk_matrix_df.columns.name = "to_id"
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
            valid_idcs = valid_idcs.reindex(pd.MultiIndex.from_product([range(ras_matrix_chunk.shape[1]), range(ras_matrix_chunk.shape[2])]))
            assert len(valid_idcs) == ras_matrix_chunk.shape[1] * ras_matrix_chunk.shape[2]
            # A dictionary to hold output.
            chunk_pairs = {}

            # Iterate over each pair (sample1, sample2) and sample similarity across (num_vars, gens) locations.
            for i, (idx_pair, idx_list) in enumerate(zip(valid_idcs.index, valid_idcs)):
                # If the pair has no valid combinations, RAS is 0
                if not isinstance(idx_list, np.ndarray):
                    ras_val = np.zeros(gens, dtype=np.float32)
                else:
                    # Randomly choose (with replacement) from the valid variant locations for this pair (sample1, sample2)
                    random_choice_idcs = rng.choice(idx_list, size=(num_vars, gens), replace=True)
                    # Extract the values of the random choice locations for this pair (sample1, sample2)
                    ras_samples = ras_matrix_chunk[random_choice_idcs, idx_pair[0], idx_pair[1]]
                    # ras_samples = np.take(tmp_matrix, random_choice_idcs)
                    # Take the mean across the `num_vars` dimension to generate `gens` estimates of the mean RAS.
                    ras_val = ras_samples.mean(axis=0)
                chunk_pairs[df.columns[start_idx + idx_pair[0]], df.columns[idx_pair[1]]] = ras_val
                # Print a progress bar.
                progress_bar(n_of_n=(i + 1, len(valid_idcs)))

            # Construct a DataFrame of the output pairs (the equivalent to the .rareAlleleSharingPairs file from the original script).
            chunk_pair_df = pd.DataFrame.from_dict(chunk_pairs, orient="index")
            chunk_pair_df.set_index(pd.MultiIndex.from_tuples(chunk_pair_df.index, names=["to_id", "from_id"]), inplace=True)
            full_pairs.append(chunk_pair_df)

            # Construct a DataFrame of the output matrix (the equivalent to the .rareAlleleSharingMatrix file from the original script).
            chunk_matrix_df = chunk_pair_df.mean(axis=1).unstack(level="to_id")
        full_matrix.append(chunk_matrix_df)

    # Matrices are constructed slightly differently depending on if we sample (default) or compute
    # full RAS (--no_sample), so we concatenate differently here.
    if no_sample:
        full_matrix_df = pd.concat(full_matrix, axis=0)
        full_pairs_df = None
    else:
        full_matrix_df = pd.concat(full_matrix, axis=1)
        full_pairs_df = pd.concat(full_pairs, axis=0)

    # Return results
    return full_matrix_df, full_pairs_df


def compute_ras(df: pd.DataFrame, gens: int, num_vars: int, random_seed: int = None, use_optimized_method: bool = False, no_sample: bool = False, matrix_mem_limit: int = None):
    """ This is a dispatch function which will select either the original (use_optimized_method=False) or new (use_optimized_method=True)
    method of computing RAS.

    :param df: An input DataFrame (n_variants, n_samples) containing a loaded genotype file.
    :param gens: The number of generations to sample for.
                 This value corresponds to the number of output samples in the '*_rasPairs.csv' file.
    :param num_vars: The number of variants to sample for each of the 'gens' iterations.
    :param random_seed: An integer random seed to allow for reproducibility (if all other arguments remain identical)
    :param use_optimized_method: Boolean flag. If selected, use a faster method of computing RAS.
    :param no_sample: A boolean flag to disable the bootstrap sampling, and instead compute RAS over all variants.
    :param matrix_mem_limit: Limit the matrix blocks we generate to be <= matrix_mem_limit bytes. Default is 4GB but
                             higher values will enable quicker execution.
    :return: A tuple containing two DataFrames:
                1. A DataFrame containing the (n_samples, n_samples) RAS matrix.
                2. A DataFrame containing the (n_samples^2, gens) RAS estimates for each pair of samples.
                   These `gens` estimates are averaged to obtain the value in the RAS matrix.
    """
    if use_optimized_method:
        ras_result = compute_ras_optimized(df=df, gens=gens, num_vars=num_vars, random_seed=random_seed, no_sample=no_sample, matrix_mem_limit=matrix_mem_limit)
    else:
        ras_result = compute_ras_original(df=df, gens=gens, num_vars=num_vars, random_seed=random_seed)
    return ras_result


if __name__ == "__main__":
    # Record start time of script.
    start_t = time.perf_counter()

    # Parse arguments.
    args = parse_args()

    # Read the VCF or .traw file into a DataFrame.
    if args.vcf is not None:
        geno_df = read_vcf(path=args.vcf, max_freq=args.max_freq)
    elif args.traw is not None:
        geno_df = read_traw(path=args.traw, max_freq=args.max_freq)
    else:
        raise RuntimeError("Either --vcf or --traw input must be specified!")

    # Compute RAS
    ras_matrix, ras_pairs = compute_ras(
        df=geno_df,
        gens=args.gens,
        num_vars=args.num_vars,
        use_optimized_method=args.optimized,
        random_seed=args.random_seed,
        no_sample=args.no_sample,
        matrix_mem_limit=args.matrix_mem_limit
    )

    # Write results
    write_results(output_prefix=args.output_prefix, ras_matrix=ras_matrix, ras_pairs=ras_pairs)

    # Record elapsed time of script.
    elapsed_t = time.perf_counter() - start_t
    log.info(f"Script execution finished in {elapsed_t:0.2f} seconds.")
