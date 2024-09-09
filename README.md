# RAS Python Script

This is a Python implementation of the `rareAlleleSharing.pl` script from https://github.com/PollockLaboratory/Schisto.

## Prerequisites

This script was tested under Python 3.11.7 but will likely work with most other versions of Python 3 as well.

This script's third-party dependencies are:
- `numpy <= 2.1.1 (latest)`
- `pandas >=1.5, <=2.2.2 (latest)`

Other versions of `numpy` and `pandas` may be compatible as well.

## Installation
To install the script, clone this repository to your local machine.

```shell
git clone https://github.com/Andrew0Hill/RASpy
```

Once cloned, I recommend creating a new virtual environment to avoid overwriting your existing Python package configuration:

```shell
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

The above code snippet will create a virtual environment, activate it, and install the required package dependencies.

## Script Execution

The script may be executed from the command line. An example command is shown below:

```shell
python3 ras.py --vcf input.vcf \
               --output_prefix output \
               --max_freq 0.1 \
               --gens 30 \
               --num_vars 500 \
               --random_seed 1
```

## Script Arguments

| flag              | description                                                                                                                                                                                                                                                                            |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--vcf`           | Path to an input `.vcf` file. `.vcf.gz` is not currently supported.                                                                                                                                                                                                                    |
| `--traw`          | Path to an input `.traw` file.                                                                                                                                                                                                                                                         |
| `--output_prefix` | Path and name for the output files that will be generated.                                                                                                                                                                                                                             |
| `--max_freq`      | MAF filter. Variants with MAF > `max_freq` (non-rare variants) will be discarded before calculating RAS.                                                                                                                                                                               |
| `--gens`          | Number of bootstrap iterations to perform.                                                                                                                                                                                                                                             | 
| `--num_vars`      | Number of variants to sample per bootstrap iteration.                                                                                                                                                                                                                                  |
| `--random_seed`   | A random seed for reproducibility.                                                                                                                                                                                                                                                     |
| `--no_sample`     | Skip bootstrap sampling and directly compute the average RAS across all `n_variants` for each pair of samples.                                                                                                                                                                         |
| `--optimized`     | If flag is set, use an optimized (but numerically equivalent) method for RAS computation which is significantly faster than the original implementation. Default behavior (flag not set) uses a near-verbatim Python translation of the RAS calculation from `rareAlleleSharing.pl`.   | 

> [!WARNING]
> If --random_seed is set, you must also ensure that all values for other flags (`--gens`, `--vars`, `--max_freq`, `--optimized`, etc.) are the same to ensure reproducible results.
> Results will be reproducible for a given Python/NumPy version, but may differ across Python/NumPy versions.

## Input Files

The script accepts both VCF and .traw (`plink2 --export Av` format) as input using the `--vcf` and `--traw` flags respectively.

VCF input is slower as it parses the file line-by-line, but may be slightly more space efficient for large files since it can filter and discard rows with MAF > `max_freq` as they are read. 

TRAW input uses `pd.read_csv`, so it will read the entire `.traw` file into memory before filtering variants.  

## Output Files

The script will generate three main output files:

1. `<output_prefix>_<date>.log` A log file containing the script's run configuration and log messages
2. `<output_prefix>_<date>_rasMatrix.csv` The (`n_samples`, `n_samples`) matrix containing pairwise RAS estimates for each pair of samples.
3. `<output_prefix>_<date>_rasPairs.csv` A file containing `--gens` estimates of RAS for each combination of samples. These estimates are averaged to create the corresponding mean estimate in the `_rasMatrix.csv` file.

