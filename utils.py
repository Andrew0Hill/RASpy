import sys
import logging

import numpy as np


def setup_logging(prefix: str):
    log_name = f"{prefix}.log"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(filename=log_name),
            logging.StreamHandler(sys.stderr)
        ],
        format="%(asctime)s [%(levelname)s] (%(filename)s) %(message)s"
    )


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
