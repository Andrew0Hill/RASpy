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