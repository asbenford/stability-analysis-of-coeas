import sys
from tqdm import tqdm

# Method that situationally updates a progress bar in terminal or prints
# progress every 0.1% to a log file.

def smart_progress(total, desc="Progress"):
    is_tty = sys.stdout.isatty()
    if is_tty:
        pbar = tqdm(total=total, desc=desc, ncols=80)
        def update(n=1):
            pbar.update(n)
        def close():
            pbar.close()
    else:
        done = 0
        update_gap = 0.1
        next_update = 0
        def update(n=1):
            nonlocal done, next_update
            done += n
            percent = 100 * done / total
            if percent > next_update:
                percent = 100 * done / total
                sys.stdout.write(f"{desc}: {percent:6.2f}%\n")
                sys.stdout.flush()
                next_update += update_gap
        def close():
            sys.stdout.write(f"{desc}: 100.00%\n")
            sys.stdout.flush()
    return update, close