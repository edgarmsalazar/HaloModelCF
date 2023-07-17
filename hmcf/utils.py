from typing import Callable
from datetime import timedelta
from time import time

BULLET = "\u25CF"


class COLS:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


OKGOOD = f"{COLS.OKGREEN}{BULLET}{COLS.ENDC} "
FAIL = f"{COLS.FAIL}{BULLET}{COLS.ENDC} "


def timer(
    procedure: Callable,
) -> Callable:
    """Decorator that prints the procedure's execution time."""

    def wrapper(*args, **kwargs):
        start = time()
        return_value = procedure(*args, **kwargs)
        print(
            f"\t{BULLET}{COLS.BOLD}{COLS.WARNING} Elapsed time:{COLS.ENDC} "
            + f"{COLS.OKCYAN}{timedelta(seconds=time()-start)}{COLS.ENDC} " 
            + f"{COLS.OKGREEN}{procedure.__name__}{COLS.ENDC}"
        )
        return return_value

    return wrapper
