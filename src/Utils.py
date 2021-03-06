from itertools import combinations
import numpy as np
import regex as re
from src.preprocessing import Symbol


def is_subsequence(caller, callee):
    m = len(caller)
    n = len(callee)
    if m == 0:
        return True
    if n == 0:
        return False

    if caller[m-1] == callee[n-1]:
        return is_subsequence(caller[:-1], callee[:-1])
    return is_subsequence(caller, callee[:-1])

def count_occurrence(text, item):
    return len(re.findall(item, text, overlapped=True))
