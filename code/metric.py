from bisect import bisect


# Actually O(N^2), but fast in practice for our data
def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):  # O(N)
        j = bisect(sorted_so_far, u)  # O(log N)
        inversions += i - j
        sorted_so_far.insert(j, u)  # O(N)
    return inversions

from dataclasses import dataclass

@dataclass 
class Score:
    cur_score:float
    total_inversions:int
    total_pairs:int

    def __init__(self, total_inversions, total_pairs):
        self.total_inversions = total_inversions
        self.total_pairs = total_pairs
        if total_pairs == 0:
            self.cur_score = 0.0
        else:
            self.cur_score = 1 - 4 * total_inversions / total_pairs

    def merge(a, b):
        return Score(a.total_inversions + b.total_inversions, a.total_pairs + b.total_pairs)


def kendall_tau_typed(ground_truth, predictions):
    total_inversions = 0  # total inversions in predicted ranks across all instances
    total_2max = 0  # maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return Score(total_inversions=total_inversions, total_pairs=total_2max)

def kendall_tau(ground_truth, predictions):
    score = kendall_tau_typed(ground_truth, predictions)
    return [score.cur_score, score.total_inversions, score.total_pairs]


def sum_scores(a, b):
    total_inversions = a[1] + b[1]
    total_2max = a[2] + b[2]
    return [1 - 4 * total_inversions / total_2max, total_inversions, total_2max]

def calc_nb_score(my_order, correct_order):
    ground_truth = [correct_order]
    predictions = [my_order]

    return kendall_tau_typed(ground_truth, predictions)