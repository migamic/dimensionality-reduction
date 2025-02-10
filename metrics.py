import numpy as np
from scipy import stats

def pq_normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)

def pq_shepard_diagram_correlation(D_high, D_low):
    return stats.spearmanr(D_high, D_low)[0]
