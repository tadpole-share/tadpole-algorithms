"""Tadpole metrics"""
import numpy as np

def mean_abs_error(estimate, filtered):
    """Returns the mean absolute error.

    Args:
        estimate ([type]): [description]
        filtered ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.mean(np.abs(estimate - filtered))


def weighted_error_score(estimate, upper_estimate, lower_estimate, filtered):
    """Returns weighted error score.

    Args:
        estimate ([type]): [description]
        upper_estimate ([type]): [description]
        lower_estimate ([type]): [description]
        filtered ([type]): [description]

    Returns:
        [type]: [description]
    """
    coeff = 1/(upper_estimate - lower_estimate)
    return np.sum(coeff * np.abs(estimate - filtered))/np.sum(coeff)


def cov_prob_acc(upper_estimate, lower_estimate, filtered):
    """Returns coverage probability accuracy.

    Args:
        upper_estimate ([type]): [description]
        lower_estimate ([type]): [description]
        filtered ([type]): [description]

    Returns:
        [type]: [description]
    """
    cov_prob = ( np.sum((lower_estimate < filtered) &
                        (upper_estimate > filtered)) * 1. )/filtered.shape[0]
    return np.abs(cov_prob - 0.5)
