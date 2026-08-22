from littlelm.backend import xp


def cross_entropy(logits, targets):
    """
    logits:  Tensor shape (B, S, V)
    targets: np.ndarray of int, shape (B, S)
    returns: scalar Tensor (the mean loss across batch * seq)
    """
    B, S, V = logits.shape
    flat_logits = logits.reshape(B * S, V)
    flat_targets = targets.reshape(-1)

    shifted = flat_logits - flat_logits.max(axis=-1, keepdims=True)
    log_sum_exp = shifted.exp().sum(axis=-1, keepdims=True).log()
    log_probs = shifted - log_sum_exp

    N = B * S
    selected = log_probs[xp.arange(N), flat_targets]
    return -selected.mean()
