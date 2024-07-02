# -*- coding: utf-8 -*-
"""CUDA-enabled LegendreDecomposition calculations"""
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
from typing import Sequence
from scipy.special import logsumexp as scipy_logsumexp

try:
    import cupy as cp
    from cupyx.scipy.special import logsumexp as cupy_logsumexp
except ImportError:
    import numpy as cp
    from scipy.special import logsumexp as cupy_logsumexp
    def get_array_module(X):
        return np
    cp.get_array_module = get_array_module


def default_B(shape: Sequence[int], order: int, xp: ModuleType = np) -> NDArray[np.intp]:
    """Vectorized implementation of the default B tensor.

    Args:
        shape: Shape of the corresponding X tensor.
        order: Order of the B tensor.
        xp (ModuleType): Array module, either numpy (CPU) or cupy (CUDA/GPU)

    Returns:
        array-like: Default B tensor of specified order.
    """
    B = xp.indices(shape).reshape(len(shape), -1).T
    mask = (B != 0).sum(axis=1) <= order
    return B[mask]

def kl(P: NDArray[np.float_], Q: NDArray[np.float_], xp: ModuleType = cp) -> np.float_:
    """Kullback-Leibler divergence.

    Args:
        P: P tensor
        Q: Q tensor
        xp (ModuleType): Array module, either numpy (CPU) or cupy

    Returns:
        KL divergence.
    """
    return xp.sum(P * xp.log(P / Q)) - xp.sum(P) + xp.sum(Q)


def get_eta(Q: NDArray[np.float_], D: int, xp: ModuleType = cp) -> NDArray[np.float_]:
    """Eta tensor.

    Args:
        Q: Q tensor
        D: Dimensionality
        xp (ModuleType): Array module, either numpy (CPU) or cupy

    Returns:
        Eta tensor.
    """
    for i in range(D):
        Q = xp.flip(xp.cumsum(xp.flip(Q, axis=i), axis=i), axis=i)
    return Q


def get_h(theta: NDArray[np.float_], D: int, xp: ModuleType = cp) -> NDArray[np.float_]:
    """H tensor.

    Args:
        theta: Theta tensor
        D: Dimensionality
        xp (ModuleType): Array module, either numpy (CPU) or cupy

    Returns:
        Updated theta.
    """
    for i in range(D):
        theta = xp.cumsum(theta, axis=i)
    return theta

def LD(X: NDArray[np.float_],
    B: NDArray[np.intp] | list[tuple[int, ...]] | None = None,
    order: int = 2,
    n_iter: int = 10,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    ngd: bool = True,
    verbose: bool = True,
    gpu: bool = True,
    exit_abs: bool = False,
    dtype: np.dtype | None = None,
) -> tuple[list[list[float]], np.float_, NDArray[np.float_], NDArray[np.float_]]:
    """Compute many-body tensor approximation.

    Args:
        X: Input tensor.
        B: B tensor.
        order: Order of default tensor B, if not provided.
        n_iter: Maximum number of iteration.
        lr: Learning rate.
        eps: (see paper).
        error_tol: KL divergence tolerance for the iteration.
        ngd: Use natural gradient.
        verbose: Print debug messages.
        gpu: Use GPU (CUDA or ROCm depending on the installed CuPy version).
        exit_abs: Previous implementation (wrongly?) uses kl- kl_prev as iteration exit criterion.
            Use abs(kl - kl_prev) instead.
        dtype: By default, the data-type is inferred from the input data.

    Returns:
        all_history_kl: KL divergence history.
        scaleX: Scaled X tensor.
        Q: Q tensor.
        theta: Theta.
    """
    all_history_kl = []
    D = len(X.shape)
    S = X.shape

    if exit_abs:
        def within_tolerance(kld: np.float_, prev_kld: np.float_):
            return abs(prev_kld - kld) < error_tol
    else:
        def within_tolerance(kld: np.float_, prev_kld: np.float_):
            return prev_kld - kld < error_tol

    if gpu:
        X = cp.asarray(X, dtype=dtype)
        eps = cp.asarray(eps, dtype=dtype)
        lr = cp.asarray(lr, dtype=dtype)
        logsumexp = cupy_logsumexp
    else:
        logsumexp = scipy_logsumexp

    xp = cp.get_array_module(X)

    if verbose:
        print("Constructing B")
    if B is None:
        B = default_B(S, order, xp)

    B_array = xp.array(B)
    B_flat = xp.ravel_multi_index(B_array.T, S)  # type: ignore
    if verbose:
        print("B shape:", B_flat.shape)

    scaleX = xp.sum(X + eps)
    P = (X + eps) / scaleX

    Q = xp.ones(P.shape, dtype=dtype)  # TODO: ones_like?
    Q = Q / xp.sum(Q)
    ### eta
    # print("Get initial eta")
    eta_hat = get_eta(P, D, xp)
    eta_hat_b = xp.take(eta_hat, B_flat)
    ###
    eta_b = xp.empty((len(B),), dtype=dtype)
    theta_b = xp.zeros((len(B),), dtype=dtype)
    G = xp.zeros((len(B), len(B)), dtype=dtype)  # TODO: Too large!
    history_kl = []
    prev_kld = None
    # evaluation
    kld = kl(P, Q, xp)
    history_kl.append(kld)

    uuu, vvv = xp.tril_indices(len(B), 0)

    uv = xp.ravel_multi_index(xp.stack((uuu, vvv)), (len(B), len(B)))  # type: ignore
    I_flat = B_flat[uuu]
    J_flat = B_flat[vvv]
    K_flat = xp.ravel_multi_index(xp.maximum(B_array[uuu], B_array[vvv]).T, S)  # type: ignore

    if verbose:
        print("iter=", 0, "kl=", kld, "mse=", xp.mean((P - Q) ** 2))
    for i in range(n_iter):
        # compute eta
        eta = get_eta(Q, D, xp)
        eta_b = xp.take(eta, B_flat)

        # compute G
        xp.put(G, uv, xp.take(eta, K_flat) - xp.take(eta, I_flat) * xp.take(eta, J_flat))
        GG = G + G.T - xp.diag(G.diagonal())

        # update theta_b
        if ngd:
            # theta_b[1:] -= lr*np.linalg.pinv(G[1:,1:])@(eta_b[1:]-eta_hat_b[1:])
            v = xp.linalg.solve(GG[1:, 1:], lr * (eta_b[1:] - eta_hat_b[1:]))
            theta_b[1:] -= v
        else:
            theta_b -= lr * (eta_b - eta_hat_b)
        # theta_b=>theta
        theta = xp.zeros(S, dtype=dtype)
        xp.put(theta, B_flat, theta_b)

        # theta => H => Q
        Hq = get_h(theta, D, xp)

        # with logsumexp
        logQ_ = Hq
        logQ = logQ_ - logsumexp(logQ_)
        Q = xp.exp(logQ) + eps

        # evaluation
        kld = kl(P, Q, xp)
        history_kl.append(kld)
        if verbose:
            print("iter=", i + 1, "kl=", kld, "mse=", xp.mean((P - Q) ** 2))
        if prev_kld is not None and within_tolerance(kld, prev_kld):
            break
        prev_kld = kld

    all_history_kl.append([float(x) for x in history_kl])
    if gpu:
        scaleX = scaleX.get()  # type: ignore
        Q = Q.get()  # type: ignore
        theta = theta.get()  # type: ignore
    return all_history_kl, scaleX, Q, theta  # type: ignore

def BP(X: NDArray[np.float_],
    theta: NDArray[np.float_],
    submfd_eta: NDArray[np.float_],
    avg_scaleX: np.float_,
    B: NDArray[np.intp] | list[tuple[int, ...]] | None = None,
    order: int = 2,
    n_iter: int = 10,
    lr: float = 1.0,
    eps: float = 1.0e-5,
    error_tol: float = 1.0e-5,
    ngd: bool = True,
    verbose: bool = True,
    gpu: bool = True,
    exit_abs: bool = True,
    dtype: np.dtype | None = None,
) -> tuple[list[list[float]], np.float_, NDArray[np.float_], NDArray[np.float_]]:
    """Compute many-body tensor approximation.

    Args:
        X: Input tensor.
        theta: Q's theta coordinates
        submfd_eta: The m-flat submanifold specified by eta.
        avg_scaleX: The average scale of the pre-projection of X.
        B: B tensor.
        order: Order of default tensor B, if not provided.
        n_iter: Maximum number of iteration.
        lr: Learning rate.
        eps: (see paper).
        error_tol: KL divergence tolerance for the iteration.
        ngd: Use natural gradient.
        verbose: Print debug messages.
        gpu: Use GPU (CUDA or ROCm depending on the installed CuPy version).
        exit_abs: Previous implementation (wrongly?) uses kl- kl_prev as iteration exit criterion.
            Use abs(kl - kl_prev) instead.
        dtype: By default, the data-type is inferred from the input data.

    Returns:
        all_history_norm: L₂ norm history.
        P: P tensor.
        theta: Theta.
    """
    all_history_norm = []
    D = len(X.shape)
    S = X.shape

    if exit_abs:
        def within_tolerance(kld: np.float_, prev_kld: np.float_):
            return abs(prev_kld - kld) < error_tol
    else:
        def within_tolerance(kld: np.float_, prev_kld: np.float_):
            return prev_kld - kld < error_tol

    if gpu:
        X = cp.asarray(X, dtype=dtype)
        theta = cp.asarray(theta, dtype=dtype)
        submfd_eta = cp.asarray(submfd_eta, dtype=dtype)
        avg_scaleX = cp.asarray(avg_scaleX, dtype=dtype)
        eps = cp.asarray(eps, dtype=dtype)
        lr = cp.asarray(lr, dtype=dtype)
        logsumexp = cupy_logsumexp
    else:
        logsumexp = scipy_logsumexp

    xp = cp.get_array_module(X)

    if verbose:
        print("Constructing B")
    if B is None:
        B = default_B(S, order, xp)

    B_array = xp.array(B)
    B_flat = xp.ravel_multi_index(B_array.T, S)  # type: ignore
    if verbose:
        print("B shape:", B_flat.shape)

    full_B = default_B(S, D+1, xp)
    full_B_array = xp.array(full_B)
    full_B_flat = xp.ravel_multi_index(full_B_array.T, S)  # type: ignore
    # scaleX = xp.sum(X + eps)
    # Q = (X + eps) / scaleX
    # Q = X # Assuming already normalized

    # P = xp.ones(Q.shape, dtype=dtype)  # TODO: ones_like?
    P = X
    P = P / xp.sum(P)

    ### eta
    # print("Get initial eta")
    eta_hat = submfd_eta
    eta_hat_b = xp.take(eta_hat, B_flat)
    eta_hat_full_b = xp.take(eta_hat, full_B_flat)
    ###
    eta_b = xp.empty((len(B),), dtype=dtype)
    eta_full_b = xp.empty((len(full_B),), dtype=dtype)
    # theta_b = xp.take(theta, B_flat)
    # theta_b = xp.zeros((len(B),), dtype=dtype)
    # theta_full_b = xp.zeros((len(full_B),), dtype=dtype)
    theta_full_b = xp.take(theta, full_B_flat)

    G = xp.zeros((len(full_B), len(full_B)), dtype=dtype)  # TODO: Too large!
    history_norm = []
    prev_norm = None
    # evaluation
    norm = np.inf
    history_norm.append(norm)

    uuu, vvv = xp.tril_indices(len(full_B), 0)

    uv = xp.ravel_multi_index(xp.stack((uuu, vvv)), (len(full_B), len(full_B)))  # type: ignore
    I_flat = full_B_flat[uuu]
    J_flat = full_B_flat[vvv]
    K_flat = xp.ravel_multi_index(xp.maximum(full_B_array[uuu], full_B_array[vvv]).T, S)  # type: ignore

    if verbose:
        print("iter=", 0, "norm=", norm)
    for i in range(n_iter):
        # compute eta
        eta = get_eta(P, D, xp)
        eta_b = xp.take(eta, B_flat)
        eta_full_b = xp.take(eta, full_B_flat)

        # compute G
        xp.put(G, uv, xp.take(eta, K_flat) - xp.take(eta, I_flat) * xp.take(eta, J_flat))
        GG = G + G.T - xp.diag(G.diagonal())

        # update theta_b
        if ngd:
            # theta_b[1:] -= lr*np.linalg.pinv(G[1:,1:])@(eta_b[1:]-eta_hat[1:])
            v = xp.linalg.solve(GG[1:, 1:], lr * (eta_full_b[1:] - eta_hat_full_b[1:]))
            theta_full_b[1:] -= v
        else:
            theta_full_b -= lr * (eta_full_b - eta_hat_full_b)
        # theta_b=>theta
        # xp.put(theta, B_flat, theta_b)
        theta = theta_full_b.reshape(S)

        # theta => H => P
        Hq = get_h(theta, D, xp)

        # with logsumexp
        logP_ = Hq
        logP = logP_ - logsumexp(logP_)
        P = xp.exp(logP) + eps

        # evaluation
        norm = xp.linalg.norm(eta_b - eta_hat_b)
        history_norm.append(norm)
        if verbose:
            print("iter=", i + 1, "norm=", norm)
        if prev_norm is not None and within_tolerance(norm, prev_norm) and norm < error_tol:
            break
        prev_norm = norm

    all_history_norm.append([float(x) for x in history_norm])

    P = P * avg_scaleX

    if gpu:
        P = P.get()  # type: ignore
        theta = theta.get()  # type: ignore

    return all_history_norm, P, theta  # type: ignore
