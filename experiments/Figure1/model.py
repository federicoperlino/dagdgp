"""Synthetic multi-fidelity example with empirical-Bayes GP inference."""

from itertools import combinations

import numpy as np
from scipy.optimize import minimize

SEED, NOISE = 23, 0.02  # random seed; noise sd of every node
BETA = 0.55  # discrepancy: direct linear effect of X on H
N_BASE, N_H = 25, 12
BASE = np.linspace(0.02, 0.98, N_BASE)  # inputs at which the nodes are observed
GAP_L1 = (18, 19, 20)  # L1 not observed at x = 0.74, 0.78, 0.82
GAP_L2 = (3, 4, 5, 6, 7)  # L2 not observed at x = 0.14, ..., 0.30
I_L1 = [i for i in range(N_BASE) if i not in GAP_L1]
I_L2 = [i for i in range(N_BASE) if i not in GAP_L2]


def f_L1(x):
    return 0.8 * np.sin(2 * np.pi * 2.9 * x + 0.8) + 0.10 * np.sin(
        2 * np.pi * 7.1 * x + 0.9
    )


def f_L2(x):
    return -0.75 * np.cos(2 * np.pi * x + 0.1) + 0.09 * np.sin(
        2 * np.pi * 4.3 * x + 0.4
    )


def peak(l1, l2, c1, c2, s):
    return np.exp(-((l1 - c1) ** 2 + (l2 - c2) ** 2) / (2 * s**2))


def g(l1, l2):
    """Transfer from the low fidelities: a weighted sum of three Gaussian peaks."""
    return (
        0.85 * peak(l1, l2, 0.30, 0.25, 0.40)
        - 0.75 * peak(l1, l2, -0.40, -0.35, 0.40)
        - 0.35 * peak(l1, l2, 0.55, -0.55, 0.32)
    )


def f_H(x):
    return g(f_L1(x), f_L2(x)) + BETA * x


def nested_design(y1_all, y2_all, k=N_H):
    """H inputs: the k inputs, among those where both L1 and L2 are observed, that are
    most spread out (maximin distance) in the standardised observed (x, L1, L2)."""
    cand = [i for i in I_L1 if i in I_L2]
    F = np.column_stack([BASE[cand], y1_all[cand], y2_all[cand]])
    F = (F - F.mean(0)) / F.std(0)
    best, score = None, -1
    for c in combinations(range(len(cand)), k):
        Q = F[list(c)]
        sc = np.sqrt(((Q[:, None] - Q[None]) ** 2).sum(-1))[np.triu_indices(k, 1)].min()
        if sc > score:
            best, score = c, sc
    return [cand[j] for j in best]


# Empirical-Bayes hyperparameters and importance sampling over the low fidelities.
NOISE_VAR_MAX = 2e-3  # upper bound on the noise variance of a single-node GP
ELLX_BOUNDS = (0.5, 2.0)  # lengthscale of the direct X -> H part: a slow trend
S2X_BOUNDS = (1e-3, 1.0)


def se(A, B, ell, s2=1.0):
    """Squared-exponential kernel (ARD when ell is a vector)."""
    A = np.atleast_2d(A.T).T / ell
    B = np.atleast_2d(B.T).T / ell
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
    return s2 * np.exp(-0.5 * np.clip(d2, 0, None))


def nlml(K, y):
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10
    a = np.linalg.solve(L, y)
    return 0.5 * a @ a + np.log(np.diag(L)).sum() + 0.5 * len(y) * np.log(2 * np.pi)


def best_of(obj, bounds, rng, restarts):
    """Run L-BFGS-B from random starting points and return the result with the lowest objective value."""
    best = None
    for _ in range(restarts):
        r = minimize(
            obj,
            [rng.uniform(lo, hi) for lo, hi in bounds],
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or r.fun < best.fun:
            best = r
    return best


def fit_se(x, y, rng, ell_min=0.03, restarts=20):
    """Type-II ML for a GP on x with an SE kernel: lengthscale, variance, noise."""

    def obj(t):
        ell, s2, n2 = np.exp(t)
        return nlml(se(x, x, ell, s2) + (n2 + 1e-8) * np.eye(len(x)), y)

    bounds = [
        (np.log(ell_min), np.log(2.0)),
        (np.log(1e-2), np.log(10.0)),
        (np.log(1e-5), np.log(NOISE_VAR_MAX)),
    ]
    ell, s2, n2 = np.exp(best_of(obj, bounds, rng, restarts).x)
    return {"ell": ell, "s2": s2, "n2": n2}


def gp_post(x, y, xs, hp, full_cov=False):
    """Posterior mean and covariance (or variance) at xs of the latent function."""
    K = se(x, x, hp["ell"], hp["s2"]) + (hp["n2"] + 1e-8) * np.eye(len(x))
    Ks = se(xs, x, hp["ell"], hp["s2"])
    L = np.linalg.cholesky(K)
    A = np.linalg.solve(L, Ks.T)
    mu = Ks @ np.linalg.solve(L.T, np.linalg.solve(L, y))
    if full_cov:
        return mu, se(xs, xs, hp["ell"], hp["s2"]) - A.T @ A
    return mu, np.clip(hp["s2"] - (A**2).sum(0), 0, None)


def k_H(xa, La, xb, Lb, hp):
    """Additive fusion kernel of H: s2_x k(x, x') + s2_l k_ARD(l, l') + s2_b."""
    return (
        se(xa, xb, hp["ell_x"], hp["s2_x"])
        + se(La, Lb, hp["ell_l"], hp["s2_l"])
        + hp["s2_b"]
    )


def fit_H(xH, LH, yH, rng, restarts=60):
    """Type-II ML for the hyperparameters of f_H, at the posterior means of L1, L2."""

    def unpack(t):
        e = np.exp(t)
        return {
            "ell_x": e[0],
            "s2_x": e[1],
            "ell_l": e[2:4],
            "s2_l": e[4],
            "s2_b": e[5],
            "n2": NOISE**2,
        }

    bounds = [
        (np.log(ELLX_BOUNDS[0]), np.log(ELLX_BOUNDS[1])),
        (np.log(S2X_BOUNDS[0]), np.log(S2X_BOUNDS[1])),
        (np.log(0.3), np.log(5.0)),
        (np.log(0.3), np.log(5.0)),
        (np.log(1e-2), np.log(5.0)),
        (np.log(1e-3), np.log(1.0)),
    ]

    def obj(t):
        hp = unpack(t)
        return nlml(k_H(xH, LH, xH, LH, hp) + hp["n2"] * np.eye(len(xH)), yH)

    return unpack(best_of(obj, bounds, rng, restarts).x)


def fit(S=4000, n_grid=300, n_surf=64, n_draws=6):
    """Generate the data and return importance-weighted posterior summaries."""
    rng = np.random.default_rng(SEED)
    y1_all = f_L1(BASE) + NOISE * rng.standard_normal(N_BASE)
    y2_all = f_L2(BASE) + NOISE * rng.standard_normal(N_BASE)
    I_H = nested_design(y1_all, y2_all)
    x1, x2, xH = BASE[I_L1], BASE[I_L2], BASE[I_H]
    y1, y2 = y1_all[I_L1], y2_all[I_L2]
    yH = f_H(xH) + NOISE * rng.standard_normal(N_H)

    # L1, L2: type-II ML, then S joint draws from their GP posteriors given their own data,
    # on the evaluation grid and at the H inputs
    hp1, hp2 = fit_se(x1, y1, rng), fit_se(x2, y2, rng)
    xg = np.linspace(0, 1, n_grid)
    P = np.concatenate([xg, xH])
    iH = np.arange(n_grid, len(P))
    draws, mean = {}, {}
    for k, (x, y, hp) in {"L1": (x1, y1, hp1), "L2": (x2, y2, hp2)}.items():
        mu, C = gp_post(x, y, P, hp, full_cov=True)
        draws[k] = (
            mu[None]
            + rng.standard_normal((S, len(P)))
            @ np.linalg.cholesky(C + 1e-8 * np.eye(len(P))).T
        )
        mean[k] = mu
    hpH = fit_H(xH, np.column_stack([mean["L1"][iH], mean["L2"][iH]]), yH, rng)

    # importance weights p(Y_H | L1, L2) and, for each draw, the GP posterior of f_H.
    # With the additive kernel the posterior mean of f_H(x, l) splits into an x-part
    # g_X(x) and an (l1, l2)-part plus constant, g_L(l) + b: both are kept, so that
    # E[f_H(x, l) | D] = surf_lat(l) + gx_mean(x) exactly.
    l1g = np.linspace(
        mean["L1"][:n_grid].min() - 0.08, mean["L1"][:n_grid].max() + 0.08, n_surf
    )
    l2g = np.linspace(
        mean["L2"][:n_grid].min() - 0.08, mean["L2"][:n_grid].max() + 0.08, n_surf
    )
    LL = np.array([(a, b) for b in l2g for a in l1g])  # l2 slowest, l1 fastest
    logw, mH, vH = np.empty(S), np.empty((S, n_grid)), np.empty((S, n_grid))
    surf, gx = np.empty((S, len(LL))), np.empty((S, n_grid))
    prior_var = hpH["s2_x"] + hpH["s2_l"] + hpH["s2_b"]
    for s in range(S):
        Lp = np.column_stack([draws["L1"][s], draws["L2"][s]])
        Lh = Lp[iH]
        Lt = np.linalg.cholesky(k_H(xH, Lh, xH, Lh, hpH) + hpH["n2"] * np.eye(N_H))
        a = np.linalg.solve(Lt, yH)
        logw[s] = -0.5 * a @ a - np.log(np.diag(Lt)).sum()
        alpha = np.linalg.solve(Lt.T, a)
        Ks = k_H(xg, Lp[:n_grid], xH, Lh, hpH)
        mH[s] = Ks @ alpha
        vH[s] = np.clip(prior_var - (np.linalg.solve(Lt, Ks.T) ** 2).sum(0), 0, None)
        surf[s] = (
            se(LL, Lh, hpH["ell_l"], hpH["s2_l"]) @ alpha + hpH["s2_b"] * alpha.sum()
        )
        gx[s] = se(xg, xH, hpH["ell_x"], hpH["s2_x"]) @ alpha
    w = np.exp(logw - logw.max())
    w /= w.sum()

    res = {}
    for k in ("L1", "L2"):
        D = draws[k][:, :n_grid]
        m = w @ D
        res[f"{k}_mean"], res[f"{k}_sd"] = (
            m,
            np.sqrt(np.clip(w @ (D - m) ** 2, 0, None)),
        )
    m = w @ mH
    res["H_mean"], res["H_sd"] = m, np.sqrt(w @ vH + w @ (mH - m) ** 2)
    res["surf_lat"] = (w @ surf).reshape(n_surf, n_surf)  # rows: l2, columns: l1
    res["gx_mean"] = w @ gx
    pick = rng.choice(S, size=n_draws, p=w)  # joint posterior draws
    res["L1_draws"], res["L2_draws"] = (
        draws["L1"][pick, :n_grid],
        draws["L2"][pick, :n_grid],
    )
    return dict(
        x=xg,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        xH=xH,
        yH=yH,
        surf_l1=l1g,
        surf_l2=l2g,
        ess=1 / (w**2).sum(),
        **res,
        **{f"hp_L1_{k}": v for k, v in hp1.items()},
        **{f"hp_L2_{k}": v for k, v in hp2.items()},
        **{f"hp_H_{k}": v for k, v in hpH.items()},
    )
