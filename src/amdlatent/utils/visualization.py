# PACKAGE IMPORTS

# System packages
import os
import numpy as np
from pathlib import Path
import pandas as pd

# Jupyter packages
import ipywidgets as widgets
from IPython.display import display

# Plotting packages
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Classification packages
import pickle as pkl
import torch
from sklearn.preprocessing import normalize
from amdlatent.models.evaluation import predextr

# CUSTOM FUNCTIONS

def classcentroids(latents, labels):
    """
    Per-class centroid of the L2-normalised latent space: each class's mean of its own
    unit-length rows, itself renormalised to unit length. The centroid-distance half of the
    "Distances" figure (inter-class separation, see interclassdistances).
        Inputs:  - [latents: np.ndarray]: The latent vectors, shape (n_samples, n_dims).
                 - [labels: array-like]: The class label of each row.
        Outputs: - [classes, centroids: tuple]: The sorted unique class labels and their centroids, shape (n_classes, n_dims).
    """
    L = normalize(latents)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    centroids = np.zeros((len(classes), L.shape[1]))
    for k, c in enumerate(classes):
        centroids[k] = L[labels == c].mean(axis=0)
    centroids = normalize(centroids)
    return classes, centroids

def interclassdistances(centroids):
    """
    Pairwise cosine distance between class centroids (for unit-length vectors, 1 minus the
    dot product), the inter-class separation half of the "Distances" figure.
        Inputs:  - [centroids: np.ndarray]: Unit-length class centroids, as returned by classcentroids.
        Outputs: - [distances: np.ndarray]: Symmetric matrix of cosine distances, shape (n_classes, n_classes).
    """
    return 1.0 - centroids @ centroids.T

def intraclassvariability(latents, labels, classes, centroids):
    """
    Intra-class spread around each class's own centroid, in cosine distance: the intra-class
    half of the "Distances" figure (its bar height is this function's "mean" reading).
    Reported as three readings side by side, since a Fisher-style ratio needs a variance-like
    denominator but the figure itself only defines the mean: the raw mean distance to
    centroid, that value squared, and the true statistical variance of the distances.
        Inputs:  - [latents: np.ndarray]: The latent vectors, shape (n_samples, n_dims).
                 - [labels: array-like]: The class label of each row.
                 - [classes: np.ndarray]: The class labels, in the order returned by classcentroids.
                 - [centroids: np.ndarray]: The class centroids, in the same order, as returned by classcentroids.
        Outputs: - [variability: dict]: {"mean": ..., "mean_sq": ..., "var": ...}, each an np.ndarray of shape (n_classes,).
    """
    L = normalize(latents)
    labels = np.asarray(labels)
    proj = L @ centroids.T
    variability = {"mean": np.full(len(classes), np.nan), "mean_sq": np.full(len(classes), np.nan), "var": np.full(len(classes), np.nan)}
    for k, c in enumerate(classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        d = 1.0 - proj[mask, k]
        mean = float(d.mean())
        variability["mean"][k] = mean
        variability["mean_sq"][k] = mean ** 2
        variability["var"][k] = float(d.var())
    return variability

def discriminabilityratio(interclass, intraclass):
    """
    Fisher-like discriminability ratio between every pair of classes, combining the two
    halves of the "Distances" figure into one number: how far apart two classes' centroids
    are, relative to how spread out each class is internally (FDR_ij = d_ij^2 / (v_i + v_j)).
    Larger means the pair is more cleanly separated relative to its own internal spread.
        Inputs:  - [interclass: np.ndarray]: Pairwise inter-class cosine distances, as returned by interclassdistances.
                 - [intraclass: np.ndarray]: One intra-class variability reading (e.g. intraclassvariability(...)["var"]).
        Outputs: - [fdr: np.ndarray]: FDR_ij, shape (n_classes, n_classes), NaN on the diagonal and wherever the denominator is 0.
    """
    K = len(intraclass)
    fdr = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            if i != j and np.isfinite(interclass[i, j]) and (intraclass[i] + intraclass[j]) > 0:
                fdr[i, j] = interclass[i, j] ** 2 / (intraclass[i] + intraclass[j])
    return fdr

def distancesbootstrap(latents, labels, patientids, variant="var", n_replicates=2000, alpha=0.05, randomstate=None):
    """
    Patient-cluster bootstrap uncertainty for the "Distances" analysis: resamples patients
    (not rows) with replacement via audit.patientbootstrapindex and recomputes
    classcentroids/interclassdistances/intraclassvariability/discriminabilityratio on each
    replicate, summarizing every cell with audit.bootstrapci. Accounts for the fact that
    several rows come from the same patient, which a plain row-level spread (e.g. the
    intraclassvariability "var" reading alone) does not. A replicate that loses an entire
    class is skipped (left as NaN), which bootstrapci's NaN handling already tolerates.
        Inputs:  - [latents, labels]: as classcentroids.
                 - [patientids: array-like]: the patient ID for every row, for cluster resampling.
                 - [variant: str]: which intraclassvariability reading to use for the FDR ratio ("mean", "mean_sq" or "var").
                 - [n_replicates: int]: number of patient-cluster bootstrap replicates.
                 - [alpha: float]: two-sided CI level.
                 - [randomstate: int, optional]: seed for the resampling.
        Outputs: - [result: dict]: point estimates ("classes", "centroids", "interclass", "intraclass", "fdr") plus "interclass_ci"/"fdr_ci" (arrays of (low, high) pairs, shape (n_classes, n_classes, 2)) and "intraclass_ci" (a dict keyed like intraclass, each value shape (n_classes, 2) — every reading gets a CI from the same bootstrap pass, not just ``variant``).
    """
    from amdlatent.dataset.audit import patientbootstrapindex, bootstrapci
    labels = np.asarray(labels)
    patientids = np.asarray(patientids)
    classes, centroids = classcentroids(latents, labels)
    interclass = interclassdistances(centroids)
    intraclass = intraclassvariability(latents, labels, classes, centroids)
    fdr = discriminabilityratio(interclass, intraclass[variant])

    K = len(classes)
    variants = list(intraclass.keys())
    rng = np.random.default_rng(randomstate)
    interreps = np.full((n_replicates, K, K), np.nan)
    intrareps = {v: np.full((n_replicates, K), np.nan) for v in variants}
    fdrreps = np.full((n_replicates, K, K), np.nan)
    for b in range(n_replicates):
        idx = patientbootstrapindex(patientids, rng)
        Lb, labelsb = latents[idx], labels[idx]
        if not set(classes).issubset(set(np.unique(labelsb))):
            continue
        classesb, centroidsb = classcentroids(Lb, labelsb)
        interb = interclassdistances(centroidsb)
        intrab = intraclassvariability(Lb, labelsb, classesb, centroidsb)
        interreps[b] = interb
        for v in variants:
            intrareps[v][b] = intrab[v]
        fdrreps[b] = discriminabilityratio(interb, intrab[variant])

    interci = np.full((K, K, 2), np.nan)
    fdrci = np.full((K, K, 2), np.nan)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            low, high, _, _ = bootstrapci(interreps[:, i, j], alpha=alpha)
            interci[i, j] = [low, high]
            low, high, _, _ = bootstrapci(fdrreps[:, i, j], alpha=alpha)
            fdrci[i, j] = [low, high]
    intraci = {v: np.full((K, 2), np.nan) for v in variants}
    for v in variants:
        for k in range(K):
            low, high, _, _ = bootstrapci(intrareps[v][:, k], alpha=alpha)
            intraci[v][k] = [low, high]

    return {
        "classes": classes,
        "centroids": centroids,
        "interclass": interclass,
        "intraclass": intraclass,
        "fdr": fdr,
        "interclass_ci": interci,
        "intraclass_ci": intraci,
        "fdr_ci": fdrci,
    }

def mahalanobisdistances(latents, labels, classes):
    """
    Squared Mahalanobis distance between class means, normalised by the pooled (sample-size
    weighted) within-class covariance — a genuine multivariate Fisher-ratio analogue to
    discriminabilityratio, using the full covariance structure instead of a scalar spread.
    The raw per-class sample covariance is singular here (every AMD-severity class has at
    most a few hundred rows in a 1024-d embedding), so it is regularised with Ledoit-Wolf
    shrinkage before pooling and inverting — the covariance ESTIMATOR is regularised, not
    the embedding itself (no dimensionality reduction is applied anywhere).
        Inputs:  - [latents: np.ndarray]: The latent vectors, shape (n_samples, n_dims).
                 - [labels: array-like]: The class label of each row.
                 - [classes: np.ndarray]: The class labels and their order, as returned by classcentroids.
        Outputs: - [maha: np.ndarray]: Squared Mahalanobis distance, shape (n_classes, n_classes), NaN on the diagonal.
    """
    from sklearn.covariance import ledoit_wolf
    labels = np.asarray(labels)
    K = len(classes)
    means, covs, counts = [], [], []
    for c in classes:
        Xc = latents[labels == c]
        means.append(Xc.mean(axis=0))
        cov, _ = ledoit_wolf(Xc)
        covs.append(cov)
        counts.append(Xc.shape[0])
    total = sum(counts)
    pooled = sum(cov * (n / total) for cov, n in zip(covs, counts))
    prec = np.linalg.inv(pooled)
    maha = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            diff = means[i] - means[j]
            maha[i, j] = float(diff @ prec @ diff)
    return maha

def frechetdistances(latents, labels, classes):
    """
    Squared (Bures-)Frechet distance between per-class Gaussians fit with Ledoit-Wolf
    shrinkage covariances: d_F^2 = ||mu_i-mu_j||^2 + Tr(Sigma_i+Sigma_j-2(Sigma_i Sigma_j)^(1/2)).
    Combines mean-shift and covariance mismatch in one number, unlike discriminabilityratio
    or mahalanobisdistances: two classes with well-separated centroids but heavily
    overlapping covariance ellipsoids score as less separated here than by centroid
    distance alone.
        Inputs:  - as mahalanobisdistances.
        Outputs: - [frechet: np.ndarray]: Squared Frechet distance, shape (n_classes, n_classes), NaN on the diagonal.
    """
    from sklearn.covariance import ledoit_wolf
    labels = np.asarray(labels)
    K = len(classes)
    means, covs = [], []
    for c in classes:
        Xc = latents[labels == c]
        means.append(Xc.mean(axis=0))
        cov, _ = ledoit_wolf(Xc)
        covs.append(cov)

    def sqrtpsd(sigma):
        w, V = np.linalg.eigh(sigma)
        w = np.clip(w, 0, None)
        return (V * np.sqrt(w)) @ V.T

    roots = [sqrtpsd(cov) for cov in covs]
    frechet = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(i + 1, K):
            B = roots[i] @ covs[j] @ roots[i]
            w = np.clip(np.linalg.eigvalsh(B), 0, None)
            trsqrt = np.sqrt(w).sum()
            diff = means[i] - means[j]
            d2 = max(float(diff @ diff + np.trace(covs[i]) + np.trace(covs[j]) - 2 * trsqrt), 0.0)
            frechet[i, j] = frechet[j, i] = d2
    return frechet

def _mmdsqdist(A, B):
    out = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.maximum(out, 0, out=out)

def mmdgamma(latents, sample=2000, randomstate=None):
    """
    RBF kernel bandwidth for mmddistances/mmdpermutation, via the median heuristic on a
    random subsample, so every pair/replicate is scored under the same, data-driven kernel.
        Inputs:  - [latents: np.ndarray]: The latent vectors to estimate the bandwidth from.
                 - [sample: int]: How many rows to subsample for the pairwise-distance median.
                 - [randomstate: int, optional]: Seed for the subsample draw.
        Outputs: - [gamma: float]: The RBF kernel bandwidth (1 / (2 * median squared distance)).
    """
    rng = np.random.default_rng(randomstate)
    idx = rng.choice(latents.shape[0], size=min(sample, latents.shape[0]), replace=False)
    X = latents[idx]
    sqd = _mmdsqdist(X, X)
    med = np.median(sqd[np.triu_indices_from(sqd, k=1)])
    return 1.0 / (2 * med) if med > 0 else 1.0

def _mmd2(X, Y, gamma):
    Kxx = np.exp(-gamma * _mmdsqdist(X, X)); np.fill_diagonal(Kxx, 0)
    Kyy = np.exp(-gamma * _mmdsqdist(Y, Y)); np.fill_diagonal(Kyy, 0)
    Kxy = np.exp(-gamma * _mmdsqdist(X, Y))
    m, n = X.shape[0], Y.shape[0]
    return float(Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.sum() / (m * n))

def mmddistances(latents, labels, classes, gamma=None):
    """
    Pairwise unbiased RBF-kernel MMD^2 between classes — a distribution-free separation
    measure with no covariance estimation at all, complementing the Gaussian-based
    mahalanobisdistances/frechetdistances above.
        Inputs:  - [latents, labels, classes]: as mahalanobisdistances.
                 - [gamma: float, optional]: RBF bandwidth; computed via mmdgamma(latents) if not given.
        Outputs: - [mmd, gamma: tuple]: The MMD^2 matrix, shape (n_classes, n_classes), and the bandwidth used.
    """
    labels = np.asarray(labels)
    if gamma is None:
        gamma = mmdgamma(latents)
    K = len(classes)
    mmd = np.full((K, K), np.nan)
    for i in range(K):
        for j in range(i + 1, K):
            v = _mmd2(latents[labels == classes[i]], latents[labels == classes[j]], gamma)
            mmd[i, j] = mmd[j, i] = v
    return mmd, gamma

def mmdpermutation(X, Y, gamma, n_permutations=2000, randomstate=None):
    """
    Row-level permutation null for MMD^2: answers whether two classes' distributions are
    distinguishable from chance, which a (non-negative) bootstrap CI on MMD cannot — unlike
    the patient-cluster bootstrap, this pools and reshuffles individual visits, so it is a
    null-significance check rather than an account of patient clustering.
        Inputs:  - [X, Y: np.ndarray]: The two classes' rows to compare.
                 - [gamma: float]: The RBF bandwidth (from mmdgamma/mmddistances).
                 - [n_permutations: int]: Number of row-level permutations.
                 - [randomstate: int, optional]: Seed for the permutation draws.
        Outputs: - [observed, pvalue: tuple]: The observed MMD^2 and the one-sided permutation p-value.
    """
    rng = np.random.default_rng(randomstate)
    pooled = np.vstack([X, Y])
    m, n = X.shape[0], Y.shape[0]
    observed = _mmd2(X, Y, gamma)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        idx = rng.permutation(m + n)
        null[b] = _mmd2(pooled[idx[:m]], pooled[idx[m:]], gamma)
    pvalue = (1 + int((null >= observed).sum())) / (1 + n_permutations)
    return observed, pvalue

def extendedseparation(latents, labels, patientids, n_replicates=200, n_permutations=2000, alpha=0.05, randomstate=None):
    """
    Mahalanobis, Frechet and MMD pairwise class separation with patient-cluster bootstrap
    CIs (all three from the SAME resampled replicate each iteration, for consistency with
    discriminabilityratio's cosine-based analysis), plus an MMD row-level permutation test
    with BH-FDR across all pairs, answering "distinguishable from chance" — a question a
    bootstrap CI on a non-negative distance cannot answer on its own. Frechet's per-replicate
    cost (an eigendecomposition per class pair) dominates runtime, so n_replicates is shared
    by all three and should be kept modest for interactive use.
        Inputs:  - [latents, labels]: as classcentroids.
                 - [patientids: array-like]: the patient ID for every row, for cluster resampling.
                 - [n_replicates: int]: patient-cluster bootstrap replicates, shared by all three metrics.
                 - [n_permutations: int]: row-level permutations for the MMD null, per pair.
                 - [alpha: float]: two-sided CI level.
                 - [randomstate: int, optional]: seed.
        Outputs: - [result: dict]: "classes", "gamma", point-estimate "mahalanobis"/"frechet"/"mmd" matrices, "<name>_ci" (low, high) arrays, and "mmd_pvalue"/"mmd_pvalue_bh" pairwise p-value matrices.
    """
    from amdlatent.dataset.audit import patientbootstrapindex, bootstrapci, bhfdr
    labels = np.asarray(labels)
    patientids = np.asarray(patientids)
    classes = np.unique(labels)
    K = len(classes)

    maha0 = mahalanobisdistances(latents, labels, classes)
    frechet0 = frechetdistances(latents, labels, classes)
    gamma = mmdgamma(latents, randomstate=randomstate)
    mmd0, _ = mmddistances(latents, labels, classes, gamma=gamma)

    rng = np.random.default_rng(randomstate)
    maharep = np.full((n_replicates, K, K), np.nan)
    frechetrep = np.full((n_replicates, K, K), np.nan)
    mmdrep = np.full((n_replicates, K, K), np.nan)
    for b in range(n_replicates):
        idx = patientbootstrapindex(patientids, rng)
        Lb, labelsb = latents[idx], labels[idx]
        if not set(classes).issubset(set(np.unique(labelsb))):
            continue
        try:
            maharep[b] = mahalanobisdistances(Lb, labelsb, classes)
        except np.linalg.LinAlgError:
            pass
        frechetrep[b] = frechetdistances(Lb, labelsb, classes)
        mmdb, _ = mmddistances(Lb, labelsb, classes, gamma=gamma)
        mmdrep[b] = mmdb

    def cigrid(reps):
        lo = np.full((K, K), np.nan); hi = np.full((K, K), np.nan)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                low, high, _, _ = bootstrapci(reps[:, i, j], alpha=alpha)
                lo[i, j], hi[i, j] = low, high
        return lo, hi

    maha_lo, maha_hi = cigrid(maharep)
    frechet_lo, frechet_hi = cigrid(frechetrep)
    mmd_lo, mmd_hi = cigrid(mmdrep)

    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    flatp = []
    for i, j in pairs:
        _, p = mmdpermutation(latents[labels == classes[i]], latents[labels == classes[j]], gamma, n_permutations, randomstate)
        flatp.append(p)
    _, adjusted = bhfdr(flatp)
    mmdpvals = np.full((K, K), np.nan)
    mmdpvalsbh = np.full((K, K), np.nan)
    for (i, j), p, padj in zip(pairs, flatp, adjusted):
        mmdpvals[i, j] = mmdpvals[j, i] = p
        mmdpvalsbh[i, j] = mmdpvalsbh[j, i] = padj

    return {
        "classes": classes,
        "gamma": gamma,
        "mahalanobis": maha0, "mahalanobis_ci": (maha_lo, maha_hi),
        "frechet": frechet0, "frechet_ci": (frechet_lo, frechet_hi),
        "mmd": mmd0, "mmd_ci": (mmd_lo, mmd_hi),
        "mmd_pvalue": mmdpvals, "mmd_pvalue_bh": mmdpvalsbh,
    }

def amdscorelabeler(pathlist, dataset, additional = None):
    amdscale = [dataset.loc[dataset['ID2'].astype(str) == str(Path(p).parts[-5]), f"SCALE_{int(str(Path(p).parts[-2]).strip())}"].iloc[0] for p in pathlist]
    if additional is None or additional == 'All':
        labels = np.array(['AMD 0' if a==0 else 'AMD 1' if a==1 else 'AMD 2' if a==2 else 'AMD 3' if a==3 else 'AMD 4' if a==4 else 'AMD 5' if a==5 else 'AMD NaN' for a in amdscale])
        cmap = {"AMD 0": "darkgreen", "AMD 1": "limegreen", "AMD 2": "greenyellow", "AMD 3": "yellow", "AMD 4": "orange", "AMD 5": "red", "AMD NaN": "gray"}
        return labels, cmap
    elif additional == 'Binary No':
        labels = np.array(['No AMD' if a == 0 else 'Intermediate / Advanced' if a in [1, 2, 3, 4, 5] else 'AMD NaN' for a in amdscale])
        cmap = {"No AMD": "green", "Intermediate / Advanced": "red", "AMD NaN": "gray"}
        return labels, cmap
    elif additional == 'Binary Late':
        labels = np.array(['No AMD / Early stage' if a in [0, 1, 2, 3] else 'Advanced' if a in [4, 5] else 'AMD NaN' for a in amdscale])
        cmap = {"No AMD / Early stage": "green", "Advanced": "red", "AMD NaN": "gray"}
        return labels, cmap

def drusensizecategorizer(ID, visno, eyetype, fundus):
    drusensize = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}DRSZWI"].iloc[0]
    drusensize = "Small" if 0 <= drusensize <= 2 else "Medium" if drusensize == 3 else "Large"
    return drusensize

def drusenareacategorizer(ID, visno, eyetype, fundus):
    drusenarea = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}DRUSF2"].iloc[0]
    drusenarea = "Large" if drusenarea == 2 else "Small"
    return drusenarea

def drusencategorizer(ID, visno, eyetype, fundus):
    drusensize = drusensizecategorizer(ID, visno, eyetype, fundus)
    drusenarea = drusenareacategorizer(ID, visno, eyetype, fundus)
    drusenscore = drusensize if drusenarea != "Large" else "Large"
    return drusenscore

def drusenlabeler(pathlist, fundus):
    drusen = []
    for p in pathlist:
        drusenscoreL = drusencategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "LE", fundus)
        drusenscoreR = drusencategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "RE", fundus)
        if drusenscoreL == "Large" and drusenscoreR == "Large":
            drusenscore = "Both"
        elif (drusenscoreL == "Large" and drusenscoreR in ["Small", "Medium", "NaN"]) or (drusenscoreR == "Large" and drusenscoreL in ["Small", "Medium", "NaN"]):
            drusenscore = "One eye"
        else: drusenscore = "None"
        drusen.append(drusenscore)
    cmap = {"None": "lightskyblue", "One eye": "cornflowerblue", "Both": "royalblue"}
    return np.array(drusen), cmap

def increasedpigmentcategorizer(ID, visno, eyetype, fundus):
    increasedpigment = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}INCPWI"].iloc[0]
    increasedpigment = "No" if increasedpigment == 0 else "Yes"
    return increasedpigment

def depigmentationcategorizer(ID, visno, eyetype, fundus):
    depigmentation = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}RPEDWI"].iloc[0]
    depigmentation = "No" if depigmentation == 0 else "Yes"
    return depigmentation

def pigmentarycategorizer(ID, visno, eyetype, fundus):
    increasedpigment = increasedpigmentcategorizer(ID, visno, eyetype, fundus)
    depigmentation = depigmentationcategorizer(ID, visno, eyetype, fundus)
    pigmentscore = "Yes" if increasedpigment == "Yes" or depigmentation == "Yes" else "No"
    return pigmentscore

def pigmentarylabeler(pathlist, fundus):
    pigment = []
    for p in pathlist:
        pigmentL = pigmentarycategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "LE", fundus)
        pigmentR = pigmentarycategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "RE", fundus)
        if pigmentL == "Yes" and pigmentR == "Yes":
            pigmentscore = "Both"
        elif (pigmentL == "Yes" and pigmentR == "No") or (pigmentR == "Yes" and pigmentL == "No"):
            pigmentscore = "One eye"
        else: pigmentscore = "None"
        pigment.append(pigmentscore)
    cmap = {"None": "violet", "One eye": "mediumorchid", "Both": "blueviolet"}
    return np.array(pigment), cmap

def geographiccentercategorizer(ID, visno, eyetype, fundus):
    geographiccenter = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}GEOACT"].iloc[0]
    geographiccenter = "Yes" if geographiccenter == 2 else "No"
    return geographiccenter

def geographicareacategorizer(ID, visno, eyetype, fundus):
    geographicarea = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}GEOACS"].iloc[0]
    geographicarea = "Yes" if geographicarea >= 2 and geographicarea <= 4 else "No"
    return geographicarea

def geographicgridcategorizer(ID, visno, eyetype, fundus):
    geographicgrid = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}GEOAWI"].iloc[0]
    geographicgrid = "Yes" if geographicgrid >= 2 and geographicgrid <= 7 else "No"
    return geographicgrid

def geographiccategorizer(ID, visno, eyetype, fundus):
    geographiccenter = geographiccentercategorizer(ID, visno, eyetype, fundus)
    geographicarea = geographicareacategorizer(ID, visno, eyetype, fundus)
    geographicgrid = geographicgridcategorizer(ID, visno, eyetype, fundus)
    geographicscore = "Yes" if geographiccenter == "Yes" or geographicarea == "Yes" or geographicgrid == "Yes" else "No"
    return geographicscore

def geographiclabeler(pathlist, fundus):
    geographic = []
    for p in pathlist:
        geoL = geographiccategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "LE", fundus)
        geoR = geographiccategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "RE", fundus)
        if geoL == "Yes" and geoR == "Yes":
            geoscore = "At least one eye"
        elif (geoL == "Yes" and geoR == "No") or (geoR == "Yes" and geoL == "No"):
            geoscore = "At least one eye"
        else: geoscore = "None"
        geographic.append(geoscore)
    cmap = {"None": "lightpink", "At least one eye": "deeppink"}
    return np.array(geographic), cmap

def hemorrhagiccategorizer(ID, visno, eyetype, fundus):
    hemorrhagic = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}SSRF2"].iloc[0]
    hemorrhagic = "Yes" if hemorrhagic == 2 else "No"
    return hemorrhagic

def exudatecategorizer(ID, visno, eyetype, fundus):
    exudate = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}HDEXF2"].iloc[0]
    exudate = "Yes" if exudate == 2 else "No"
    return exudate

def subhemorragecategorizer(ID, visno, eyetype, fundus):
    subhemorrage = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}SUBHF2"].iloc[0]
    subhemorrage = "Yes" if subhemorrage == 2 else "No"
    return subhemorrage

def fibrosiscategorizer(ID, visno, eyetype, fundus):
    fibrosis = fundus.loc[(fundus["ID2"].astype(str) == ID) & (fundus["VISNO"] == visno), f"{eyetype}SUBFF2"].iloc[0]
    fibrosis = "Yes" if fibrosis == 2 else "No"
    return fibrosis

def neovascularcategorizer(ID, visno, eyetype, fundus):
    hemorragic = hemorrhagiccategorizer(ID, visno, eyetype, fundus)
    exudate = exudatecategorizer(ID, visno, eyetype, fundus)
    subhemorrage = subhemorragecategorizer(ID, visno, eyetype, fundus)
    fibrosis = fibrosiscategorizer(ID, visno, eyetype, fundus)
    neovascular = "Yes" if hemorragic == "Yes" or exudate == "Yes" or subhemorrage == "Yes" or fibrosis == "Yes" else "No"
    return neovascular

def neovascularlabeler(pathlist, fundus):
    neovascular = []
    for p in pathlist:
        neoL = neovascularcategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "LE", fundus)
        neoR = neovascularcategorizer(str(Path(p).parts[-5]), int(str(Path(p).parts[-2]).strip()), "RE", fundus)
        if neoL == "Yes" and neoR == "Yes":
            neovascularscore = "At least one eye"
        elif (neoL == "Yes" and neoR == "No") or (neoR == "Yes" and neoL == "No"):
            neovascularscore = "At least one eye"
        else: neovascularscore = "None"
        neovascular.append(neovascularscore)
    cmap = {"None": "lightseagreen", "At least one eye": "teal"}
    return np.array(neovascular), cmap

def agelabeler(pathlist, enrollment):
    baselineage = [enrollment.loc[enrollment['ID2'].astype(str) == str(Path(path).parts[-5]), 'ENROLLAGE'].iloc[0] for path in pathlist]
    imagesage = [np.round(baselineage[i] + 0.5 * int(str(Path(path).parts[-2].strip()))) for i, path in enumerate(pathlist)]
    imagesage = np.array(imagesage).astype(int)
    return imagesage, None

def sexlabeler(pathlist, enrollment):
    sex = [enrollment.loc[enrollment['ID2'].astype(str) == str(Path(path).parts[-5]), 'SEX'].iloc[0] for path in pathlist]
    cmap = {"M": "blue", "F": "pink"}
    return np.array(sex), cmap

def bmilabeler(pathlist, enrollment):
    bmi = [enrollment.loc[enrollment['ID2'].astype(str) == str(Path(path).parts[-5]), 'BMI_R'].iloc[0] for path in pathlist]
    bmi = np.array(bmi).astype(int)
    bmi = np.array(['Severe Underweight' if x < 16 else 'Moderate Underweight' if 16 <= x < 17 else 'Mild Underweight' if 17 <= x < 18.5 else 'Normal' if 18.5 <= x < 25 else 'Overweight' if 25 <= x < 30 else 'Obese I' if 30 <= x < 35 else 'Obese II' if 35 <= x < 40 else 'Severe Obese' for x in bmi])
    cmap = {"Severe Underweight": "lightblue", "Moderate Underweight": "deepskyblue", "Mild Underweight": "dodgerblue", "Normal": "green", "Overweight": "yellow", "Obese I": "orange", "Obese II": "orangered", "Severe Obese": "red"}
    return bmi, cmap

def glaucomalabeler(pathlist, followup):
    glaucoma = []
    for p in pathlist:
        id2 = str(Path(p).parts[-5])
        visno = int(str(Path(p).parts[-2]).strip())
        glaucomaval = followup.loc[(followup["ID2"].astype(str) == id2) & (followup["VISNO"] == visno), "GLAUCOMA"]
        if visno == 0 or glaucomaval.empty or pd.isna(glaucomaval.iloc[0]):
            currglaucoma = "N/A"
        else:
            currglaucoma = glaucomaval.iloc[0]
        glaucoma.append(currglaucoma)
    glaucoma = ["Yes" if g == "Y" else "No" if g == "N" else "N/A" for g in glaucoma]
    cmap = {"Yes": "red", "No": "green", "N/A": "lightgray"}
    return np.array(glaucoma), cmap

def diabeteslabeler(pathlist, followup):
    diabetes = []
    for p in pathlist:
        id2 = str(Path(p).parts[-5])
        visno = int(str(Path(p).parts[-2]).strip())
        diabetesval = followup.loc[(followup["ID2"].astype(str) == id2) & (followup["VISNO"] == visno), "DIABETES"]
        if visno == 0 or diabetesval.empty or pd.isna(diabetesval.iloc[0]):
            currdiabetes = "N/A"
        else:
            currdiabetes = diabetesval.iloc[0]
        diabetes.append(currdiabetes)
    diabetes = ["Yes" if g == "Y" or g == "I" or g == "P" or g == "D" else "No" if g == "N" else "N/A" for g in diabetes]
    cmap = {"Yes": "red", "No": "green", "N/A": "lightgray"}
    return np.array(diabetes), cmap

def splitlabeler(splitlabels):
    return splitlabels, None

def classifierlabeler(latents, modelinfo):
    examplepaths = None
    if isinstance(modelinfo, dict) and "path" in modelinfo:
        with open(modelinfo["path"], 'rb') as f:
            model = pkl.load(f)
            classifier = modelinfo["path"].split(os.sep)[-1]
            classifier = classifier.split("_")[0] if len(classifier.split("_")) == 2 else classifier.split("_")[0] + " " + classifier.split("_")[1]
            labels = predextr(model, latents, classifier)

    if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
    if len(np.unique(labels)) == 2: 
        labels = np.array(['No AMD / Early stage' if l == np.unique(labels)[0] else 'Intermediate / Advanced' for l in labels])
        cmap = {"No AMD / Early stage": "green", "Intermediate / Advanced": "red"}
    elif len(np.unique(labels)) == 6: 
        labels = np.array(['AMD 0' if l == np.unique(labels)[0] else 'AMD 1' if l == np.unique(labels)[1] else 'AMD 2' if l == np.unique(labels)[2] else 'AMD 3' if l == np.unique(labels)[3] else 'AMD 4' if l == np.unique(labels)[4] else 'AMD 5' for l in labels])
        cmap = {"AMD 0": "darkgreen", "AMD 1": "limegreen", "AMD 2": "greenyellow", "AMD 3": "yellow", "AMD 4": "orange", "AMD 5": "red", "AMD NaN": "gray"}
    return labels, cmap, examplepaths

def viewlabeler(pathlist):
    return np.array(['LS' if 'LS' in p else 'RS' for p in pathlist]), None

def hoverdatalabeler(pathlist):
    hoverdata = {
        'ID': [str(Path(p).parts[-5]) for p in pathlist],
        'Visit': [str(Path(p).parts[-2]) for p in pathlist],
        'View': ['LS' if 'LS' in p else 'RS' for p in pathlist],
    }
    return hoverdata

def featurelabeler(latents, feature):
    return latents[:, feature], None

def coloringextr(coloringtype, pathlist, dataset, latents, additional):
    colorings = {}
    marker_info = None
    if coloringtype == 'View': labels, cmap = viewlabeler(pathlist)
    elif coloringtype == 'Classifier': labels, cmap, _ = classifierlabeler(latents, additional)
    elif coloringtype == 'AMD Score': labels, cmap = amdscorelabeler(pathlist, dataset)
    elif coloringtype == 'Ground Truth': labels, cmap = amdscorelabeler(pathlist, dataset, additional)
    elif coloringtype == 'Set Split': labels, cmap = splitlabeler(additional)
    elif coloringtype == 'Drusen': labels, cmap = drusenlabeler(pathlist, additional)
    elif coloringtype == 'Pigment': labels, cmap = pigmentarylabeler(pathlist, additional)
    elif coloringtype == 'Glaucoma': labels, cmap = glaucomalabeler(pathlist, additional)
    elif coloringtype == 'Diabetes': labels, cmap = diabeteslabeler(pathlist, additional)
    elif coloringtype == 'Geographic Atrophy': labels, cmap = geographiclabeler(pathlist, additional)
    elif coloringtype == 'Neovascular': labels, cmap = neovascularlabeler(pathlist, additional)
    elif coloringtype == 'Feature': labels, cmap = featurelabeler(latents, additional)
    elif coloringtype == 'Age': labels, cmap = agelabeler(pathlist, additional)
    elif coloringtype == 'Sex': labels, cmap = sexlabeler(pathlist, additional)
    elif coloringtype == 'BMI': labels, cmap = bmilabeler(pathlist, additional)
    colorings[coloringtype] = (labels, cmap, marker_info)
    return colorings

def colormapping(fig, coloringnames, colorings, umap, hoverdata):
        n_dims = umap.shape[1]
        def _point_customdata(indices):
            return np.array([[hoverdata[k][i] for k in hoverdata] + [i] for i in indices], dtype=object)

        def _scatter_kwargs(x, y, z=None):
            return {"x": x, "y": y} if n_dims == 2 else {"x": x, "y": y, "z": z}

        scatter_cls = go.Scatter if n_dims == 2 else go.Scatter3d
        current_idx = 0
        trace_start = {}
        for cname in coloringnames:
            labels, cdiscmap, marker_info = colorings[cname]
            trace_start[cname] = current_idx
            if cdiscmap is not None:
                categories = list(cdiscmap.keys())
                present = [c for c in categories if c in labels]
                for cat in present:
                    mask = labels == cat
                    indices = np.where(mask)[0]
                    customdata = _point_customdata(indices)
                    marker_cfg = dict(size=12, color=cdiscmap[cat])
                    if marker_info is not None:
                        marker_cfg["symbol"] = marker_info["symbol"][mask]
                        marker_cfg["size"] = marker_info["size"][mask]
                    fig.add_trace(scatter_cls(
                        **_scatter_kwargs(umap[mask, 0], umap[mask, 1], umap[mask, 2] if n_dims == 3 else None),
                        mode='markers',
                        marker=marker_cfg,
                        name=cat,
                        legendgroup=cname,
                        legendgrouptitle_text=cname,
                        customdata=customdata,
                        hovertemplate="<br>".join([f"{k}=%{{customdata[{i}]}}" for i, k in enumerate(hoverdata.keys())]) + "<extra></extra>",
                        visible=(cname == coloringnames[0])
                    ))
                    current_idx += 1
            else:
                try:
                    numeric = labels.astype(float)
                    colorbar_title = cname
                    marker_cfg = dict(size=12, color=numeric, colorscale='BuPu', showscale=True, colorbar=dict(title=colorbar_title))
                    if marker_info is not None:
                        marker_cfg["symbol"] = marker_info["symbol"]
                        marker_cfg["size"] = marker_info["size"]
                    fig.add_trace(scatter_cls(
                        **_scatter_kwargs(umap[:, 0], umap[:, 1], umap[:, 2] if n_dims == 3 else None),
                        mode='markers',
                        marker=marker_cfg,
                        name=cname,
                        legendgroup=cname,
                        customdata=_point_customdata(np.arange(len(umap))),
                        hovertemplate="<br>".join([f"{k}=%{{customdata[{i}]}}" for i, k in enumerate(hoverdata.keys())]) + "<extra></extra>",
                        visible=(cname == coloringnames[0])
                    ))
                except (ValueError, TypeError):
                    categories = sorted(set(labels))
                    palette = px.colors.qualitative.Plotly
                    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
                    for cat in categories:
                        mask = labels == cat
                        indices = np.where(mask)[0]
                        customdata = _point_customdata(indices)
                        marker_cfg = dict(size=12, color=color_map[cat])
                        if marker_info is not None:
                            marker_cfg["symbol"] = marker_info["symbol"][mask]
                            marker_cfg["size"] = marker_info["size"][mask]
                        fig.add_trace(scatter_cls(
                            **_scatter_kwargs(umap[mask, 0], umap[mask, 1], umap[mask, 2] if n_dims == 3 else None),
                            mode='markers',
                            marker=marker_cfg,
                            name=cat,
                            legendgroup=cname,
                            legendgrouptitle_text=cname,
                            customdata=customdata,
                            hovertemplate="<br>".join([f"{k}=%{{customdata[{i}]}}" for i, k in enumerate(hoverdata.keys())]) + "<extra></extra>",
                            visible=(cname == coloringnames[0])
                        ))
                        current_idx += 1
                    continue
                current_idx += 1
        return fig, current_idx, trace_start

def umapdisplay(colorings, pathlist, umap, pointimagepaths, figreturn = False):
    coloringnames = list(colorings.keys())
    hoverdata = hoverdatalabeler(pathlist)
    fig = go.Figure()
    fig, total_traces, trace_start = colormapping(fig, coloringnames, colorings, umap, hoverdata)
    buttons = []
    for cname in coloringnames:
        start = trace_start[cname]
        end   = trace_start[coloringnames[coloringnames.index(cname) + 1]] if coloringnames.index(cname) + 1 < len(coloringnames) else total_traces
        visibility = [False] * total_traces
        for i in range(start, end):
            visibility[i] = True
        buttons.append(dict(
            label=cname,
            method='update',
            args=[{'visible': visibility}, {'title': f"UMAP — colored by {cname}"}]
        ))
    fig.update_layout(
        title=f"UMAP - colored by {coloringnames[0]}",
        width=1800,
        height=1200,
        legend=dict(
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            orientation="v",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            tracegroupgap=6
        ),
        margin=dict(l=80, r=360, t=80, b=60),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.01,
            xanchor="left",
            y=1.02,
            yanchor="top",
            buttons=buttons,
            showactive=True,
            bgcolor="white",
            bordercolor="gray",
        )]
    )

    def imageviewer(trace, points, selector):
        if not points.point_inds: return
        point_idx = points.point_inds[0]
        global_idx = int(trace.customdata[point_idx][-1])
        selected_paths = pointimagepaths[global_idx]

        with image_box:
            image_box.clear_output(wait=True)
            valid_paths = [str(p) for p in selected_paths if pd.notna(p)]
            if not valid_paths:
                return
            fig_img, axes = plt.subplots(1, len(valid_paths), figsize=(6 * len(valid_paths), 6))
            if len(valid_paths) == 1:
                axes = [axes]
            for ax, image_path in zip(axes, valid_paths):
                image = plt.imread(image_path)
                ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
                ax.axis('off')
                ax.set_title(Path(image_path).name)
            plt.tight_layout()
            plt.show()
    fig_widget = go.FigureWidget(fig)
    image_box  = widgets.Output()
    for trace in fig_widget.data: trace.on_click(imageviewer)
    display(widgets.VBox([fig_widget, image_box]))
    if figreturn: return fig_widget