# PACKAGE IMPORTS

# System packages
import os
import numpy as np
import pandas as pd

# Statistical packages
from scipy.stats import spearmanr, brunnermunzel, rankdata
import pingouin as pg
from statsmodels.stats.multitest import multipletests

# CUSTOM FUNCTIONS

def naneval(imagespath, dataset):
    """
    Function to evaluate the presence of a NaN as the AMD score reported on the dataset and the image of the same visit (simplified evaluation, reporting just the patient and the visit).
        Inputs:  - [imagespath: str]: The path to the image dataset directory.
                 - [dataset: pd.DataFrame]: The dataframe containing the AMD scores of the patients, with the column "ID2" containing the patient IDs and the columns "SCALE_{visit}" containing the AMD score for each visit.
        Outputs: - [nans: pd.DataFrame]: A dataframe containing the info about the patients with NaN values in the dataset, reporting the ID of the patient and visit.
    """
    nans = pd.DataFrame(columns=["Patient", "Visit"])
    for patient in os.listdir(imagespath):
        for viewtype in ["F2_LS", "F2_RS"]:
            for eyetype in ["LE", "RE"]:
                folder = os.path.join(imagespath, patient, viewtype, eyetype)
                if not os.path.exists(folder):
                    continue
                visits = sorted(os.listdir(folder))
                for visit in visits:
                    visit_num = str(int(visit)) if visit.isdigit() else visit
                    column_name = f"SCALE_{visit_num}"
                    if column_name in dataset.columns:
                        patient_data = dataset[dataset["ID2"] == patient]
                        if not patient_data.empty and patient_data[column_name].isna().any():
                            already_added = (
                                (nans["Patient"] == patient) &
                                (nans["Visit"] == visit)
                            ).any()
                            if not already_added:
                                nans.loc[len(nans)] = {
                                    "Patient": patient,
                                    "Visit": visit
                                }
    return nans

def correlationanalysis(latents, labelscompare, method):
    if method == 'Spearman':
        corr, pvalue = spearmanr(latents, labelscompare)
    elif method in ('Mann-Whitney', 'Brunner-Munzel'):
        numclasses = len(np.unique(labelscompare))
        if numclasses == 2:
            group0 = latents[np.array(labelscompare) == 0]
            group1 = latents[np.array(labelscompare) == 1]
        elif numclasses == 3:
            group0 = latents[np.array(labelscompare) != 2]
            group1 = latents[np.array(labelscompare) == 2]
        else: raise ValueError("Mann-Whitney/Brunner-Munzel test is only applicable for 2 or 3 binarized classes.")
        # The Brunner-Munzel point estimator is algebraically identical to pingouin's RBC
        # (both are rank-based relative-effect measures); the two tests differ only in
        # which null variance they use to turn that same effect size into a p-value, so
        # the effect size is still taken from pg.mwu and only the p-value swaps out for
        # scipy's Brunner-Munzel, which does not assume equal variance between groups.
        res = pg.mwu(group0, group1)
        corr = res.RBC.to_numpy()[0]
        if method == 'Mann-Whitney':
            pvalue = res.p_val.to_numpy()[0]
        else:
            pvalue = brunnermunzel(group0, group1).pvalue
    return corr, pvalue

def fastspearman(x, y):
    """
    Spearman's rho computed directly from ranks, numerically identical to
    scipy.stats.spearmanr but cheap enough to call inside a bootstrap loop of thousands
    of replicates (where correlationanalysis's per-call scipy/pingouin overhead dominates).
        Inputs:  - [x, y: array-like]: The two variables to correlate.
        Outputs: - [rho: float]: Spearman's rank correlation coefficient.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 3:
        return np.nan
    rx = rankdata(x); rx = rx - rx.mean()
    ry = rankdata(y); ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom != 0 else np.nan

def fastrankbiserial(values, group):
    """
    Rank-biserial correlation for group 0 vs group 1, numerically identical to the RBC
    pingouin.mwu returns (and thus to correlationanalysis's 'Mann-Whitney'/'Brunner-Munzel'
    effect size), but cheap enough to call inside a bootstrap loop. Positive means values
    are higher in the group coded 0, matching correlationanalysis's sign convention.
        Inputs:  - [values: array-like]: The variable being compared.
                 - [group: array-like]: The binary group code (0/1).
        Outputs: - [rbc: float]: The rank-biserial correlation.
    """
    group = np.asarray(group)
    group0 = group == 0
    group1 = ~group0
    nx, ny = int(group0.sum()), int(group1.sum())
    if nx == 0 or ny == 0:
        return np.nan
    r = rankdata(np.asarray(values, dtype=float))
    ux = r[group0].sum() - nx * (nx + 1) / 2.0
    return float(2.0 * ux / (nx * ny) - 1.0)

def patientbootstrapindex(patientids, rng):
    """
    One patient-cluster bootstrap resample: patients (not individual rows) are drawn with
    replacement, and every row of a drawn patient is included, duplicated if that patient
    is drawn more than once. Accounts for visits from the same patient being correlated,
    which a plain row-level bootstrap (or a test that treats every row as independent)
    understates.
        Inputs:  - [patientids: array-like]: The patient ID for every row.
                 - [rng: np.random.Generator]: The random generator driving the resample.
        Outputs: - [rowindices: np.ndarray]: Row indices into the original arrays for one bootstrap replicate.
    """
    patientids = np.asarray(patientids)
    uniqueids, inverse = np.unique(patientids, return_inverse=True)
    rowsbypatient = [np.flatnonzero(inverse == k) for k in range(len(uniqueids))]
    drawn = rng.integers(0, len(uniqueids), size=len(uniqueids))
    return np.concatenate([rowsbypatient[k] for k in drawn])

def bootstrapci(replicates, alpha=0.05):
    """
    95% (or 1-alpha) percentile confidence interval and a two-sided bootstrap p-value from
    a set of bootstrap replicate statistic values (e.g. correlationanalysis's effect size,
    recomputed once per patientbootstrapindex resample).
        Inputs:  - [replicates: array-like]: The statistic recomputed on each bootstrap resample.
                 - [alpha: float]: The two-sided significance level for the interval (default 0.05).
        Outputs: - [low, high, pvalue, n: tuple]: The percentile CI bounds, the bootstrap p-value (2 x the smaller tail past 0, floored at 1/n), and the number of finite replicates used.
    """
    r = np.asarray(replicates, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan, np.nan, np.nan, 0
    low, high = np.percentile(r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    plow = np.mean(r <= 0)
    phigh = np.mean(r >= 0)
    pvalue = min(1.0, 2 * min(plow, phigh))
    pvalue = max(pvalue, 1.0 / r.size)
    return low, high, pvalue, r.size

def bhfdr(pvalues, alpha=0.05):
    """
    Benjamini-Hochberg false discovery rate correction over a family of p-values (e.g. one
    latent dimension against every phenotype), via statsmodels.
        Inputs:  - [pvalues: array-like]: The p-values to correct, may contain NaNs.
                 - [alpha: float]: The target false discovery rate (default 0.05).
        Outputs: - [reject, adjusted: tuple of np.ndarray]: Boolean rejection decisions and BH-adjusted p-values, aligned with the input (NaNs preserved).
    """
    p = np.asarray(pvalues, dtype=float)
    ok = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    adjusted = np.full(p.shape, np.nan)
    if ok.any():
        rej, padj, _, _ = multipletests(p[ok], alpha=alpha, method="fdr_bh")
        reject[ok] = rej
        adjusted[ok] = padj
    return reject, adjusted