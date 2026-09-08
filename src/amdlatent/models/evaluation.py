# PACKAGE IMPORTS

# System packages
import numpy as np
import os
import pandas as pd
from IPython.display import display

# Graphing packages
import matplotlib.pyplot as plt

# Machine Learning packages
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, roc_auc_score, mean_absolute_error
import torch
import scipy.stats as stats
import scikit_posthocs as sp

# CUSTOM FUNCTIONS

def predextr(model, X, classifier):
    if classifier.lower() == "random forest" or classifier.lower() == "adaboost" or classifier.lower() == "mlp" or classifier.lower() == "svm" or classifier.lower() == "svm linear":
        return model.predict(X)

def predproba(model, X, classifier):
    if classifier.lower() in ["random forest", "adaboost", "mlp", "svm", "svm linear"]:
        return model.predict_proba(X)[:, 1]
    return None

def evaluatemodel(model, X_train, y_train, X_val, y_val, X_test, y_test, hyper, classifier):
    preds_train = predextr(model, X_train, classifier)
    preds_val = predextr(model, X_val, classifier) if X_val is not None and y_val is not None else None
    preds_test = predextr(model, X_test, classifier)
    proba_train = predproba(model, X_train, classifier)
    proba_val = predproba(model, X_val, classifier) if preds_val is not None else None
    proba_test = predproba(model, X_test, classifier)

    currinfo = [
        classifier,
        (
            f"n_estimators={hyper}" if classifier in ["Random Forest", "AdaBoost"]
            else f"C={hyper[0]}, gamma={hyper[1]}" if classifier in ["SVM"]
            else f"C={hyper}" if classifier == "SVM Linear"
            else f"hidden_layer_sizes={hyper}" if classifier == "MLP"
            else "N/A"
        ),
        accuracy_score(y_train, preds_train),
        accuracy_score(y_val, preds_val) if preds_val is not None else np.nan,
        accuracy_score(y_test, preds_test),
        balanced_accuracy_score(y_train, preds_train),
        balanced_accuracy_score(y_val, preds_val) if preds_val is not None else np.nan,
        balanced_accuracy_score(y_test, preds_test),
    ]

    if len(np.unique(y_train)) == 2:
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, preds_train).ravel()
        tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val, preds_val).ravel() if preds_val is not None else (np.nan,) * 4
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, preds_test).ravel()
        currinfo.extend([
            tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else np.nan,
            tp_val / (tp_val + fn_val) if preds_val is not None and (tp_val + fn_val) > 0 else np.nan,
            tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else np.nan,
            tn_train / (tn_train + fp_train) if (tn_train + fp_train) > 0 else np.nan,
            tn_val / (tn_val + fp_val) if preds_val is not None and (tn_val + fp_val) > 0 else np.nan,
            tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else np.nan,
            tp_train / (tp_train + fp_train) if (tp_train + fp_train) > 0 else np.nan,
            tp_val / (tp_val + fp_val) if preds_val is not None and (tp_val + fp_val) > 0 else np.nan,
            tp_test / (tp_test + fp_test) if (tp_test + fp_test) > 0 else np.nan,
            tn_train / (tn_train + fn_train) if (tn_train + fn_train) > 0 else np.nan,
            tn_val / (tn_val + fn_val) if preds_val is not None and (tn_val + fn_val) > 0 else np.nan,
            tn_test / (tn_test + fn_test) if (tn_test + fn_test) > 0 else np.nan,
            roc_auc_score(y_train, proba_train if proba_train is not None else preds_train) if (tp_train + fn_train) > 0 else np.nan,
            roc_auc_score(y_val, proba_val if proba_val is not None else preds_val) if preds_val is not None and (tp_val + fn_val) > 0 else np.nan,
            roc_auc_score(y_test, proba_test if proba_test is not None else preds_test) if (tp_test + fn_test) > 0 else np.nan
        ])
    elif len(np.unique(y_train)) > 2:
        currinfo.extend([
            mean_absolute_error(y_train, preds_train),
            mean_absolute_error(y_val, preds_val) if preds_val is not None else np.nan,
            mean_absolute_error(y_test, preds_test),
        ])

    for preds, y_true in [(preds_train, y_train), (preds_val, y_val), (preds_test, y_test)]:
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        for cls in np.sort(np.unique(y_train)):
            currinfo.append(
                float((preds[np.array(y_true) == cls] == cls).mean()) if np.any(np.array(y_true) == cls) else np.nan
            )
    return currinfo

def classsort(v):
    return int(v) if str(v).isdigit() else str(v)

def csvloader(classtype, metrictype, subset, graphtype, splittype, encoder = "retclip"):
    subset_map = {"Training": "Train", "Validation": "Val", "Test": "Test"}.get(subset, subset)
    base = "metrics" if graphtype in ["Bars with STD", "Bars with Min-Max"] else "robustness"
    metrics_dir = os.path.join("results", "areds", "classifiers", encoder, base, splittype.split("-")[0].replace(" ", "").lower(), classtype.lower().replace(" ", ""))
    records = []

    for file in sorted(os.listdir(metrics_dir)):
        if not file.endswith(".csv"): continue

        df = pd.read_csv(os.path.join(metrics_dir, file), header=[0, 1, 2])
        if len(file.split("_")) == 3: clf_name = file.split("_")[0] + "_" + file.split("_")[1]
        else: clf_name = file.split("_")[0]

        if clf_name != "veunfrozen":
            hp_col = [col for col in df.columns if col[0] == "Hyperparameters"]
            if hp_col:
                df = df[df[hp_col[0]].duplicated(keep=False)]

        if metrictype in ["Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "ROC AUC", "MAE"]:
            metric_cols = [col for col in df.columns if col[0] == metrictype and col[1] == subset_map]
            if not metric_cols:
                continue

            values = pd.to_numeric(df[metric_cols].to_numpy().ravel(), errors="coerce")
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            if graphtype == "Robustness":
                records.append({"Classifier": clf_name, "Values": values})
            else:
                records.append({
                    "Classifier": clf_name,
                    "Mean": float(values.mean()),
                    "Std": float(values.std()),
                    "Min": float(values.min()),
                    "Max": float(values.max())
                })

        elif metrictype == "Class Accuracy":
            metric_cols = [col for col in df.columns if col[0] == metrictype and col[1] == subset_map]
            if not metric_cols:
                continue

            for col in metric_cols:
                class_id = col[2]
                values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
                if values.size == 0:
                    continue

                if graphtype == "Robustness":
                    records.append({"Classifier": clf_name, "Class": class_id, "Values": values})
                else:
                    records.append({
                        "Classifier": clf_name,
                        "Class": class_id,
                        "Mean": float(values.mean()),
                        "Std": float(values.std()),
                        "Min": float(values.min()),
                        "Max": float(values.max())
                    })

    return records

def plotbaracc(ax, metrics, graphtype, cmap):
    classifiers = list(dict.fromkeys(metrics["Classifier"]))
    x = np.arange(len(classifiers))
    colors = {clf: cmap(i % 10) for i, clf in enumerate(classifiers)}

    means, errs = [], []
    for clf in classifiers:
        row = metrics[metrics["Classifier"] == clf]
        means.append(row["Mean"].mean())
        if graphtype == "Bars with STD":
            errs.append(row["Std"].mean())
        else:
            errs.append([row["Mean"].mean() - row["Min"].mean(), row["Max"].mean() - row["Mean"].mean()])

    if graphtype == "Bars with STD":
        ax.bar(x, means, yerr=errs, capsize=4, color=[colors[c] for c in classifiers])
    else:
        ax.bar(x, means, yerr=np.array(errs).T, capsize=4, color=[colors[c] for c in classifiers])

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in classifiers],
        labels=classifiers,
        title="Classifier"
    )

def plotbarclassacc(ax, metrics, graphtype, cmap):
    classifiers = list(dict.fromkeys(metrics["Classifier"]))
    classes = sorted(metrics["Class"].unique(), key=classsort)
    x = np.arange(len(classes))
    group_width = 0.8
    bar_width = group_width / max(len(classifiers), 1)
    colors = {clf: cmap(i % 10) for i, clf in enumerate(classifiers)}

    for i, clf in enumerate(classifiers):
        means, errs = [], []
        for cls_id in classes:
            row = metrics[(metrics["Classifier"] == clf) & (metrics["Class"] == cls_id)]
            means.append(row["Mean"].mean() if not row.empty else np.nan)
            if graphtype == "Bars with STD":
                errs.append(row["Std"].mean() if not row.empty else 0.0)
            else:
                errs.append([
                    row["Mean"].mean() - row["Min"].mean() if not row.empty else 0.0,
                    row["Max"].mean() - row["Mean"].mean() if not row.empty else 0.0
                ])

        xpos = x - group_width / 2 + (i + 0.5) * bar_width
        if graphtype == "Bars with STD":
            ax.bar(xpos, means, width=bar_width * 0.9, yerr=errs, capsize=4, color=colors[clf], label=clf)
        else:
            ax.bar(xpos, means, width=bar_width * 0.9, yerr=np.array(errs).T, capsize=4, color=colors[clf], label=clf)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(title="Classifier")

def plotrobacc(ax, records, cmap):
    clf_values = {}
    for r in records:
        clf_values.setdefault(r["Classifier"], []).append(r["Values"])

    classifiers = list(clf_values.keys())
    colors = {clf: cmap(i % 10) for i, clf in enumerate(classifiers)}
    x = np.arange(len(classifiers))
    data = [np.concatenate(clf_values[clf]) for clf in classifiers]

    bp = ax.boxplot(data, positions=x, widths=0.6, patch_artist=True, showfliers=False)
    for patch, clf in zip(bp["boxes"], classifiers):
        patch.set_facecolor(colors[clf])
        patch.set_alpha(0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in classifiers],
        labels=classifiers,
        title="Classifier"
    )

def plotrobclassacc(ax, records, cmap):
    classifiers = list(dict.fromkeys([r["Classifier"] for r in records]))
    classes = sorted(list(dict.fromkeys([r["Class"] for r in records])), key=classsort)
    colors = {clf: cmap(i % 10) for i, clf in enumerate(classifiers)}

    x = np.arange(len(classes))
    group_width = 0.8
    slot_width = group_width / max(len(classifiers), 1)

    for i, clf in enumerate(classifiers):
        data, positions = [], []
        for j, cls_id in enumerate(classes):
            vals_list = [r["Values"] for r in records if r["Classifier"] == clf and r["Class"] == cls_id]
            if not vals_list:
                continue

            vals = np.concatenate(vals_list)
            if vals.size == 0:
                continue

            data.append(vals)
            positions.append(x[j] - group_width / 2 + (i + 0.5) * slot_width)

        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=slot_width * 0.8,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[clf])
                patch.set_alpha(0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in classifiers],
        labels=classifiers,
        title="Classifier"
    )

def metricsgraph(classtype, metrictype, subset, graphtype, splittype, encoder = "retclip"):
    records = csvloader(classtype, metrictype, subset, graphtype, splittype, encoder=encoder)
    if not records:
        return

    _, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("tab10")

    if graphtype in ["Bars with STD", "Bars with Min-Max"]:
        metrics = pd.DataFrame(records)
        if metrictype in ["Accuracy", "Balanced Accuracy", "MAE"]:
            plotbaracc(ax, metrics, graphtype, cmap)
        elif metrictype == "Class Accuracy":
            plotbarclassacc(ax, metrics, graphtype, cmap)

    elif graphtype == "Robustness":
        if metrictype in ["Accuracy", "Balanced Accuracy", "MAE"]:
            plotrobacc(ax, records, cmap)
        elif metrictype == "Class Accuracy":
            plotrobclassacc(ax, records, cmap)

    ax.set_title(f"{metrictype} on {subset.lower()} set - {classtype} features")
    ax.set_ylabel(metrictype)
    if metrictype != "MAE": ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def robustnesstable(classtype, subset, splittype, encoder = "retclip"):
    if classtype == "All": metric_names = ["Accuracy", "Balanced Accuracy", "MAE", "Class Accuracy"]
    else: metric_names = ["Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "ROC AUC"]
    metrics_list = []
    for metric in metric_names:
        data = csvloader(classtype, metric, subset, "Metrics", splittype, encoder=encoder)
        metrics_list.append(data)
    results = []
    for classifier_data in metrics_list[0]:
        classifier_name = classifier_data["Classifier"]
        row = {"Classifier": classifier_name}
        for i, metric in enumerate(metric_names):
            metric_data = next((item for item in metrics_list[i] if item["Classifier"] == classifier_name), None)
            if metric == "Class Accuracy":
                class_metrics = [item for item in metrics_list[i] if item["Classifier"] == classifier_name]
                for class_metric in class_metrics:
                    class_label = class_metric.get("Class", "Class")
                    row[f"{metric} ({class_label})"] = f"{class_metric['Mean']:.2f} ± {class_metric['Std']:.2f}"
            elif metric_data:
                row[metric] = f"{metric_data['Mean']:.2f} ± {metric_data['Std']:.2f}"
        results.append(row)
    results = pd.DataFrame(results)
    return results

def statisticaldiff(classtype, metric, subset, type, splittype, p_adjust = "holm", encoder = "retclip"):
    performances = csvloader(classtype, metric, subset, "Robustness", type, splittype, encoder=encoder)
    performances = {item["Classifier"]: list(item["Values"]) for item in performances}
    groups = list(performances.values())
    stat, p_value = stats.friedmanchisquare(*groups) #stats.kruskal(*groups)
    if p_value < 0.05:
        df = pd.DataFrame(performances)
        nemenyi_results = sp.posthoc_nemenyi_friedman(df)
        print(f"Significant differences found between models (Friedman p-value: {p_value:.6f}). Nemenyi test results:")
        display(nemenyi_results)
    else:
        print(f"No significant differences found between models (Friedman p-value: {p_value:.6f}).")

def weightedbalancedaccuracy(y_true, preds, weights):
    """
    Balanced accuracy of one or many cached prediction sets at once, under per-row
    weights. Meant for a patient-cluster bootstrap over a fixed test set: statisticaldiff's
    Friedman/Nemenyi test only sees uncertainty from the manuscript's 10 training-split
    repetitions, never from which patients ended up in that fixed test set. Resampling test
    patients with replacement and giving a patient drawn k times a row weight of k (via
    audit.patientbootstrapindex) and recomputing this per replicate exposes that second,
    usually much larger, source of uncertainty. Classes with zero weight in a replicate are
    dropped, matching sklearn.metrics.balanced_accuracy_score's handling of absent classes.
        Inputs:  - [y_true: array-like]: The ground-truth labels, shape (n_rows,).
                 - [preds: np.ndarray]: Predicted labels, shape (n_rows,) or (n_models, n_rows).
                 - [weights: array-like]: The per-row weight for this replicate, shape (n_rows,).
        Outputs: - [balancedaccuracy: np.ndarray]: One value per model (row of ``preds``), shape (n_models,).
    """
    y_true = np.asarray(y_true)
    preds = np.atleast_2d(preds)
    weights = np.asarray(weights, dtype=float)
    recalls = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        denom = weights[mask].sum()
        if denom <= 0:
            continue
        correct = (preds[:, mask] == cls).astype(float)
        recalls.append((correct * weights[mask]).sum(axis=1) / denom)
    if not recalls:
        return np.full(preds.shape[0], np.nan)
    return np.mean(np.vstack(recalls), axis=0)