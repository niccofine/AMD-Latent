# PACKAGE IMPORTS

# System packages
import numpy as np
import os
import pandas as pd

# Classification packages
from amdlatent.models.preprocessing import imagepreprocessing, squaremaskimage
import torch
import torch.nn as nn
from torchvision import transforms

# Statistical packages
import shap

# Visualization packages
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# CUSTOM FUNCTIONS

def computeshap(model, X, classifier):
    """
    SHAP values for the positive class, dispatched by classifier type: LinearExplainer for
    the linear SVM, TreeExplainer for tree ensembles (Random Forest/AdaBoost), each with
    the rows being explained also used as the background set. Feeds extractheatmaps'
    shap_vec/valueheatmap once collapsed to one class or one row.
        Inputs:  - [model]: The fitted classifier (as returned by classifiers.trainmodel).
                 - [X: np.ndarray]: The rows to explain, shape (n_samples, n_features).
                 - [classifier: str]: The classifier name (as used by classifiers.trainmodel/evaluation.predextr).
        Outputs: - [shapvalues: np.ndarray]: SHAP values, shape (n_samples, n_features).
    """
    if classifier.lower() == "svm linear":
        try:
            explainer = shap.LinearExplainer(model, X)
        except Exception:
            explainer = shap.LinearExplainer((np.asarray(model.coef_).ravel(), float(np.ravel(model.intercept_)[0])), X)
        return np.asarray(explainer.shap_values(X))
    explainer = shap.TreeExplainer(model)
    shapvalues = explainer.shap_values(X)
    if isinstance(shapvalues, list):
        shapvalues = shapvalues[1]
    shapvalues = np.asarray(shapvalues)
    if shapvalues.ndim == 3:
        shapvalues = shapvalues[:, :, 1]
    return shapvalues

def rankdimensions(shapperrep, topk=3, window=20):
    """
    Rank latent dimensions by cross-repetition consistency of SHAP importance, then by mean
    importance: the stability count is how many repetitions place a dimension in that
    repetition's own top-``window`` by mean |SHAP|, which is more robust to a single
    repetition's noise than ranking by the pooled mean alone.
        Inputs:  - [shapperrep: np.ndarray]: Mean |SHAP| per repetition, shape (repetitions, dimensions), e.g. stacking np.abs(computeshap(...)).mean(axis=0) over repeated train/val re-splits.
                 - [topk: int]: How many top-ranked dimensions to flag as selected.
                 - [window: int]: The per-repetition top-N used for the stability count.
        Outputs: - [ranking: pd.DataFrame]: One row per dimension, sorted best-first, with a "Selected" flag on the top ``topk``.
    """
    nrep, ndim = shapperrep.shape
    counts = np.zeros(ndim, dtype=int)
    for r in range(nrep):
        counts[np.argsort(-shapperrep[r])[:window]] += 1
    ranking = pd.DataFrame({
        "Dimension": np.arange(ndim),
        "MeanAbsSHAP": shapperrep.mean(axis=0),
        "SdAbsSHAP": shapperrep.std(axis=0),
        f"InTop{window}Count": counts,
    })
    ranking = ranking.sort_values([f"InTop{window}Count", "MeanAbsSHAP"], ascending=[False, False]).reset_index(drop=True)
    ranking["Selected"] = False
    ranking.loc[:topk - 1, "Selected"] = True
    return ranking

def load_and_preprocess(img_path):
    img = imagepreprocessing(img_path)
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return tensor

def infer_patch_grid(num_patches, expected_grid_size=None):
    if expected_grid_size is not None:
        expected_count = expected_grid_size[0] * expected_grid_size[1]
        if expected_count == num_patches:
            return expected_grid_size
    side = int(np.sqrt(num_patches))
    if side * side == num_patches:
        return (side, side)
    for height in range(side, 0, -1):
        if num_patches % height == 0:
            return (height, num_patches // height)
    raise ValueError(f"cannot infer patch grid from {num_patches} patches")


def valueheatmap(img_tensor, model, shap_vec, grid_size=None):
    """
    img_tensor: [1, 3, H, W]
    shap_vec: shape [output_dim]
    grid_size: optional (h_grid, w_grid)
    """
    with torch.no_grad():
        patch_embeds = model.forward_all_tokens(img_tensor)[0].detach().cpu().numpy()
    if patch_embeds.shape[0] == shap_vec.shape[0] + 1:
        patch_embeds = patch_embeds[1:]
    patch_scores = np.dot(patch_embeds, shap_vec)
    inferred_grid_size = infer_patch_grid(patch_scores.size, grid_size)
    patch_map = patch_scores.reshape(inferred_grid_size)
    max_abs = np.max(np.abs(patch_map))
    patch_map = patch_map / (max_abs + 1e-10)
    return patch_map

def extractheatmaps(LEpath, REpath, vencoder, type = "value", shap_vec = None, feat = None):
    imgL = load_and_preprocess(LEpath)
    imgR = load_and_preprocess(REpath)
    shap_vec = shap_vec.flatten()
    half_feats = len(shap_vec) // 2
    if type == "attention" and feat is None: return attentionheatmap(vencoder, imgL), attentionheatmap(vencoder, imgR)
    elif type == "value":
        contribLE = np.zeros(half_feats)
        contribRE = np.zeros(half_feats)
        if feat is None:
            contribLE = shap_vec[:half_feats]
            contribRE = shap_vec[half_feats:]
        else:
            if feat < half_feats: contribLE[feat] = shap_vec[feat]
            else: contribRE[feat - half_feats] = shap_vec[feat]
        heatmapLE = valueheatmap(imgL, vencoder, contribLE)
        heatmapRE = valueheatmap(imgR, vencoder, contribRE)
    elif type == "attention" and feat is not None:
        if feat >= half_feats:
            heatmapLE = np.abs(np.zeros((14, 14)))
            heatmapRE = np.abs(attentionheatmap(vencoder, imgR, featidx=feat - half_feats))
        else:
            heatmapLE = np.abs(attentionheatmap(vencoder, imgL, featidx=feat))
            heatmapRE = np.abs(np.zeros((14, 14)))
        if feat >= half_feats: feat = feat - half_feats
    return heatmapLE, heatmapRE

def heatmapscarousel(LEpath, REpath, vencoder, y_pred, y_true, X, type="value", shap1 = None, feats = None, returnfig = True):
    if type == "attention":
        colormap = "inferno"
        title = "Attention Heatmaps"
    elif type == "value":
        colormap = "jet"
        title = "Value Heatmaps"
    if feats is not None: bestfeats = feats
    else: bestfeats = [0]
    origL = np.asarray(squaremaskimage(Image.open(LEpath).convert('RGB')))
    origR = np.asarray(squaremaskimage(Image.open(REpath).convert('RGB')))
    hL, wL = origL.shape[:2]
    hR, wR = origR.shape[:2]
    _, axes = plt.subplots(len(bestfeats)+1, 2, figsize=(12, 4 * (len(bestfeats)+1)))
    axes[0, 0].imshow(origL)
    axes[0, 0].set_title("Cropped Left Eye Image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(origR)
    axes[0, 1].set_title("Cropped Right Eye Image")
    axes[0, 1].axis("off")

    for i, f in enumerate(bestfeats):
        if feats is None:
            label = "All Features"
            f = None
        else: label = f"Feature {f} (SHAP best {i+1} - Class 1) - Value {X[0, f]:.4f}"
        heatmapLE, heatmapRE = extractheatmaps(LEpath, REpath, vencoder, type=type, shap_vec=shap1, feat=f)
        axes[i+1, 0].imshow(origL)
        axes[i+1, 0].imshow(cv2.resize(heatmapLE, (wL, hL), interpolation=cv2.INTER_CUBIC), alpha=0.55, cmap=colormap)
        axes[i+1, 0].set_title(label)
        axes[i+1, 0].axis("off")

        axes[i+1, 1].imshow(origR)
        axes[i+1, 1].imshow(cv2.resize(heatmapRE, (wR, hR), interpolation=cv2.INTER_CUBIC), alpha=0.55, cmap=colormap)
        axes[i+1, 1].set_title(label)
        axes[i+1, 1].axis("off")
    plt.suptitle(f"{title}: Patient {LEpath.split(os.sep)[-5]} - Visit {LEpath.split(os.sep)[-2]} - View {LEpath.split(os.sep)[-4]} (True: {y_true}, Predicted: {y_pred})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if returnfig: return plt.gcf()
    else: plt.show()

def attentionextr(vencoder, x):
    """
    Extract attention computing attention directly from model architecture
    """
    with torch.no_grad():
        x = vencoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            vencoder.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ), x
        ], dim=1)
        x = x + vencoder.positional_embedding.to(x.dtype)
        x = vencoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        for blk in vencoder.transformer.resblocks[:-1]:
            x = blk(x)
        last_block = vencoder.transformer.resblocks[-1]
        x_ln = last_block.ln_1(x)
        attn_module = last_block.attn
        L, B, D = x_ln.shape
        num_heads = attn_module.num_heads
        head_dim = D // num_heads
        in_proj_weight = attn_module.in_proj_weight
        in_proj_bias = attn_module.in_proj_bias if attn_module.in_proj_bias is not None else None
        W_q, W_k, W_v = in_proj_weight.chunk(3)
        b_q, b_k, b_v = (in_proj_bias.chunk(3) if in_proj_bias is not None else (None, None, None))
        q = torch.nn.functional.linear(x_ln, W_q, b_q)
        k = torch.nn.functional.linear(x_ln, W_k, b_k)
        v = torch.nn.functional.linear(x_ln, W_v, b_v)
        q = q.reshape(L, B, num_heads, head_dim).permute(1, 2, 0, 3)
        k = k.reshape(L, B, num_heads, head_dim).permute(1, 2, 0, 3)
        v = v.reshape(L, B, num_heads, head_dim).permute(1, 2, 0, 3)
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return attn_weights, head_dim, v

def attentionheatmap(vencoder, x, featidx = None):    
    attn, head_dim, v = attentionextr(vencoder, x)
    nh = attn.shape[1]
    w_featmap = 14
    h_featmap = 14
    if featidx is None:
        attentions = attn[0, :, 0, 1:].reshape(nh, -1)
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = attentions.detach()
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=16, mode="nearest")[
            0].cpu().numpy()
        map = np.mean(attentions, axis=0)
    else:
        target_head = featidx // head_dim
        head_feature_idx = featidx % head_dim
        cls_attn_to_patches = attn[0, target_head, 0, 1:]
        patch_values_for_feature = v[0, target_head, 1:, head_feature_idx]
        map = cls_attn_to_patches * patch_values_for_feature
        map = map.reshape(1, 1, w_featmap, h_featmap)
        map = nn.functional.interpolate(map, scale_factor=16, mode="bilinear", align_corners=False)
        map = map[0, 0].cpu().numpy()
    return map