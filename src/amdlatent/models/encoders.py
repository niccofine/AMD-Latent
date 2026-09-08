# PACKAGE IMPORTS

# System packages
import os
import sys
import json
import gdown
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# PyTorch packages
import torch
from torchvision import transforms

# Dataset packages
from amdlatent.dataset.extraction import matchimages
from amdlatent.models.preprocessing import imagepreprocessing, imagenetpreprocessing

# RET-CLIP packages
sys.path.append(str(Path().resolve().parents[1] / "external" / "RET-CLIP-PLUS"))
from RET_CLIP_PLUS.clip.model import CLIP

# CUSTOM FUNCTIONS

def modeldef():
    """
    Function to download the RET-CLIP-PLUS model checkpoint and load the model configuration.
        Outputs: - [model: CLIP]: The loaded RET-CLIP-PLUS model.
                 - [ckpt: dict]: The state dictionary of the model checkpoint.
    """
    if not os.path.exists(os.path.join("models", "ret-clip-plus", "checkpoints")): os.makedirs(os.path.join("models", "ret-clip-plus", "checkpoints"))
    if not os.path.exists(os.path.join("models", "ret-clip-plus", "checkpoints", "ret-clip.pt")):
        gdown.download("https://drive.google.com/uc?id=1lYrAg5qzFbNghEW-3UB36v9WL-mo5eN9", os.path.join("models", "ret-clip-plus", "checkpoints", "ret-clip.pt"), quiet=False)
        
    with open(os.path.join("external", "RET-CLIP-PLUS", "RET_CLIP_PLUS", "clip", "model_configs", "ViT-B-16.json"), "r", encoding="utf-8") as f:
        modelinfo = json.load(f)
    if isinstance(modelinfo['vision_layers'], str):
        modelinfo['vision_layers'] = eval(modelinfo['vision_layers'])
    with open(os.path.join("external", "RET-CLIP-PLUS", "RET_CLIP_PLUS", "clip", "model_configs", "RoBERTa-wwm-ext-base-chinese.json"), "r", encoding="utf-8") as f:
        text_cfg = json.load(f)
    for k, v in text_cfg.items():
        modelinfo[k] = v
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIP(**modelinfo).to(device)
    ckpt = torch.load(os.path.join("models", "ret-clip-plus", "checkpoints", "ret-clip.pt"), map_location=device)
    return model, ckpt

def extrmodel():
    """
    Function to load the RET-CLIP-PLUS model checkpoint into the model architecture, ensuring that the weights are correctly assigned to the visual encoder and global feature mapping layers.
        Inputs:  - [model: CLIP]: The loaded RET-CLIP-PLUS model architecture.
                 - [ckpt: dict]: The state dictionary of the model checkpoint containing the trained weights.
        Outputs: - [model: CLIP]: The RET-CLIP-PLUS model with weights loaded from the checkpoint, ready for feature extraction.
    """
    model, ckpt = modeldef()
    sd = {}
    for k, v in ckpt.items():
        if "bert.pooler" in k:
            continue
        k = k.replace("module.", "")
        if k.startswith("visual.ln_post."):
            sd[k] = v
            sd[k.replace("visual.ln_post.", "visual.transformer.ln_post.", 1)] = v
            continue
        sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(unexpected) > 0:
        raise RuntimeError(f"Unexpected keys while loading RET-CLIP checkpoint: {unexpected}")
    model.eval()
    return model

def extrvencoder(model, ckpt):
    """
    Function to extract the visual encoder from the RET-CLIP-PLUS model checkpoint.
        Inputs:  - [model: CLIP]: The loaded RET-CLIP-PLUS model.
                 - [ckpt: dict]: The state dictionary of the model checkpoint.
        Outputs: - [vencoder: nn.Module]: The visual encoder of the RET-CLIP-PLUS model, with weights loaded from the checkpoint.
    """
    sd = {}
    for k, v in ckpt.items():
        if "bert.pooler" in k:
            continue
        k = k.replace("module.", "")
        if k.startswith("visual.ln_post."):
            sd[k] = v
            sd[k.replace("visual.ln_post.", "visual.transformer.ln_post.", 1)] = v
            continue
        sd[k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(unexpected) > 0:
        raise RuntimeError(f"Unexpected keys while loading RET-CLIP checkpoint: {unexpected}")

    model.eval()
    model.global_feature_mapping.eval()

    vencoder = model.visual
    vencoder.fusion_layer = model.global_feature_mapping
    return vencoder

def extrvencoderclip():
    """
    Downloads standard OpenAI CLIP and extracts its visual encoder.
    Common choices for model_name: "ViT-B/32", "ViT-B/16", "RN50"
    """
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return model

def extrvencoderresnet():
    """
    Downloads torchvision's ImageNet-pretrained ResNet-50 (IMAGENET1K_V2 weights) and
    strips its classification head, exposing the penultimate/avgpool 2048-d features as
    the visual encoder: a generic, non-medical baseline encoder.
        Outputs: - [vencoder: nn.Module]: The frozen ResNet-50 backbone, ready for feature extraction.
    """
    from torchvision.models import ResNet50_Weights, resnet50
    vencoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    vencoder.fc = torch.nn.Identity()
    vencoder.eval()
    return vencoder

def extrduallatent(img_path, vencoder, modeltype="RETCLIP", flipL=False):
    """
    Function to extract latent space features from a pair of images (left and right) using the visual encoder of the RET-CLIP-PLUS model, concatenating the features.
        Inputs:  - [img_path: tuple]: A tuple containing the paths to the left and right images to process.
                 - [vencoder: nn.Module]: The visual encoder of the RET-CLIP-PLUS model, used to extract features from the left and right images.
        Outputs: - [feature: np.ndarray]: The extracted latent space features for the pair of images, which can be used for classification or other downstream tasks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = imagenetpreprocessing if modeltype == "ResNet50" else imagepreprocessing
    imageL = preprocess(img_path[0])
    if flipL:
        imageL = transforms.ToPILImage()(imageL)
        imageL = transforms.functional.hflip(imageL)
    imageL = transforms.ToTensor()(imageL).unsqueeze(0).to(device)
    imageR = preprocess(img_path[1])
    imageR = transforms.ToTensor()(imageR).unsqueeze(0).to(device)
    if modeltype == "RETCLIP":
        left_feature = vencoder(imageL)
        right_feature = vencoder(imageR)
    elif modeltype == "CLIP":
        left_feature = vencoder.get_image_features(pixel_values=imageL).pooler_output
        right_feature = vencoder.get_image_features(pixel_values=imageR).pooler_output
    elif modeltype == "ResNet50":
        left_feature = vencoder(imageL)
        right_feature = vencoder(imageR)
    feature = torch.cat((left_feature, right_feature), dim=1)
    feature = feature.cpu().numpy()
    return feature

def extrlatents(imagepaths, vencoder, modeltype="RETCLIP", matchimage = True, flipL = False):
    """
    Function to extract latent space features from a list of image paths using the visual encoder of the RET-CLIP-PLUS model.
        Inputs:  - [image_paths: list]: A list of image paths to process.
                 - [vencoder: nn.Module]: The visual encoder of the RET-CLIP-PLUS model, used to extract features from the images.
        Outputs: - [embeddings: np.ndarray]: A 2D array containing the extracted features for each image.
                 - [paths: list]: A list of image paths corresponding to the extracted features.
    """
    if len(imagepaths) == 0:
        raise ValueError(f"No images found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): vencoder = vencoder.to('cuda')
    embeddings = []
    paths = []
    
    with torch.no_grad():
        if matchimage:
            matchedimages = matchimages(imagepaths)
        else:
            matchedimages = pd.DataFrame(imagepaths, columns=['LE', 'RE'])
        for img_path in tqdm(matchedimages.itertuples(index=False), total=len(matchedimages)):
            feature = extrduallatent(img_path, vencoder, modeltype, flipL=flipL)
            embeddings.append(feature)
            paths.append(img_path)
    if len(embeddings) == 0:
        raise ValueError(f"No embeddings were extracted")
    return np.vstack(embeddings), paths