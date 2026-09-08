# PACKAGE IMPORTS

# System packages
import numpy as np

# Image processing packages
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label

# Machine learning packages
from torchvision import transforms

# CUSTOM FUNCTIONS

def normalization(mean, std):
    """
    Function to create a normalization transformation for image preprocessing, using the specified mean and standard deviation values.
        Inputs:  - [mean: tuple]: A tuple of mean values for each channel (R, G, B) to use for normalization.
                 - [std: tuple]: A tuple of standard deviation values for each channel (R, G, B) to use for normalization.
        Outputs: - [transforms.Compose]: A composed transformation that converts an image to a tensor and normalizes it using the provided mean and standard deviation values.
    """
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

def imagepreprocessing(imagepath):
    """
    Function to load and preprocess an image based on the CLIP preprocessing type, including resizing, normalization, and optional cropping and padding to make the image square.
        Inputs:  - [imagepath: str]: The path to the image to load and preprocess.
        Outputs: - [image: np.ndarray]: The preprocessed image as a NumPy array.
    """
    resizing = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]) 
    image = Image.open(imagepath).convert("RGB")
    image = squaremaskimage(image)
    image = resizing(image).permute(1, 2, 0).numpy()
    normalizer = normalization((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))
    image = normalizer(image).permute(1, 2, 0).numpy()
    return image

def imagenetpreprocessing(imagepath):
    """
    Function to load and preprocess an image the same way as imagepreprocessing (Otsu-crop
    and square-pad, resize, tensor conversion) but normalized with ImageNet statistics
    instead of CLIP's, for torchvision ImageNet-pretrained encoders (e.g. ResNet-50).
        Inputs:  - [imagepath: str]: The path to the image to load and preprocess.
        Outputs: - [image: np.ndarray]: The preprocessed image as a NumPy array.
    """
    resizing = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    image = Image.open(imagepath).convert("RGB")
    image = squaremaskimage(image)
    image = resizing(image).permute(1, 2, 0).numpy()
    normalizer = normalization((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    image = normalizer(image).permute(1, 2, 0).numpy()
    return image

def squaremaskimage(image):
    """
    Function to crop an image to its bounding box and then pad it to make it square, centering the original content.
        Inputs:  - [image: PIL.Image]: The input image to process.
        Outputs: - [masked: PIL.Image]: The processed image, cropped to the bounding box of the non-background content and padded to be square, with the original content centered.
    """
    binary = np.array(image.convert("L")) > threshold_otsu(np.array(image.convert("L")))
    regions = regionprops(label(binary))
    largestregion = max(regions, key=lambda r: r.area)
    bbox = largestregion.bbox
    roi = image.crop((bbox[1], bbox[0], bbox[3], bbox[2]))
    masked = Image.new("RGB", (max(roi.size), max(roi.size)))
    masked.paste(roi, ((max(roi.size) - roi.size[0]) // 2, (max(roi.size) - roi.size[1]) // 2))
    return masked