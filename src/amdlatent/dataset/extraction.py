# PACKAGE IMPORTS

# System packages
import os
import numpy as np
import pandas as pd
import re
from pathlib import Path

# Machine learning packages
from sklearn.cluster import MeanShift
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from sklearn.model_selection import train_test_split

# CUSTOM FUNCTIONS

def setextraction(setpaths, dataset, paths = None, latents = None, scaler = None):
    setpaths = setpaths[setpaths.columns[0]].tolist()
    setlabels = [dataset.loc[dataset["ID2"].astype(str) == str(Path(p).parts[-5]), f"SCALE_{int(str(Path(p).parts[-2]).strip())}"].iloc[0] for p in setpaths]
    if paths is not None:
        pathidx = {p: i for i, p in enumerate(paths[paths.columns[0]].tolist())}
        idxs = [pathidx[p] for p in setpaths]
        setlatents = latents[idxs]
        if scaler is not None: setlatents = scaler.transform(setlatents)
        setpaths = [[path, path.replace("LE", "RE")] for path in setpaths]
        return setlatents, setlabels, setpaths
    else: return setlabels

def classlabeler(labels, classtype):
    classtype = classtype.split("_")[0]
    if classtype == "Binary No":
        return [0 if y == 0 else 1 for y in labels]
    elif classtype == "Binary Late":
        return [0 if y <= 3 else 1 for y in labels]
    else: return labels

def setsextraction(dataset, classtype, splittype, mixing = False, randomstate = None, testpath = False, encoder = "retclip"):
    latents = np.load(os.path.join("results", "areds", "dataset", "encoding", encoder, "latentvec.npy"))
    paths = np.load(os.path.join("results", "areds", "dataset", "encoding", encoder, "imagepaths.npy"))
    paths = pd.DataFrame(paths)
    trainpaths = pd.read_csv(os.path.join("results", "areds", "dataset", "splits", f"trainpaths_{splittype.split('-')[0].lower().replace(' ', '')}.csv"))
    valpaths = pd.read_csv(os.path.join("results", "areds", "dataset", "splits", f"valpaths_{splittype.split('-')[0].lower().replace(' ', '')}.csv"))
    if mixing is True:
        constrpaths = pd.concat([trainpaths, valpaths], ignore_index=True)
        trainpaths, valpaths, _ = datasetsplit(constrpaths, splittype, constrsize=None, trainsize=0.8, randomstate = randomstate, dataset = dataset)
    testpaths = pd.read_csv(os.path.join("results", "areds", "dataset", "splits", f"testpaths_{splittype.split('-')[0].lower().replace(' ', '')}.csv"))
    if not os.path.exists(os.path.join("results", "areds", "classifiers", encoder, "scalers", splittype.split('-')[0].lower().replace(' ', ''), classtype.split("_")[0].lower().replace(' ', ''), "scaler.pkl")):
        os.makedirs(os.path.join("results", "areds", "classifiers", encoder, "scalers", splittype.split('-')[0].lower().replace(' ', ''), classtype.split("_")[0].lower().replace(' ', '')), exist_ok=True)
        constrpaths = pd.concat([trainpaths, valpaths], ignore_index=True)
        constr, _, _ = setextraction(constrpaths, dataset, paths, latents)
        scaler = MinMaxScaler()
        scaler.fit(constr)
        with open(os.path.join("results", "areds", "classifiers", encoder, "scalers", splittype.split('-')[0].lower().replace(' ', ''), classtype.split("_")[0].lower().replace(' ', ''), "scaler.pkl"), 'wb') as f: pkl.dump(scaler, f)
    else:
        with open(os.path.join("results", "areds", "classifiers", encoder, "scalers", splittype.split('-')[0].lower().replace(' ', ''), classtype.split("_")[0].lower().replace(' ', ''), "scaler.pkl"), 'rb') as f: scaler = pkl.load(f)
    X_train, y_train, _ = setextraction(trainpaths, dataset, paths, latents, scaler)
    X_val, y_val, _ = setextraction(valpaths, dataset, paths, latents, scaler)
    X_test, y_test, testpaths = setextraction(testpaths, dataset, paths, latents, scaler)
    y_train = classlabeler(y_train, classtype)
    y_val = classlabeler(y_val, classtype)
    y_test = classlabeler(y_test, classtype)
    if testpath is True:
        return X_train, y_train, X_val, y_val, X_test, y_test, trainpaths, testpaths
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test, trainpaths

def besthypersextraction(splittype, classtype, classifier, encoder = "retclip"):
    classifierinfo = pd.read_csv(os.path.join("results", "areds", encoder, "metrics", splittype.split('-')[0].lower().replace(' ', ''), classtype.split("_")[0].lower().replace(' ', ''), f"{classifier.lower().replace(' ', '_')}.csv"))
    best_hyperparameters = classifierinfo.iloc[-1]['Hyperparameters']
    if classifier in ["SVM"]:
        params = dict(item.strip().split("=") for item in str(best_hyperparameters).split(","))
        besthyper = (float(params["C"]), float(params["gamma"]))
    elif classifier == "SVM Linear":
        value = str(best_hyperparameters).split("=")[-1].strip()
        besthyper = float(value)
    elif classifier in ["Random Forest", "AdaBoost"]:
        value = str(best_hyperparameters).split("=")[-1].strip()
        besthyper = int(float(value))
    elif classifier == "MLP":
        m = re.search(r"hidden_layer_sizes=\((.*?)\)", str(best_hyperparameters))
        besthyper = tuple(int(v.strip()) for v in m.group(1).split(",") if v.strip())
    else:
        besthyper = None
    return besthyper

def getimages(imagespath):
    """
    Function to get all the image paths in a given directory.
        Inputs:  - [imagespath: str]: The path to the image dataset directory.
        Outputs: - [imagepaths: list]: A list of image paths.
    """
    imagepaths = []
    for root, _, files in os.walk(imagespath):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                imagepaths.append(os.path.join(root, file))
    return imagepaths

def matchimages(selectedimages):
    """
    Function to match left eye (LE) and right eye (RE) images based on their filenames, creating a DataFrame with matched pairs.
        Inputs:  - [selectedimages: list]: A list of image paths to match.
        Outputs: - [matchedimages: pd.DataFrame]: A DataFrame containing matched pairs of left eye (LE) and right eye (RE) images, with columns 'LE' and 'RE'.
    """
    re_lookup = {}
    rows = []
    for path in selectedimages:
        if 'RE' in path:
            key = Path(path).name.replace('RE', 'RE')
            if key not in re_lookup:
                re_lookup[key] = path
    for path in selectedimages:
        if 'LE' in path:
            key = Path(path).name.replace('LE', 'RE')
            if key in re_lookup:
                rows.append({'LE': path, 'RE': re_lookup[key]})
    matchedimages = pd.DataFrame(rows, columns=['LE', 'RE'])
    return matchedimages

def filteringimages(imagepaths, blacklist=None):
    """
    Function to filter image paths based on certain criteria, in addition to the images labeled with a non-numeric visit number.
        Inputs:  - [imagepaths: list]: A list of image paths to filter.
                 - [blacklist: str or pd.DataFrame, optional]: A string to exclude from the paths or a DataFrame containing 'Patient' and 'Visit' columns to exclude specific patient-visit combinations.
        Outputs: - [filtered_paths: list]: A list of filtered image paths.
    """
    if isinstance(imagepaths, pd.DataFrame):
        for col in imagepaths.columns:
            deletedrows = imagepaths[col].apply(lambda p: not (isinstance(p, str) and len(Path(p).parts) > 1 and Path(p).parts[-2].isdigit()))
            imagepaths = imagepaths.loc[~deletedrows]
        imagepaths = imagepaths.reset_index(drop=True)
        if isinstance(blacklist, str):
            for col in imagepaths.columns:
                deletedrows = imagepaths[col].apply(lambda p: isinstance(p, str) and blacklist in p)
                imagepaths = imagepaths.loc[~deletedrows]
            imagepaths = imagepaths.reset_index(drop=True)
        elif isinstance(blacklist, pd.DataFrame):
            for col in imagepaths.columns:
                deletedrows = imagepaths[col].apply(lambda p: isinstance(p, str) and any((blacklist['Patient'] == Path(p).parts[-5]) & (blacklist['Visit'] == Path(p).parts[-2])))
                imagepaths = imagepaths.loc[~deletedrows]
            imagepaths = imagepaths.reset_index(drop=True)
    else:
        imagepaths = [path for path in imagepaths if path.split("/")[-2].isdigit()]
        if isinstance(blacklist, str):
            imagepaths = [path for path in imagepaths if blacklist not in path]
        elif isinstance(blacklist, pd.DataFrame):
            imagepaths = [path for path in imagepaths if not any((blacklist['Patient'] == path.split("/")[-5]) & (blacklist['Visit'] == path.split("/")[-2]))]
    return imagepaths

def trainextract(kshots, trainpaths, y_train, X_train, type = "first"):
    trainlatents = []
    trainlabels = []
    paths = []
    for label in np.unique(y_train):
        idx = np.where(np.asarray(y_train) == label)[0]
        latents = X_train[idx]
        if type == "centroid":
            centroid = MeanShift(bandwidth=1).fit(latents).cluster_centers_[0]
            distances = np.linalg.norm(latents - centroid, axis=1)
            selected_idx = np.argsort(distances)[:int(kshots)]
        elif type == "first":
            selected_idx = np.arange(min(int(kshots), len(latents)))
        elif type == "random":
            selected_idx = np.random.choice(len(latents), size=int(kshots), replace=False)
        trainlatents.extend(latents[selected_idx])
        trainlabels.extend(np.asarray(y_train)[idx][selected_idx])
        if hasattr(trainpaths, "iloc"):
            paths.extend(trainpaths.iloc[idx].iloc[selected_idx].values.tolist())
        else:
            paths.extend(np.asarray(trainpaths)[idx][selected_idx])
    return list(zip(trainlatents, trainlabels)), paths

def datasetsplit(paths, splittype, constrsize=0.8, trainsize=0.8, randomstate=42, dataset = None):
    IDs = paths[paths.columns[0]].apply(lambda p: str(Path(p).parts[-5])).drop_duplicates().reset_index(drop=True)
    if constrsize is not None:
        constrIDs, testIDs = train_test_split(IDs, train_size=constrsize, random_state=randomstate)
        constrpaths = paths[paths[paths.columns[0]].apply(lambda p: str(Path(p).parts[-5])).isin(constrIDs)]
        testpaths = paths[paths[paths.columns[0]].apply(lambda p: str(Path(p).parts[-5])).isin(testIDs)]
    else: constrpaths = paths; constrIDs = IDs; testpaths = None
    if splittype == "Couples-wise":
        trainpaths, valpaths = train_test_split(constrpaths, train_size=trainsize, random_state=randomstate)
    elif splittype == "Couples stratified-wise":
        y_constr = setextraction(constrpaths, dataset)
        trainpaths, valpaths = train_test_split(constrpaths, train_size=trainsize, random_state=randomstate, stratify=y_constr)
    elif splittype == "Patient-wise":
        trainIDs, valIDs = train_test_split(constrIDs, train_size=trainsize, random_state=randomstate)
        trainpaths = constrpaths[constrpaths[constrpaths.columns[0]].apply(lambda p: str(Path(p).parts[-5])).isin(trainIDs)]
        valpaths = constrpaths[constrpaths[constrpaths.columns[0]].apply(lambda p: str(Path(p).parts[-5])).isin(valIDs)]
    return trainpaths, valpaths, testpaths

def findsmokingstatus(enrollment, followup):
    smokebefore = enrollment['SMOKEDYN']
    smokingdata = {}
    for index, row in enrollment.iterrows():
        smokingpatient = []
        pID = str(row['ID2'])
        smokingpatient.append(row['SMKCURR'])
        for _, followup_row in followup[followup['ID2'].astype(str) == pID].iterrows():
            smokingpatient.append(followup_row['SMKCURR5'])
        smokingdata[pID] = smokingpatient
    for pID, smokingpatient in smokingdata.items():
        if pd.isna(smokingpatient[0]):
            smokingpatient[0] = smokebefore[enrollment['ID2'].astype(str) == pID].values[0]
        for i in range(1, len(smokingpatient)):
            if pd.isna(smokingpatient[i]):
                smokingpatient[i] = smokingpatient[i-1]
        for i in range(len(smokingpatient), max(len(smokingpatient) for smokingpatient in smokingdata.values())):
            smokingpatient.append(smokingpatient[-1])
    smokingtable = {}
    for pID, smokingpatient in smokingdata.items():
        statuspatient = []
        for i in range(len(smokingpatient)):
            if smokingpatient[i] == "Y": statuspatient.append("Current")
            elif smokingpatient[i] == "N":
                if smokingpatient[0] == "N" and all(s == "N" for s in smokingpatient[:i+1]): statuspatient.append("Never")
                elif smokingpatient[0] == "Y" or any(s == "Y" for s in smokingpatient[:i]): statuspatient.append("Former")
                else: statuspatient.append("Unknown")
            else: statuspatient.append("Unknown")
        smokingtable[pID] = statuspatient
    return smokingtable