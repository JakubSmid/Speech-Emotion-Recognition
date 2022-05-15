import os

CREMA_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMO_EMOTIONS = {
    "W": "angry",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "N": "neutral",
    "T": "sad",
    "L": "boredom"
    }

# returns list [filename, emotion_id]
def load_crema_dataset():
    dataset = []
    for file in os.listdir("../datasets/crema-d/"):
        identifiers = file.replace(".wav", "").split("_")
        dataset.append(["../datasets/crema-d/"+file, CREMA_EMOTIONS[identifiers[2]]])
    return dataset

def load_crema_dataset_IWW(excluded = False):
    dataset = []
    for file in os.listdir("../datasets/crema-d/"):
        identifiers = file.replace(".wav", "").split("_")
        if not excluded and identifiers[1]=='IWW':
            dataset.append(["../datasets/crema-d/"+file, CREMA_EMOTIONS[identifiers[2]]])
        if excluded and identifiers[1] != 'IWW':
            dataset.append(["../datasets/crema-d/"+file, CREMA_EMOTIONS[identifiers[2]]])
    return dataset

def load_ravdess_dataset(raw=False):
    dataset = []
    for file in os.listdir("../datasets/ravdess/"):
        identifiers = file.replace(".wav", "").split("-")
        if not raw and (identifiers[2] != "02" and identifiers[2] != "08"):
            dataset.append(["../datasets/ravdess/"+file, RAVDESS_EMOTIONS[identifiers[2]]])
        elif raw:
            dataset.append(["../datasets/ravdess/"+file, RAVDESS_EMOTIONS[identifiers[2]]])
    return dataset

def load_emo_dataset(raw=False):
    dataset = []
    for file in os.listdir("../datasets/emo-db/"):
        identifier = file[5]
        if not raw and (identifier != "L"):
            dataset.append(["../datasets/emo-db/"+file, EMO_EMOTIONS[identifier]])
        elif raw:
            dataset.append(["../datasets/emo-db/"+file, EMO_EMOTIONS[identifier]])
    return dataset