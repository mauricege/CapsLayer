import librosa
import numpy as np
import os
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

WS = 0.5  # WINDOW_SIZE_IN_SECONDS
HL = 0.25  # HOP_LENGTH_IN_SECONDS
SR = 44000  # sample rate


def extract_spectrograms(data_dir, oversample=True):
    train_csv = os.path.join(data_dir, 'train.csv')
    eval_csv = os.path.join(data_dir, 'eval.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    le = LabelEncoder()
    if os.path.isfile(train_csv):
        dataset = []
        labels = []
        fileslist = open(train_csv, "r").readlines()
        for line in tqdm(fileslist[1:]):
            wavfile, label = line.strip().split(",")

            fullpath = os.path.join(data_dir, 'wav', wavfile)
            y, sr = librosa.load(fullpath, sr=SR)
            frames = librosa.util.frame(y, frame_length=int(WS * sr), hop_length=int(HL * sr))
            for frame in frames.T:
                spc = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=32)
                dataset.append(spc)
                labels.append(label)

        le.fit(sorted(set(labels)))
        with open(os.path.join(data_dir, 'label_map.txt'), 'w') as f:
            f.writelines([f"{label}: {i}\n" for i, label in enumerate(le.classes_)])

        dataset = np.array(dataset)
        labels = np.array(le.transform(labels))
        print("dataset (no resampling): {}".format(dataset.shape))
        print("labels (no resampling): {}".format(labels.shape))
        if oversample == True:
            print('resampling...')
            data_idx = np.reshape(range(dataset.shape[0]), (-1, 1))
            ros = RandomOverSampler(random_state=42)
            data_idx, labels = ros.fit_resample(data_idx, labels)
            dataset = dataset[data_idx.flatten()]
            print("final dataset: {}".format(dataset.shape))
            print("final labels: {}".format(labels.shape))
        np.save(os.path.join(data_dir, 'train.npy'), dataset)
        np.save(os.path.join(data_dir, 'train-labels.npy'), labels)
    if os.path.isfile(eval_csv):
        dataset = []
        labels = []
        fileslist = open(eval_csv, "r").readlines()
        for line in tqdm(fileslist[1:]):
            wavfile, label = line.strip().split(",")
            fullpath = os.path.join(data_dir, 'wav', wavfile)
            y, sr = librosa.load(fullpath, sr=SR)
            frames = librosa.util.frame(y, frame_length=int(WS * sr), hop_length=int(HL * sr))
            for frame in frames.T:
                spc = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=32)
                dataset.append(spc)
                labels.append(label)

        dataset = np.array(dataset)
        labels = np.array(le.transform(labels))
        sort_index = np.argsort(labels)
        labels = labels[sort_index]
        dataset = dataset[sort_index, :, :]


        print("final dataset: {}".format(dataset.shape))
        print("final labels: {}".format(labels.shape))
        np.save(os.path.join(data_dir, 'eval.npy'), dataset)
        np.save(os.path.join(data_dir, 'eval-labels.npy'), labels)

    if os.path.isfile(test_csv):
        dataset = []
        labels = []
        fileslist = open(test_csv, "r").readlines()
        for line in tqdm(fileslist[1:]):
            wavfile, label = line.strip().split(",")
            fullpath = os.path.join(data_dir, 'wav', wavfile)
            y, sr = librosa.load(fullpath, sr=SR)
            frames = librosa.util.frame(y, frame_length=int(WS * sr), hop_length=int(HL * sr))
            for frame in frames.T:
                spc = librosa.feature.melspectrogram(y=frame, sr=sr, n_mels=32)
                dataset.append(spc)
                labels.append(label)

        dataset = np.array(dataset)
        labels = np.array(le.transform(labels))
        sort_index = np.argsort(labels)
        labels = labels[sort_index]
        dataset = dataset[sort_index, :, :]

        print("final dataset: {}".format(dataset.shape))
        print("final labels: {}".format(labels.shape))
        np.save(os.path.join(data_dir, 'eval.npy'), dataset)
        np.save(os.path.join(data_dir, 'eval-labels.npy'), labels)

def maybe_extract_spectrograms(data_dir, oversample=True):
    npys = [npy for npy in os.listdir(data_dir) if os.path.join(data_dir, npy).endswith('.npy')]
    if not npys:
        extract_spectrograms(data_dir, oversample=oversample)
    npys = [os.path.join(data_dir,npy) for npy in os.listdir(data_dir) if npy.endswith('.npy') and not npy.endswith('-labels.npy')]
    with open(os.path.join(data_dir, 'label_map.txt'), 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        print(f"label map:\n{lines}")
    data = np.load(npys[0])
    return data.shape[1], data.shape[2], len(lines)

