import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from cnst import *
from sklearn.model_selection import KFold, cross_val_score

import glob, os

def motor_to_vec(x):
    d = np.zeros(NUM_CLASS)
    d[x] = 1.0
    return d

FILE_PATH = './Dataset/'


allFiles = glob.glob(FILE_PATH + "*.csv")
data = pd.DataFrame()
list_ = []
for file_ in allFiles:
    print(file_)
    df = pd.read_csv(file_,index_col=None, header=None)
    list_.append(df)
data = pd.concat(list_)

labels_1 = []
EEG_1 = []
labels_2 = []
EEG_2 = []

index = 0
total = data.shape[0]
flag = np.zeros(NUM_CLASS)

for index, row in data.iterrows():
    print('motor :', int(row[len(row) - 1]))
    motor = motor_to_vec(int(row[len(row) - 1]))
    eeg = np.reshape(row[:-1], [19, 300])

    if index >= 0 and index < 140:
        EEG_1.append(eeg.T)
        labels_1.append(motor)

    if index >= 140 and index < 280:
        EEG_2.append(eeg.T)
        labels_2.append(motor)

    index += 1
    print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

print('EEG_1',np.shape(EEG_1),np.shape(labels_1),np.shape(EEG_2),np.shape(labels_2))

k_fold = KFold(n_splits=10)

np_labels_1 = np.array(labels_1)
np_EEG_1 = np.array(EEG_1)
np_labels_2 = np.array(labels_2)
np_EEG_2 = np.array(EEG_2)
DATA_SET = []

print('np_labels_1',np.shape(np_labels_1),np.shape(np_EEG_1),np.shape(np_labels_2),np.shape(np_EEG_2))

for train_indices, test_indices in k_fold.split(np.array(EEG_1)):
    np.random.shuffle(train_indices)
    # for (x_train,y_train),(x_test,y_test),(x_valid,y_valid) in
    TRAIN_EEG = np.append(np_EEG_1[train_indices[:-70]],np_EEG_2[train_indices[:-70]], axis=0)
    TRAIN_LAB = np.append(np_labels_1[train_indices[:-70]], np_labels_2[train_indices[:-70]], axis=0)

    VALID_EEG = np.append(np_EEG_1[train_indices[-70:]],np_EEG_2[train_indices[-70:]], axis=0)
    VALID_LAB = np.append(np_labels_1[train_indices[-70:]], np_labels_2[train_indices[-70:]], axis=0)

    TEST_EEG = np.append(np_EEG_1[test_indices],np_EEG_2[test_indices], axis=0)
    TEST_LAB = np.append(np_labels_1[test_indices], np_labels_2[test_indices], axis=0)

    DATA_SET.append([TRAIN_EEG,TRAIN_LAB,TEST_EEG,TEST_LAB,VALID_EEG,VALID_LAB])

print('np_labels_1',np.shape(TRAIN_EEG),np.shape(TRAIN_LAB),np.shape(VALID_EEG),np.shape(VALID_LAB),np.shape(TEST_EEG),np.shape(TEST_LAB),np.shape(np.array(DATA_SET[0])))

np.save('EEG_csv_total.npy', np.array(DATA_SET))
