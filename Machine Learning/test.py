from IPython.utils import io
from Preprocessing import *
from Features_Selection import *
from Features_Extraction import *
from io_modules import *
from Classification import *
import pickle

Ws = np.load('weights.npy')
indexes = np.load('indexes.npy')
bands = np.load('bands.npy')

tot_raw_eeg = [np.load('saved_streams/stream_RL.npy'), np.load('saved_streams/stream_RNL.npy')]

channels = 5
fs = 128

n_samples = 0

model = pickle.load(open('finalized_model.sav', 'rb'))

preds = []
ur_trial = np.zeros((5, 128))


for inx, raw_eeg in enumerate(tot_raw_eeg):
    prs = []
    preds.append(str(inx + 1) + ": \n")
    for i, trial in enumerate(raw_eeg):

        with io.capture_output() as captured:
            filtered_test = filterData(trial, 1, fs, bands)
        X_test = extractTopFeatures(filtered_test, bands, Ws, indexes)

        probs = model.predict_proba(X_test)
        if probs[0][0] >= 0.8:
            prediction = 1
        elif probs[0][1] >= 0.8:
            prediction = 2
        else:
            prediction = 0
        preds.append(str(model.predict_proba(X_test)) + '  ' + str(prediction) + '\n')

        if inx == 0:
            if i < 20 and prediction == 1:
                prs.append(1)
            elif i >= 20 and prediction == 2:
                prs.append(1)
            if i == 19:
                preds.append(str(len(prs) / 20 * 100) + '%\n')
                prs = []
        else:
            if i < 20 and prediction == 1:
                prs.append(1)
            elif i >= 20 and i < 30 and prediction == 0:
                prs.append(1)
            elif i >= 30 and prediction == 2:
                prs.append(1)
            if i == 19:
                preds.append(str(len(prs) / 20 * 100) + '%\n')
                prs = []
            elif i == 29:
                preds.append(str(len(prs) / 10 * 100) + '%\n')
                prs = []
        
    preds.append(str(len(prs) / 20 * 100) + '%\n\n')

file_object = open('results.txt', 'a')
for string in preds:
    file_object.write(string)
file_object.write('\n')
file_object.close()
