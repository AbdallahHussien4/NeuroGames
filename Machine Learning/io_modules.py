import csv
import numpy as np
import pandas as pd
import scipy.io


def loadFromMat(filenames: list, type: str, channels: dict, fs: int, trial_start_sec: int, trial_end_sec: int):
    '''
    Loads the EEG data from some files and returns the device's channels' readings and labels for all trials.
            
            Parameters:
                    filenames (list): A list of filenames to read data from
                    type (str): The type of the dataset (k3b/IVa/biomedical)
                    channels (dict): A dictionary containing each channel's position initialized with an empty array to be filled later 
                    fs (int): The sampling frequency of the device used to collect the data
                    trial_start_sec (int): The starting time in a trial where the user began imagining the action
                    trial_end_sec (int): The ending time in a trial where the user stopped imagining the action

            Returns:
                    channels: A dictionary containing each channel's position and its readings throughout all given experiments
                    labels: The labels for each trial for the given experiments
    '''
    labels = []

    for file in filenames:
        mat = scipy.io.loadmat(file + ".mat")

        if type == "k3b":
            # The whole data 986780 (reading) * 60 (channels)
            data = mat['s']

            # The start time of each trial
            trial_starts = mat['HDR']['TRIG'][0][0]
            # The label for each trial [1(Left), 2(right), 3(foot), 4(tongue)]
            labels.extend(mat['HDR']['Classlabel'][0][0].flatten())
            curLabels = mat['HDR']['Classlabel'][0][0]
            
            # Group each trial data together and remove `nan` trials and change any `nan` number with 0 
            for i in range(len(trial_starts)):
                if np.isnan(curLabels[i]) or (curLabels[i][0] != 1 and curLabels[i][0] != 2):
                    continue
                trial_beginning = int(trial_starts[i] + trial_start_sec * fs)
                trial_end = int(trial_starts[i] + trial_end_sec * fs)
                trial_data = data[trial_beginning: trial_end]
                trial_data[np.isnan(trial_data)] = 0
                for channel in channels.keys():
                    channels[channel].extend(np.array(trial_data[:, channel]).flatten())
            
        elif type == "IVa":

            data = mat['cnt']

            # The start time of each trial
            trial_starts = mat['mrk']['pos'][0][0][0]
            # The label for each trial [1(Left), -1(foot)]
            labels.extend(mat['mrk']['y'][0][0][0])
            curLabels = mat['mrk']['y'][0][0][0]

            # Group each trial data together and remove `nan` trials and change any `nan` number with 0 
            for i in range(len(trial_starts)):
                if np.isnan(curLabels[i]):
                    continue
                trial_beginning = int(trial_starts[i] + trial_start_sec * fs)
                trial_end = int(trial_starts[i] + trial_end_sec * fs)
                trial_data = data[trial_beginning: trial_end]
                trial_data[np.isnan(trial_data)] = 0
                for channel in channels.keys():
                    channels[channel].extend(np.array(trial_data[:, channel]).flatten())

        else:

            # Biomedical
            names = ['EEG_right', 'EEG_left', 'EEG_baseline']

            for index, name in enumerate(names):
                data = mat[name]
                labels.extend(list(np.ones(data.shape[2]) * index))
                for trial in range(data.shape[2]):
                    trial_beginning = int(trial_start_sec * fs)
                    trial_end = int(trial_end_sec * fs)
                    trial_data = data[trial_beginning: trial_end, :, trial]
                    for channel in channels.keys():
                        channels[channel].extend(np.array(trial_data[:, channel]).flatten())
    
    # filter nan labels and non left and right
    labels = [i for i in labels if not np.isnan(i) and i == 1 or i == 2]

    return channels, labels



def writeToCSV(rawEEGFileName: str, labelsFileName: str, channels: dict, labels: list):
    '''
    Writes the channels' reading and the labels into csv files.
            
            Parameters:
                    rawEEGFileName (str): The name for the csv file to save EEG data into
                    labelsFileName (str): The name for the csv file to save labels into
                    channels (dict): A dictionary containing each channel's position and its readings
                    labels (list): The labels for all trials

    '''
    with open(rawEEGFileName, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        rows = []
        for channel in channels.keys():
            rows.append(channels[channel])
        writer.writerows(rows)

    with open(labelsFileName, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for label in labels:
            writer.writerow([label])



def loadFromCSV(rawEEGFileName: str, labelsFileName: str):
    '''
    Reads the channels' reading and the labels from csv files.
            
            Parameters:
                    rawEEGFileName (str): The name for the csv file to load EEG data from
                    labelsFileName (str): The name for the csv file to load labels from

            Returns:
                    raw_eeg (2d numpy array): The raw_eeg data for each channel in the device
                    labels (1d numpy array): The labels for the experiments' trials
    '''
    raw_eeg = np.genfromtxt(rawEEGFileName, delimiter=',')
    labels = np.genfromtxt(labelsFileName, delimiter=',')
    # nw_labels = []
    # for label in labels:
    #     for _ in range(4):
    #         nw_labels.append(label)

    return raw_eeg, labels


def reportCSV(output_filename: str, subject_name: str,
                num_channels: int, bands: list, csp_n_components: int, 
                mibif_k: int, classifier: str, accuracy: float):

    '''
    Adds a subject's hyperparameters and best accuracy to a csv file.
            
            Parameters:
                    output_filename (str): The name for the csv file to save the subject's data into
                    subject_name (str): The name of the subject
                    num_channels (int): The number of channels used for this subject
                    bands (list): A list of the bands used during filtering
                    csp_n_components (int): The number of components used in CSP
                    mibif_k (int): The number of features extracted using MIBIF
                    classifier (str): The name of the used classifier
                    accuracy (float): The score of the classifier in %

    '''
    with open(output_filename, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        bands_str = ' '.join(bands)
        writer.writerow([subject_name, num_channels, bands_str, csp_n_components, mibif_k, classifier, accuracy])


def ReadEmotivData(fs: int, trial_start_sec: int, trial_end_sec: int, r_filename = 'data/francois.csv', labels_filename = 'data/francois_labels.csv', w_filename='raw_eeg.csv'):
    '''
    Reads the channels' reading from emotiv insight headset and convert it into suitable format.
            
            Parameters:
                    fs (int): The sampling frequency of the device used to collect the data
                    trial_start_sec (int): The starting time in a trial where the user began imagining the action
                    trial_end_sec (int): The ending time in a trial where the user stopped imagining the action
    '''
    df = pd.read_csv(r_filename)
    NewDF = df[['EEG.AF3','EEG.T7','EEG.Pz','EEG.T8','EEG.AF4', 'EEG.RawCq']]
    labels = np.genfromtxt(labels_filename, delimiter=',')
    channels = {'EEG.AF3':[],'EEG.T7':[],'EEG.Pz':[],'EEG.T8':[],'EEG.AF4':[]}
    minZeros = 28600
    for trial in range(len(labels)):
        trial_beginning = int(trial*fs*10 + trial_start_sec * fs)
        trial_end = int(trial*fs*10 + trial_end_sec * fs)
        trial_data = NewDF.iloc[trial_beginning: trial_end,:]
        zeros = sum(trial_data['EEG.RawCq'] == 0)
        if zeros < minZeros:
            minZeros = zeros
    for trial in range(len(labels)):
        trial_beginning = int(trial*fs*10 + trial_start_sec * fs)
        trial_end = int(trial*fs*10 + trial_end_sec * fs)
        trial_data = NewDF.iloc[trial_beginning: trial_end,:]
        z = 0
        for index, i in enumerate(trial_data['EEG.RawCq']):
            if i == 0 and z < minZeros:
                z += 1
                continue
            for channel in channels.keys():
                channels[channel].append(trial_data[channel][index + trial_beginning])
    with open(w_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        rows = []
        for channel in channels.keys():
            rows.append(channels[channel])
        writer.writerows(rows)