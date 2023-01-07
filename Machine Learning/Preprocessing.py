import csv
import numpy as np
from numpy.lib.function_base import average
from scipy.signal import butter, lfilter
import mne


# Read Data and labels
def readData (dataFileName):
    dataFile = open(dataFileName)
    data = csv.reader(dataFile)
    rows = []
    for row in data:
        row = list(map(float, row))
        rows.append(row)
    dataFile.close()
    return np.array(rows)

# Average each trials data to create on value for each trial.
def averageData(data):
    result=[]
    for row in data:
        dic ={}
        for index,value in enumerate(row):
            if index%5 not in dic:
                dic[index%5] = []
            dic[index%5].append(value)
        result.append([average(dic[L]) for L in dic])
    return np.array(result)


def initializeVars(type: str, headset: str):
    '''
    Initializes the variables of interest to extract the data properly.
            
            Parameters:
                    type (str): The type of the dataset (k3b/IVa/biomedical/self_recorded)
                    headset (str): The headset that we want to extract its channels only

            Returns:
                    channels: A dictionary containing each channel's position initialized with an empty array to be filled later 
                    classes: A list of the possible classes that we'll classify between
                    fs: The sampling frequency of the device used to collect the data
                    trial_start_sec: The starting time in a trial where the user began imagining the action
                    trial_end_sec: The ending time in a trial where the user stopped imagining the action
    '''
    if type == 'k3b':

        # k3b / k6b / l1b
        fs = 250
        trial_start_sec = 2
        trial_end_sec = 5

        num_channels = 60

        if headset == 'insight':
            channels = {5:[], 7:[], 25: [], 35: [], 48: []}
        elif headset == 'crown':
            channels = {42:[], 26:[], 2: [], 8: [], 30: [], 46: []}
        else:
            # All channels
            channels = {}
            for c in range(num_channels):
                channels[c] = []   
        classes = [1, 2] 

    elif type == 'IVa':

        # IVa     aa / al / av / aw / ay
        fs = 100
        trial_start_sec = 1
        trial_end_sec = 3

        num_channels = 118

        if headset == 'insight':
            channels = {49:[], 57:[], 90: [], 6: [], 7: []}
        elif headset == 'crown':
            channels = {69:[], 73:[], 51: [], 55: [], 103: [], 107: [], 14: [], 20: []}
        else:
            # All channels
            channels = {}
            for c in range(num_channels):
                channels[c] = []   
        classes = [1, 2] 

    elif type == 'self_recorded':

            fs = 128
            trial_start_sec = 4.5
            trial_end_sec = 9

            num_channels = 5

            # All channels
            channels = {}
            for c in range(num_channels):
                channels[c] = []   
            classes = [1, 2] 
    else:

        # biomedical
        fs = 256
        trial_start_sec = 3
        trial_end_sec = 8

        num_channels = 16
        channels = {}
        for c in range(num_channels):
            channels[c] = []
        classes = [0, 1] 
    
    return channels, classes, fs, trial_start_sec, trial_end_sec


def reshape(data, num_trials, num_channels):
    '''
    Reshapes the raw EEG data into the 3d form (Trials * Channels * Readings per trial).
            
            Parameters:
                    data (2d array): The raw eeg data int the 2d format (Channels * Readings per channel)
                    num_trials (int): The total number of trials
                    num_channels (int): The total number of channels

            Returns:
                    reshaped_data: The data reshaped into 3d format
    '''
    readings_per_channel = len(data[0])
    readings_per_trial = readings_per_channel // num_trials
    reshaped_data = np.zeros((num_trials, num_channels, readings_per_trial))
    for i in range(num_channels):
        for j in range(num_trials):
            reshaped_data[j][i] = data[i][j*readings_per_trial: (j+1)*readings_per_trial]
    
    return reshaped_data

def reshape_back(reshaped_data):
    '''
    Reshapes the 3d form (Trials * Channels * Readings per trial) into its original 2d raw EEG form (Channels * Readings per channel).
            
            Parameters:
                    data (3d array): The reshaped raw_eeg (Trials * Channels * Readings per trial)

            Returns:
                    raw_eeg form data: The data reshaped back into its original 2d form.
    '''
    num_trials = reshaped_data.shape[0]
    num_channels = reshaped_data.shape[1]
    readings_per_trial = reshaped_data.shape[2]
    readings_per_channel = readings_per_trial * num_trials
    raw_eeg = np.zeros((num_channels, readings_per_channel))
    for i in range(num_channels):
        for j in range(num_trials):
            raw_eeg[i][j*readings_per_trial: (j+1)*readings_per_trial] = reshaped_data[j][i]
    
    return raw_eeg

def butter_bandpass_filter(signal, lowcut: int, highcut: int, fs: int, order: int = 1):
    '''
    Applies a butter band-pass filter to a signal (the raw EEG data).
            
            Parameters:
                    signal (nd array): The raw eeg data that should be filtered
                    lowcut (int): The low-cut frequency of the filter
                    highcut (int): The high-cut frequency of the filter
                    fs (int): The sampling frequency
                    order (int): The order od the butter filter

            Returns:
                    y: The filtered data
    '''
    # Calculate nyquist rate (half the sampling rate) to get the new lowcut and highcut for the filter
    nyquist_rate = 0.5 * fs
    low = lowcut / nyquist_rate
    high = highcut / nyquist_rate

    # Get the denominator and numerator polynomials of the Butterworth filter
    b, a = butter(order, [low, high], btype='band')

    # Filter the data using the designed filter
    y = lfilter(b, a, signal)
    return y



def butter_bandpass(bands: list, fs: int, raw, filterType='fir'):
    '''
    Applies a butter band-pass filter to a signal (the raw EEG data) over a range of bands.
            
            Parameters:
                    bands (list): A list of bands to apply band pass filters with
                    fs (int): The sampling frequency
                    raw (mne raw object): The raw mne object containing the EEG data
                    interval (int): The interval between each two frequencies between the low-cut and high-cut
                    filterType (str): The type of filter to use (fir or butter)

            Returns:
                    filtered_data (dict): The filtered data for all bands between low-cut and high-cut
    '''
    filtered_data = {}
       
    for band in bands:
        
        band_cuts = band.split('_')
        lowcut = int(band_cuts[0])
        highcut = int(band_cuts[1])
        
        # Bandpass filtering
        if filterType == 'fir':
            filtered_data[band] = raw.copy().filter(lowcut, highcut, fir_design='firwin', skip_by_annotation='edge')
        else:
            filtered_data[band] = butter_bandpass_filter(raw.get_data(), lowcut, highcut, fs)

    return filtered_data



def filterData(raw_eeg, num_trials: int, fs: int, bands: list):
    '''
    Applies a butter band-pass filter to a signal (the raw EEG data) over a range of bands.
            
            Parameters:
                    raw_eeg (2d ndarray): The 2d numpy array of the EEG data in the form (Channels * Reading per channel)
                    fs (int): The sampling frequency of the current data
                    bands (list): A list of bands to apply band pass filters with

            Returns:
                    filtered_data (dict): The filtered EEG data
    '''
    num_channels = raw_eeg.shape[0]
    ch_names = [str(i) for i in range(num_channels)]
    ch_types = ['eeg'] * num_channels

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=fs)
    raw = mne.io.RawArray(raw_eeg, info)

    filtered_data = butter_bandpass(bands, fs, raw)

    # Reshape filtered data into 3d form (Trials * Channels * Readings per Trial)
    for band in bands:
        filtered_data[band] = reshape(filtered_data[band].get_data(), num_trials, num_channels)

    return filtered_data
