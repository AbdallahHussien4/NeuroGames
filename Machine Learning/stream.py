from random import randint
from sqlite3 import Timestamp
from time import sleep
import sys
from Preprocessing import *
from Features_Extraction import *
from Features_Selection import *
from Classification import *
from io_modules import *
import pickle
from pylsl import StreamInlet, resolve_stream
import os


# The action is sent to the game by running this py script from a node script.
# Then the py script can send the action by printing it and flushing the output stream.
# The node script listens on the std output stream and fetches the action accordingly. 


# Functions to disable printing of the filter function, so that the data is sent to the node script correctly
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

# Resolve an EEG stream on the lab network
streams = resolve_stream('type', 'EEG')

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

# Loading CSP and MIBIF parameters from saved numpy files
weights = np.load('weights.npy')
indexes = np.load('indexes.npy')
bands = np.load('bands.npy')
model = pickle.load(open('finalized_model.sav', 'rb'))

# Number of channels used (a constant 5 since we use Emotiv Insight)
NUM_CHANNELS = 5

# The sampling frequency used (128 Hz in Emotiv Insight)
FS = 128

# The number of samples needed to predict one action
SAMPLES_PER_ACTION = 256

# The number of trials to filter the input data appropriately (It is always one in the case of streaming since we predict each action seperately)
NUM_TRIALS = 1

# A variable to keep track of how many samples were fetched
n_samples = 0

# A 2d list to contain the readings for each trial
raw_eeg = [[], [], [], [], []]
while True:
    # fetch a sample from the inlet
    sample, timestamp = inlet.pull_sample()
    # Check that the timestamp isn't None (data was streamed properly) and the sample contains some readings
    if timestamp != None and len(sample) != 0:
        try:
            # Get the readings of the five channels from the current sample
            for i in range(NUM_CHANNELS):
                raw_eeg[i].append(sample[3 + i])
            
            n_samples += 1

            # Check if we reached the number of samples needed to predict the action
            if(n_samples == SAMPLES_PER_ACTION):

                # Block printing inside the filtering function so that no data is sent to the node interface by mistake
                blockPrint()

                # Apply band pass filtering to the raw EEG data
                filtered_test = filterData(np.array(raw_eeg), NUM_TRIALS, FS, bands)
                enablePrint()

                # Extract the top features from the filtered data by transforming using CSP 
                #   then selecting the most important features using the indexes list learnt through MIBIF
                top_features = extractTopFeatures(filtered_test, bands, weights, indexes)
                
                # Use the saved LDA model to predict an action with probabilities to know how certain the model is about the prediction
                probs = model.predict_proba(top_features)
                
                right_action_probability = probs[0][0]
                left_action_probability = probs[0][1]

                # If the probability of either action (left/right) is at least 80% 
                #   then send that action to the game throught the node interface
                if right_action_probability >= 0.8:
                    print(1)
                elif left_action_probability >= 0.8:
                    print(2)

                # If the model isn't certain of the action then the user was neutral 
                # and we send 0 so that the game knows that it shouldn't do anything now
                else:
                    print(0)
                
                # Flush the output stream for the node script to fetch it
                sys.stdout.flush()

                # Reset the variables again before the following iteration
                n_samples = 0
                raw_eeg = [[], [], [], [], []]
            
        # Whenever an exception is caught, write that error message iniside a text file. 
        except Exception as e:
            file_object = open('err.txt', 'a')
            file_object.write(e + "\n")
            file_object.close()
