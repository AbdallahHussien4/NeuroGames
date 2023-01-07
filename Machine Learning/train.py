from Preprocessing import *
from Features_Extraction import *
from Features_Selection import *
from Classification import *
from io_modules import *
import pickle
from IPython.utils import io


# The frequency pass bands used in the filtering phase. Each string is parsed to apply the filter between 
bands = ['4_8', '8_12', '12_16', '16_20', '20_24', '24_28', '28_32', '32_36', '36_40']

# The number of features to select after applying CSP transformation for each band
n_components = 4

# The top number of features to select from all bands after CSP, done using MIBIF
k_features = 9


# Variables for controlling how different types of data are parsed and saved
dataset_type = 'self_recorded'
headset = 'insight'
subject = 'francois'

data_source_file_name = 'data' + '/' + subject + '.csv'

parsed_data_file_name = 'csv_data' + '/' + dataset_type + '/' + subject + '.csv'

labels_file_name = 'data' + '/' + subject + '_labels.csv'

# Initialize some variables for data parsing and preprocessing. Different datasets have different formats and dimensions 
channels, classes, fs, trial_start_sec, trial_end_sec = initializeVars(dataset_type, headset)

# Read the self recorded data by specifying the trial start and end for each experiment to trim it correctly.
#   Then save the parsed data into a separate csv file.
ReadEmotivData(fs, trial_start_sec, trial_end_sec, data_source_file_name, labels_file_name, parsed_data_file_name)

# Load the parsed data from the newly saved csv files.
raw_eeg, labels = loadFromCSV(parsed_data_file_name, labels_file_name)

# Get the number of trials recorded in the session which is equal to the number of labels.
num_trials_train = len(labels)

# Apply band pass filtering to the raw EEG data.
with io.capture_output() as captured:
    filtered_train = filterData(raw_eeg, num_trials_train, fs, bands)

# Extract n CSP features and get the transformation matrices learned.
csp_train, weights, y_train = extract_CSP(filtered_train, labels, classes, bands, n_components)

# Get the top k features and their indexes to be used on test data.
X_train, indexes = mutual_information(csp_train, y_train, classes, k_features)

# Save the parameters used in filtering, and learnt by CSP & MIBIF. 
np.save('bands', bands)
np.save('weights', weights)
np.save('indexes', indexes)

# Use LDA classifier and fit it on the training data.
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)

# Save the LDA model
model_filename = 'finalized_model.sav'
pickle.dump(clf, open(model_filename, 'wb'))
