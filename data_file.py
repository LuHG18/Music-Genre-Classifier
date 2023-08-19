# The code for the MFCC extraction is largely lifted from Valerio Velardo's YouTube course on machine learning for audio. The link to his entire project is below. 
# https://github.com/musikalkemist/DeepLearningForAudioWithPython/tree/master

import os
from librosa import load
from librosa.feature import mfcc
from math import ceil
import json

dataset_folder = "Data"
json_path = "data.json"

track_duration = 30 #Each track is 30 seconds long in this dataset
sample_rate = 22050 #Standard sampling rate that will work for this basic level classification

# We are using num_segments to break down each WAV file into 3 segments which will give our model more data
def save_mfcc(dataset_path, json_file, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    data = {

        # We create a dictionary that keeps track of each music genre and the MFCC extraction, and then labels each MFCC vector to the proper genre
        "genres": [],
        "mfccs": [],
        "labels": []
    }
    numSamplesPerTrack = sample_rate * track_duration
    numSamplesPerSegment = int(numSamplesPerTrack / num_segments)
    expectedMfccVectorCountPerSeg = ceil(numSamplesPerSegment / hop_length)

    print(dataset_path)
    label_counter = -1  # Initialize a label counter for assigning labels

    for i, (dirpath, _, filename) in enumerate(os.walk(dataset_path)):
        
        print("dirpath:", dirpath)
        print("dataset_path:", dataset_path)
        # Ensures that we are not in the root
        if dirpath != dataset_path:
            # Using walk(), we recursively go through the directory path, each folder in the directory, and each file in these folders
            genre_label = os.path.basename(dirpath)
            # Only append the genre label if it's not "genres_original" 
            if genre_label != "genres_original":
                data["genres"].append(genre_label)
                print("\nProcessing: {}".format(genre_label))
                # Increment the label counter for each new genre encountered to prevent duplicate genre labels
                label_counter += 1

            for file in filename:
                # .DS_Store is a default file on Mac dealing with folder attributes that causes a major issue when dealing with non-MacOS systems
                if file == ".DS_Store":
                    continue
                print(file)
                # Produce the correct file path using os.path.join()
                file_path = os.path.join(dirpath, file)
                print(file_path)
                # Define the audio signal and set our chosen sample rate to 22050 Hz using the librosa library 
                signal, sr = load(file_path, sr=sample_rate)

                # Obtain an interval of the audio signal corresponding to which segment we are addressing
                for segment in range(num_segments):
                    start_sample = numSamplesPerSegment * segment
                    finish_sample = start_sample + numSamplesPerSegment
                    # Gathers the MFCC results for each segment in the file and appends this data to the dictionary.
                    mfcc_result = mfcc(y=signal[start_sample:finish_sample], sr=sample_rate, n_fft=n_fft,
                                       n_mfcc=n_mfcc, hop_length=hop_length)
                    # Get the transpose of this array 
                    mfcc_result = mfcc_result.T

                    #It's possible for the data to not be exactly the expeced duration so we account for that before storing the features
                    if len(mfcc_result) == expectedMfccVectorCountPerSeg:
                        data["mfccs"].append(mfcc_result.tolist())
                        data["labels"].append(label_counter)  # Use the label_counter instead of i
                        print("{}, segment:{}".format(file_path, segment+1))

    with open(json_file, "w") as output:
        json.dump(data, output, indent=4)

if __name__ == "__main__":
    save_mfcc(dataset_folder, json_path)

