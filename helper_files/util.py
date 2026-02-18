from collections import Counter
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt



def listShape(list):
    if len(list) == 0:
        return f"(0, 0)"
    return f"({len(list):,}, {len(list[0]):,})"





def display(derived_samples: np.array, labels: np.array):
    print(f"\nNumber of labels: {labels.size:,}")
    print(f"Number of samples: {derived_samples.shape[0]:,}")
    print()
    print(f"Shape of labels: {labels.shape}")
    print(f"Shape of samples: {derived_samples.shape}")
    print()

    if (derived_samples.shape[0] == 0):
        print("WARNING: THERE WAS NO DATA DERIVED.")
        print("Check that the dataset exists.")
        return
    
    print(f"One sample: \n{derived_samples[0]}")

    print()
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("The counts:")
    for lab, c in zip(unique_labels, counts):
        print(f"{str(f"{c:,}"):8s} \"{lab}\"")






def showSpectrum(spectrum):

    # Convert to dB for better visibility
    S_db = 10 * np.log10(spectrum + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        S_db.T,                 # transpose so freq is vertical
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency Bin")
    plt.title("Spectrogram")
    plt.show()








def printMetrics(
    base_name, 
    model, 
    perc_accuracy, 
    notes: str,
    cr, 
    cm, 
    labels,
    max_files,
    window_size,
    overlap,
    num_features,
    num_training_files,
    num_evaluation_files,
    training_dataset: str="not provided",
    eval_dataset: str="not provided",
):
    print(f"Model: {model}")
    print(f"Accuracy: {perc_accuracy:.2f}%")
    print(f"Date: {datetime.now()}")
    print(f"Training Dataset: {training_dataset}")
    print(f"Evaluation Dataset: {eval_dataset}")
    print(f"Notes: {notes}")
    print()
    
    print(f"=========PARAMETERS=========")
    print(f"WINDOW_LEN = {window_size:,}")
    print(f"OVERLAP = {overlap}")
    print(f"NUM_FEATURES = {num_features}")
    print(f"NUM_TRAINING_FILES = {num_training_files}")
    print(f"NUM_EVALUATION_FILES = {num_evaluation_files}")

    print("")
    print("Count\tLabel")
    counts = Counter(labels)
    for item, count in counts.items():
        print(f"{count:,}\t{item}")
    print("")
    print("Classification Report:")
    print(f"{cr}\n")
    print("Confusion Matrix:")
    print(f"{cm}")




def saveMetricsToFile(
    base_name, 
    model, 
    perc_accuracy, 
    notes: str,
    cr, 
    cm, 
    labels,
    max_files,
    window_size,
    overlap,
    num_features,
    num_training_files,
    num_evaluation_files,
    training_dataset: str="not provided",
    eval_dataset: str="not provided",
): 
    # metric's functions 
    def checkForFilename(base_name): 
        ext = ".txt"
        i = 1
        filename = f"{base_name}{ext}"
        while os.path.exists(filename):
            filename = f"{base_name}-{i}{ext}"
            i += 1
        return filename

    filename = checkForFilename(f"./metrics/{base_name}_{(perc_accuracy * 100):.0f}")

    with open(filename, "w") as f:
        f.write(f"Model: {model}\n")
        f.write(f"Accuracy: {perc_accuracy:.2f}%\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Training Dataset: {training_dataset}\n")
        f.write(f"Evaluation Dataset: {eval_dataset}\n")
        f.write(f"Notes: {notes}\n\n")
        f.write(f"=========PARAMETERS=========\n")
        f.write(f"WINDOW_LEN = {window_size:,}\n")
        f.write(f"OVERLAP = {overlap}\n")
        f.write(f"NUM_FEATURES = {num_features}\n")
        f.write(f"NUM_TRAINING_FILES = {num_training_files}\n")
        f.write(f"NUM_EVALUATION_FILES = {num_evaluation_files}\n")

        f.write("\n")
        f.write("Count\tLabel\n")
        counts = Counter(labels)
        for item, count in counts.items():
            f.write(f"{count:,}\t{item}\n")
        f.write("\n")
        f.write("Classification Report:\n")
        f.write(f"{cr}\n\n")
        f.write("Confusion Matrix:\n")
            
        f.write(f"{cm}\n")


