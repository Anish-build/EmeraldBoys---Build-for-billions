import os
import librosa
import pandas as pd
import numpy as np

DATASET_PATH = "audio_data"

features = []

for label in ["normal", "abnormal"]:

    folder_path = os.path.join(DATASET_PATH, label)

    print("\nProcessing:", label)

    for file_name in os.listdir(folder_path):

        if file_name.lower().endswith(".wav"):

            file_path = os.path.join(folder_path, file_name)

            print("Processing:", file_name)

            try:
                audio, sample_rate = librosa.load(
                    file_path,
                    sr=None
                )

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sample_rate,
                    n_mfcc=13
                )

                zcr = librosa.feature.zero_crossing_rate(audio)
                rms = librosa.feature.rms(y=audio)

                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio,
                    sr=sample_rate
                )

                spectral_bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio,
                    sr=sample_rate
                )

                spectral_rolloff = librosa.feature.spectral_rolloff(
                    y=audio,
                    sr=sample_rate
                )

                row = {
                    "file_name": file_name,
                    "label": label,

                    "zcr_mean": np.mean(zcr),
                    "zcr_std": np.std(zcr),

                    "rms_mean": np.mean(rms),
                    "rms_std": np.std(rms),

                    "spectral_centroid_mean": np.mean(spectral_centroid),
                    "spectral_centroid_std": np.std(spectral_centroid),

                    "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
                    "spectral_bandwidth_std": np.std(spectral_bandwidth),

                    "spectral_rolloff_mean": np.mean(spectral_rolloff),
                    "spectral_rolloff_std": np.std(spectral_rolloff)
                }

                for i in range(13):
                    row[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
                    row[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

                features.append(row)

            except Exception as e:
                print("ERROR processing", file_name, ":", e)

df = pd.DataFrame(features)

df.to_csv(
    "audio_features.csv",
    index=False
)

print("\n================================")
print("FEATURE EXTRACTION COMPLETED")
print("================================")

print("Total audio files processed:", len(df))
print("Total features extracted:", len(df.columns) - 2)
print("CSV file created:", "audio_features.csv")

print("\nClass distribution:")
print(df["label"].value_counts())