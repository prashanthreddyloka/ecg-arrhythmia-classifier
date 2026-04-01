# ECG Arrhythmia Classifier

This is a real standalone time-series project for ECG arrhythmia detection using the MIT-BIH Arrhythmia Database. It uses heartbeat-centered ECG windows, sliding-window normalization, Fourier-transform features, and a bidirectional LSTM classifier to detect abnormal beats.

## What The Project Does

- downloads or reads the real MIT-BIH Arrhythmia Database
- extracts heartbeat windows around expert annotations
- applies sliding-window normalization to reduce baseline drift and local noise
- computes FFT magnitude features from each beat
- trains a bidirectional LSTM classifier with class weighting
- reports ROC AUC, accuracy, sensitivity, specificity, and precision

## Dataset

This project is built using official MIT-BIH Arrhythmia Database on PhysioNet:

- MIT-BIH Arrhythmia Database: [physionet.org/physiobank/database/mitdb](https://physionet.org/physiobank/database/mitdb/)

Implementation note:
- the train/test split uses the common inter-patient split with DS1 for training and DS2 for testing
- that split choice is an implementation inference based on common MIT-BIH evaluation practice and is not copied from the PhysioNet page itself

## Project Structure

```text
ecg-arrhythmia-classifier/
  README.md
  requirements.txt
  data/
    raw/
    processed/
  artifacts/
  src/
    data.py
    model.py
    prepare_data.py
    train.py
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Prepare the MIT-BIH dataset:

```powershell
python src/prepare_data.py --download
```

4. Train the classifier:

```powershell
python src/train.py
```

## Data Pipeline

1. Read MIT-BIH records and annotation files.
2. Extract one ECG beat window around each annotated heartbeat.
3. Keep supported beat symbols and map them to:
   normal = 0
   abnormal = 1
4. Normalize each beat with a sliding local window.
5. Compute FFT magnitude features.
6. Cache the processed train/test arrays to `data/processed/mitbih_beats.npz`.

## Model

The model combines:

- a bidirectional LSTM over the beat waveform
- FFT magnitude features concatenated to the temporal embedding
- a dense classification head for binary abnormal-beat detection

## Key Commands

Prepare data with default settings:

```powershell
python src/prepare_data.py --download
```

Use a different lead or beat window size:

```powershell
python src/prepare_data.py --download --lead-index 1 --pre-samples 160 --post-samples 200
```

Train for more epochs:

```powershell
python src/train.py --epochs 25 --batch-size 256
```

## Current Limitation In This Workspace

The code is prepared for a real dataset workflow, but execution depends on a working Python environment with package installation available. If Python or pip is missing from PATH, install Python first or run with an available interpreter.
