# ASL Recognition System

This system recognizes American Sign Language (ASL) alphabet signs and builds words as users make different hand signs.

## Features

- Recognizes ASL alphabet signs (A-Z)
- Builds words letter by letter
- Uses thumbs up gesture to complete words
- Displays completed words prominently

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe (<0.10, the code currently relies on the legacy `mp.solutions` API)
- NumPy
- scikit-learn

> ⚠️ If you install a newer MediaPipe release (0.10 or later) you may see
> ``AttributeError: module 'mediapipe' has no attribute 'solutions'``. Either
> downgrade (`pip install 'mediapipe<0.10'`) or update the code to use the
> new Tasks API.

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`

> **Note:** the recognizer works in two modes:
>
> * **Legacy** – with `mediapipe<0.10` the original `mp.solutions` API is used.
> * **Tasks** – with `mediapipe>=0.10` the newer Tasks API is used. A
>   `hand_landmarker.task` model bundle is required; the code will attempt to
>   download a default copy for you or you can supply your own by setting the
>   `HAND_LANDMARKER_MODEL` environment variable.


## Usage

1. Run the main application: `python run.py`
2. Show ASL letter signs to the camera
3. Use thumbs up gesture to complete words

## Data Collection (Optional)

To collect your own training data:

1. Run `python collect_training_data.py`
2. Follow the prompts to collect sign language data
3. Train the model with `python train_model.py`

## File Structure

- `asl_recognizer.py`: Main recognition system
- `run.py`: Launcher script
- `collect_training_data.py`: Tool to collect training data
- `train_model.py`: Script to train the recognition model
- `data/`: Directory for training data
- `models/`: Directory for trained models
- `logs/`: Directory for logs
