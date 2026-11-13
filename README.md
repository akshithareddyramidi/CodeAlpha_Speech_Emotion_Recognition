# CodeAlpha Speech Emotion Recognition — VibeSense AI

VibeSense AI is a Speech Emotion Recognition project developed using Python, Streamlit, Librosa, TensorFlow, and a Convolutional Neural Network (CNN).

The application analyzes speech audio and predicts the possible emotion expressed in the voice using MFCC (Mel-Frequency Cepstral Coefficients) audio features.

## Features

- **Modern Dashboard:** Dark-themed interface with Overview, Predict Emotion, Audio Explorer, and About sections.
- **Audio Upload:** Upload WAV audio files for emotion analysis.
- **Live Microphone Recording:** Record your voice directly in the Predict Emotion section.
- **Audio Explorer Recording:** Record audio directly or upload a WAV file to explore its audio features.
- **Emotion Prediction:** Predicts one of eight emotion categories from speech audio.
- **Confidence Display:** Shows the prediction confidence score.
- **Emotion Score Visualization:** Displays scores for different emotion categories using a graph.
- **Prediction History:** Maintains up to 10 recent predictions during the current session.
- **Previous Analysis:** Select an earlier prediction to view its emotion, confidence, audio playback, and emotion score graph again.
- **Clear Audio:** Clear the current audio without deleting prediction history.
- **Clear History:** Clear prediction history separately.
- **Waveform Visualization:** Displays the waveform of the selected audio.
- **MFCC Visualization:** Displays Mel-Frequency Cepstral Coefficients extracted from the audio.

## Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

This project uses the **RAVDESS audio-only speech dataset** for model training.

The model recognizes eight emotion categories:

| Code | Emotion |
|---|---|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

Download the RAVDESS dataset from its official source and follow its usage and licensing terms.

Place the extracted speech WAV files inside the `dataset` folder. Keep the original filenames so the training script can identify the emotion labels.

## Installation and Setup

### 1. Clone or Download the Repository

Download the project from GitHub:

[CodeAlpha Speech Emotion Recognition Repository](https://github.com/akshithareddyramidi/CodeAlpha_Speech_Emotion_Recognition)

Extract the downloaded project folder and open it in VS Code.

### 2. Create a Virtual Environment

Open Windows PowerShell in the project folder and run:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install Required Libraries

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Pre-trained Model

The trained model is provided separately through the GitHub Release because of its file size.

Download `emotion_model.keras` from:

[VibeSense AI — Version 1.0.0 Release](https://github.com/akshithareddyramidi/CodeAlpha_Speech_Emotion_Recognition/releases/tag/v1.0.0)

Place the downloaded file inside the `models` folder, alongside `labels.json`.

The folder should contain:

```text
models/
    emotion_model.keras
    labels.json
```

**Note:** The pre-trained model is required to run emotion predictions. You do not need to train the model again if you use the provided model.

### 5. Add the Dataset (For Training)

If you want to train the model again, download the RAVDESS audio-only speech dataset and place the WAV files inside the `dataset` folder.

Example folder structure:

```text
dataset/
    Actor_01/
        03-01-01-01-01-01-01.wav
        ...
    Actor_02/
        ...
```

The training script processes the supported audio-only speech files.

### 6. Train the Model (Optional)

Run this command from the project root directory:

```powershell
python src/train.py
```

After successful training, the following files are generated:

```text
models/
    emotion_model.keras
    labels.json
```

### 7. Run the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal to use VibeSense AI.

## Project Structure

```text
CodeAlpha_Speech_Emotion_Recognition/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── RAVDESS audio files (used for training)
│
├── models/
│   ├── emotion_model.keras
│   └── labels.json
│
└── src/
    ├── __init__.py
    ├── audio_utils.py
    └── train.py
```

## How the Project Works

1. The user uploads a WAV file or records speech using a microphone.
2. Librosa processes the audio and extracts MFCC features.
3. The extracted features are passed to the trained CNN model.
4. The model predicts an emotion category.
5. The application displays the predicted emotion and confidence score.
6. Graphs help visualize emotion scores, waveform, and MFCC features.
7. Recent predictions can be revisited through session history.

## Model Evaluation

The CNN model was trained and evaluated using the RAVDESS speech dataset with a train/test split.

The trained model achieved approximately **40.3% test accuracy**.

Model performance depends on the dataset, audio quality, speakers, and training configuration. Further improvements can be explored through data augmentation, model tuning, and additional training.

Evaluation metrics such as accuracy, precision, recall, F1-score, and a confusion matrix can be used for more detailed performance analysis.

The displayed confidence score represents the model's prediction score and should not be interpreted as a guarantee that the predicted emotion is correct.

## Notes and Responsible Use

- This is an educational Speech Emotion Recognition project.
- Predictions are estimates based on learned audio patterns.
- Performance may vary with speakers, accents, languages, microphones, and background noise.
- Prediction history is maintained only during the current application session and is not permanently stored.
- The RAVDESS dataset should be used according to its licensing and attribution requirements.
- Emotion predictions should not be treated as definitive measurements of a person's actual emotional state.
- Understand, test, and explain the implementation before submitting the project.

## Author

Developed as part of the CodeAlpha Internship Project.
