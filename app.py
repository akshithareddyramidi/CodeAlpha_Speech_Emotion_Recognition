
import json
import io
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.audio_utils import load_audio, extract_features, emotion_name


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "emotion_model.keras"
LABELS_PATH = ROOT / "models" / "labels.json"


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="VibeSense | Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, #1c2850 0%, transparent 38%),
        radial-gradient(ellipse at 90% 15%, #183c49 0%, transparent 32%),
        #090d18;
    color: #eef3ff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

.hero {
    padding: 30px 34px;
    border: 1px solid #344363;
    border-radius: 22px;
    background: linear-gradient(
        115deg,
        rgba(35,51,91,.88),
        rgba(18,48,60,.78)
    );
}

.eyebrow {
    color: #65e4dc;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 700;
}

.hero h1 {
    font-size: 42px;
    line-height: 1.1;
    margin: 10px 0;
}

.hero p {
    color: #c1cde3;
    font-size: 16px;
    max-width: 720px;
}

.metric {
    background: rgba(20,31,53,.82);
    border: 1px solid #2d3c5d;
    border-radius: 16px;
    padding: 18px;
}

.metric .k {
    color: #9aa9c5;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric .v {
    font-family: 'Space Grotesk';
    font-size: 24px;
    font-weight: 700;
    margin-top: 5px;
}

.metric .small-muted {
    color: #9aa9c5;
    font-size: 13px;
}

.panel {
    background: rgba(17,26,45,.78);
    border: 1px solid #2b3958;
    padding: 20px;
    border-radius: 16px;
}

.small-muted {
    color: #9aa9c5;
    font-size: 13px;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

hr {
    border-color: #2b3958;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL FUNCTIONS
# --------------------------------------------------

def model_ready():
    return MODEL_PATH.exists() and LABELS_PATH.exists()


@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state["history"] = []

if "history_counter" not in st.session_state:
    st.session_state["history_counter"] = 0

if "microphone_recordings" not in st.session_state:
    st.session_state["microphone_recordings"] = []

if "mic_widget_version" not in st.session_state:
    st.session_state["mic_widget_version"] = 0

if "mic_recording_counter" not in st.session_state:
    st.session_state["mic_recording_counter"] = 0

if "selected_recording_id" not in st.session_state:
    st.session_state["selected_recording_id"] = None

# Audio Explorer microphone state
if "explorer_recording" not in st.session_state:
    st.session_state["explorer_recording"] = None

if "explorer_mic_version" not in st.session_state:
    st.session_state["explorer_mic_version"] = 0


# --------------------------------------------------
# MICROPHONE RECORDING MANAGEMENT
# --------------------------------------------------

def capture_microphone_recording(widget_key):

    audio = st.session_state.get(widget_key)

    if audio is None:
        return

    audio_bytes = audio.getvalue()

    if not audio_bytes:
        return

    st.session_state["mic_recording_counter"] += 1

    number = st.session_state["mic_recording_counter"]

    recording = {
        "id": number,
        "name": f"Microphone recording {number}",
        "data": audio_bytes,
        "time": datetime.now().strftime("%H:%M:%S")
    }

    st.session_state["microphone_recordings"].append(recording)

    st.session_state["selected_recording_id"] = number


def clear_prediction_audio():

    st.session_state["microphone_recordings"] = []
    st.session_state["selected_recording_id"] = None

    st.session_state["mic_widget_version"] += 1

    st.session_state.pop("predict_audio", None)
    st.session_state.pop("audio_source_choice", None)


def clear_prediction_history():

    st.session_state["history"] = []


# --------------------------------------------------
# AUDIO EXPLORER RECORDING MANAGEMENT
# --------------------------------------------------

def capture_explorer_recording(widget_key):

    audio = st.session_state.get(widget_key)

    if audio is not None:

        audio_bytes = audio.getvalue()

        if audio_bytes:
            st.session_state["explorer_recording"] = audio_bytes


def clear_explorer_audio():

    st.session_state["explorer_recording"] = None

    st.session_state["explorer_mic_version"] += 1

    st.session_state.pop("explore_audio", None)
    st.session_state.pop("explorer_source_choice", None)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">CODEALPHA · SPEECH EMOTION RECOGNITION</div>
    <h1>VibeSense <span style="color:#65e4dc">AI</span></h1>
    <p>
        Record or upload a voice sample, explore its audio features,
        and see which emotion the trained model identifies.
    </p>
</div>
""", unsafe_allow_html=True)


st.write("")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric">
        <div class="k">Project</div>
        <div class="v">Speech Analysis</div>
        <div class="small-muted">Audio classification</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric">
        <div class="k">Main techniques</div>
        <div class="v">MFCC + CNN</div>
        <div class="small-muted">Feature extraction and learning</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    status = "Ready" if model_ready() else "Unavailable"

    st.markdown(f"""
    <div class="metric">
        <div class="k">Model status</div>
        <div class="v">{status}</div>
        <div class="small-muted">Trained speech model</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab_home, tab_predict, tab_explore, tab_about = st.tabs([
    "⌂  Overview",
    "🎙️  Predict emotion",
    "📊  Audio explorer",
    "About"
])


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

with tab_home:

    st.subheader("Welcome")

    st.write(
        "Use a microphone recording or an existing WAV file "
        "to explore speech and estimate the emotion expressed "
        "through its vocal characteristics."
    )

    a, b = st.columns(2)

    with a:
        st.markdown("""
        <div class="panel">
            <h3>01 · Record or upload</h3>
            <p class="small-muted">
                Record your voice directly or select an existing
                WAV audio file from your computer.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="panel">
            <h3>02 · Analyze speech</h3>
            <p class="small-muted">
                Extract MFCC features and use the trained model
                to estimate the emotion in the recording.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    if model_ready():
        st.success("The trained model is available.")
    else:
        st.warning(
            "The trained model could not be found. "
            "Check the models folder and training files."
        )


# --------------------------------------------------
# PREDICT EMOTION
# --------------------------------------------------

with tab_predict:

    st.subheader("Emotion prediction")

    st.write(
        "Record your voice or upload a WAV file to analyse speech."
    )

    col1, col2 = st.columns(2)

    mic_widget_key = (
        f"recorded_audio_{st.session_state['mic_widget_version']}"
    )

    with col1:

        st.markdown("### 🎙️ Microphone")

        st.audio_input(
            "Record your voice",
            sample_rate=22050,
            key=mic_widget_key,
            on_change=capture_microphone_recording,
            args=(mic_widget_key,)
        )

        recordings = st.session_state["microphone_recordings"]

        if recordings:

            st.caption(
                f"{len(recordings)} microphone recording(s) available."
            )

            recording_ids = [
                item["id"] for item in recordings
            ]

            selected_id = st.session_state.get(
                "selected_recording_id"
            )

            if selected_id not in recording_ids:
                st.session_state["selected_recording_id"] = recording_ids[-1]

            def recording_label(recording_id):

                for item in recordings:
                    if item["id"] == recording_id:
                        return f"{item['name']} · {item['time']}"

                return str(recording_id)

            selected_recording_id = st.selectbox(
                "Select a recording",
                options=recording_ids,
                format_func=recording_label,
                key="selected_recording_id"
            )

        else:
            st.caption("No microphone recordings yet.")

    with col2:

        st.markdown("### 📁 Audio upload")

        uploaded_audio = st.file_uploader(
            "Choose a WAV file",
            type=["wav"],
            key="predict_audio"
        )

    st.write("")

    # --------------------------------------------------
    # CLEAR AUDIO
    # --------------------------------------------------

    clear_col, info_col = st.columns([1, 3])

    with clear_col:

        st.button(
            "Clear Audio",
            key="clear_prediction_audio_button",
            on_click=clear_prediction_audio,
            use_container_width=True
        )

    with info_col:

        st.caption(
            "Clears the current microphone recordings and uploaded "
            "audio. Your prediction history remains unchanged."
        )

    # --------------------------------------------------
    # SELECT AUDIO SOURCE
    # --------------------------------------------------

    recordings = st.session_state["microphone_recordings"]

    selected_recording = None

    if recordings:

        selected_id = st.session_state.get(
            "selected_recording_id"
        )

        for item in recordings:
            if item["id"] == selected_id:
                selected_recording = item
                break

    audio_bytes = None
    audio_source = None

    if selected_recording is not None and uploaded_audio is not None:

        source_choice = st.radio(
            "Choose which audio to analyse",
            ["Microphone recording", "Uploaded WAV"],
            horizontal=True,
            key="audio_source_choice"
        )

        if source_choice == "Microphone recording":
            audio_bytes = selected_recording["data"]
            audio_source = selected_recording["name"]
        else:
            audio_bytes = uploaded_audio.getvalue()
            audio_source = uploaded_audio.name

    elif selected_recording is not None:

        audio_bytes = selected_recording["data"]
        audio_source = selected_recording["name"]

    elif uploaded_audio is not None:

        audio_bytes = uploaded_audio.getvalue()
        audio_source = uploaded_audio.name

    # --------------------------------------------------
    # AUDIO ANALYSIS
    # --------------------------------------------------

    if audio_bytes is not None:

        st.markdown("#### Selected audio")

        st.audio(audio_bytes, format="audio/wav")

        try:

            audio_buffer = io.BytesIO(audio_bytes)

            signal, sr = load_audio(audio_buffer)

            if len(signal) == 0:
                raise ValueError("The selected recording is empty.")

            duration = len(signal) / sr

            st.caption(
                f"Duration: {duration:.2f} seconds  ·  "
                f"Sample rate: {sr:,} Hz"
            )

            if st.button(
                "Analyze voice",
                type="primary",
                key="analyze_voice_button"
            ):

                if not model_ready():

                    st.error(
                        "The trained model is missing. "
                        "Please check the models folder."
                    )

                else:

                    with st.spinner("Analyzing the recording..."):

                        model = load_model()
                        labels = load_labels()

                        features = extract_features(signal, sr)

                        x = features[None, ..., None]

                        probs = model.predict(x, verbose=0)[0]

                        order = np.argsort(probs)[::-1]

                        best = int(order[0])

                        predicted_emotion = emotion_name(
                            labels[best]
                        )

                        confidence = float(probs[best]) * 100

                        chart_data = {
                            emotion_name(labels[i]): float(probs[i])
                            for i in order
                        }

                    # Save complete result in session history.

                    st.session_state["history_counter"] += 1

                    new_prediction = {
                        "id": st.session_state["history_counter"],
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Audio": audio_source,
                        "Predicted emotion": predicted_emotion,
                        "Confidence": f"{confidence:.1f}%",
                        "confidence_value": float(probs[best]),
                        "scores": chart_data,
                        "audio_data": audio_bytes
                    }

                    st.session_state["history"].insert(
                        0,
                        new_prediction
                    )

                    st.session_state["history"] = (
                        st.session_state["history"][:10]
                    )

                    st.success("Analysis completed.")

                    st.markdown("---")
                    st.markdown("### Analysis result")
                    st.markdown(f"## {predicted_emotion}")

                    st.progress(
                        float(probs[best]),
                        text=f"Model confidence: {confidence:.1f}%"
                    )

                    st.markdown("#### Emotion scores")
                    st.bar_chart(chart_data)

        except Exception as exc:

            st.error(
                "The audio could not be analyzed. "
                "Please try another clear WAV recording."
            )

            st.caption(f"Details: {exc}")

    else:

        st.info(
            "Record your voice or upload a WAV file to get started."
        )


    # --------------------------------------------------
    # RECENT PREDICTIONS
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("Recent predictions")

    history = st.session_state["history"]

    if history:

        st.caption(
            "Your 10 most recent predictions in this session."
        )

        history_col, button_col = st.columns([3, 1])

        with history_col:
            st.caption(
                "History is temporary and is not stored permanently."
            )

        with button_col:

            st.button(
                "Clear History",
                key="clear_history_button",
                on_click=clear_prediction_history,
                use_container_width=True
            )

        # Show a simple history table without internal data.

        display_history = [
            {
                "Time": item["Time"],
                "Audio": item["Audio"],
                "Predicted emotion": item["Predicted emotion"],
                "Confidence": item["Confidence"]
            }
            for item in history
        ]

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # VIEW PREVIOUS ANALYSIS
        # --------------------------------------------------

        st.markdown("---")

        st.subheader("View Previous Analysis")

        st.write(
            "Select any previous prediction to view its complete result again."
        )

        history_ids = [
            item["id"] for item in history
        ]

        def history_label(prediction_id):

            for item in history:
                if item["id"] == prediction_id:
                    return (
                        f"{item['Audio']} · "
                        f"{item['Predicted emotion']} · "
                        f"{item['Time']}"
                    )

            return str(prediction_id)

        selected_history_id = st.selectbox(
            "Choose a previous prediction",
            options=history_ids,
            format_func=history_label,
            key="view_history_selection"
        )

        selected_history = next(
            (
                item for item in history
                if item["id"] == selected_history_id
            ),
            None
        )

        if selected_history is not None:

            st.markdown("### Previous analysis result")

            st.markdown(
                f"## {selected_history['Predicted emotion']}"
            )

            st.progress(
                selected_history["confidence_value"],
                text=(
                    "Model confidence: "
                    f"{selected_history['Confidence']}"
                )
            )

            st.markdown("#### Emotion scores")

            st.bar_chart(selected_history["scores"])

            st.markdown("#### Original audio")

            st.audio(
                selected_history["audio_data"],
                format="audio/wav"
            )

            st.caption(
                f"Audio: {selected_history['Audio']} · "
                f"Analysed at: {selected_history['Time']}"
            )

    else:

        st.info(
            "No predictions have been saved in this session yet."
        )


# --------------------------------------------------
# AUDIO EXPLORER
# --------------------------------------------------

with tab_explore:

    st.subheader("Audio explorer")

    st.write(
        "Record your voice or upload a WAV file to view "
        "its waveform and MFCC features."
    )

    record_col, upload_col = st.columns(2)

    with record_col:

        st.markdown("### 🎙️ Record Audio")

        explorer_mic_key = (
            f"explorer_mic_{st.session_state['explorer_mic_version']}"
        )

        st.audio_input(
            "Record your voice",
            sample_rate=22050,
            key=explorer_mic_key,
            on_change=capture_explorer_recording,
            args=(explorer_mic_key,)
        )

    with upload_col:

        st.markdown("### 📁 Upload Audio")

        explorer_file = st.file_uploader(
            "Choose a WAV file",
            type=["wav"],
            key="explore_audio"
        )

    # --------------------------------------------------
    # SELECT EXPLORER AUDIO
    # --------------------------------------------------

    recording_data = st.session_state["explorer_recording"]

    audio_data = None

    if recording_data is not None and explorer_file is not None:

        source_choice = st.radio(
            "Choose audio for visualization",
            ["Microphone recording", "Uploaded WAV"],
            horizontal=True,
            key="explorer_source_choice"
        )

        if source_choice == "Microphone recording":
            audio_data = recording_data
        else:
            audio_data = explorer_file.getvalue()

    elif recording_data is not None:

        audio_data = recording_data

    elif explorer_file is not None:

        audio_data = explorer_file.getvalue()

    # --------------------------------------------------
    # CLEAR EXPLORER AUDIO
    # --------------------------------------------------

    st.button(
        "Clear Audio",
        key="clear_explorer_button",
        on_click=clear_explorer_audio
    )

    # --------------------------------------------------
    # WAVEFORM AND MFCC VISUALIZATION
    # --------------------------------------------------

    if audio_data is not None:

        try:

            signal, sr = load_audio(io.BytesIO(audio_data))

            if len(signal) == 0:
                raise ValueError("The selected recording is empty.")

            st.markdown("---")

            st.markdown("### Audio preview")

            st.audio(audio_data, format="audio/wav")

            st.caption(
                f"Duration: {len(signal) / sr:.2f} seconds  ·  "
                f"Sample rate: {sr:,} Hz  ·  "
                f"Samples: {len(signal):,}"
            )

            left, right = st.columns(2)

            # Waveform

            with left:

                st.markdown("**Waveform**")

                fig, ax = plt.subplots(figsize=(7, 2.7))

                times = np.arange(len(signal)) / sr

                ax.plot(
                    times,
                    signal,
                    linewidth=0.55
                )

                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel("Amplitude")
                ax.grid(alpha=0.2)

                fig.patch.set_alpha(0)
                ax.set_facecolor("#111a2d")

                st.pyplot(fig, use_container_width=True)

                plt.close(fig)

            # MFCC

            with right:

                st.markdown("**MFCC feature map**")

                mfcc = librosa.feature.mfcc(
                    y=signal,
                    sr=sr,
                    n_mfcc=20
                )

                fig2, ax2 = plt.subplots(figsize=(7, 2.7))

                image = ax2.imshow(
                    mfcc,
                    aspect="auto",
                    origin="lower"
                )

                ax2.set_xlabel("Time frames")
                ax2.set_ylabel("MFCC coefficient")

                fig2.colorbar(
                    image,
                    ax=ax2,
                    fraction=0.025
                )

                fig2.patch.set_alpha(0)

                st.pyplot(fig2, use_container_width=True)

                plt.close(fig2)

        except Exception as exc:

            st.error(
                "The audio visualization could not be completed."
            )

            st.caption(f"Details: {exc}")

    else:

        st.info(
            "Record your voice or upload a WAV file "
            "to view the waveform and MFCC features."
        )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

with tab_about:

    st.subheader("About VibeSense AI")

    st.write(
        "VibeSense AI is a speech emotion recognition project "
        "that studies vocal characteristics and estimates the "
        "emotion expressed in a speech recording."
    )

    st.markdown("""
    **Emotion categories**

    - Neutral
    - Calm
    - Happy
    - Sad
    - Angry
    - Fearful
    - Disgust
    - Surprised

    **Technologies used**

    - Python
    - Streamlit
    - Librosa
    - NumPy
    - TensorFlow / Keras
    - Matplotlib

    **Project features**

    - Microphone recording
    - WAV audio upload
    - Speech emotion prediction
    - Waveform visualization
    - MFCC feature visualization
    - Recent prediction history
    - Previous analysis viewing
    """)

    st.info(
        "The result is an estimate based on vocal patterns. "
        "It does not determine a person's actual feelings, "
        "and predictions may vary with speakers and recording conditions."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<hr>
<p style="text-align:center;color:#8292b1;font-size:12px">
    VibeSense AI · Speech Emotion Recognition
</p>
""", unsafe_allow_html=True)
