# README

## Overview

This repository contains two projects: `sign2speech` and `speech2sign`. Each project is designed to convert between sign language and speech/text using various technologies.

## Installation Instructions

### Step 1: Create a Virtual Environment

First, navigate to the specific folder (`sign2speech` or `speech2sign`) you want to run. Open your terminal or command prompt and run the following command to create a virtual environment.

#### On Windows:
```bash
python -m venv env
```

#### On Linux/MacOS:
```bash
python3 -m venv env
```

### Step 2: Activate the Virtual Environment

Activate the virtual environment you just created:

#### On Windows:
```bash
.\env\Scripts\activate
```

#### On Linux/MacOS:
```bash
source env/bin/activate
```

Once the virtual environment is activated, your command prompt or terminal should show the environment name (e.g., `(env)`).

### Step 3: Install Dependencies

install the necessary Python packages listed in `requirements.txt` by running:

```bash
pip install -r requirements.txt
```

This will install all the dependencies required for the project.

## File Overview

### Folder: `sign2speech`

- **gradio_elevenlabs_video_to_speech_and_text.py**: A Python script that uses Gradio to create a web interface. It takes a video of sign language and converts it to text and speech using the ElevenLabs API.
- **gradio_google_video_to_speech_and_text.py**: A Python script similar to the one above, but it uses Google Text-to-Speech for speech synthesis.
- **Amiri-Regular.ttf**: A font file used in the code for text rendering.
- **Arsl_gesture_recognizer.task**: A MediaPipe gesture recognition model that recognizes Arabic language letters.

### Folder: `speech2sign`

- **gradio_speech_to_text_and_video.py**: A Python script that uses Gradio to create a web interface. It takes speech input, converts it to text using Google Speech-to-Text, and then converts the text to a video of sign language.

## Usage

### For `sign2speech`:

1. Navigate to the `sign2speech` folder and activate the virtual environment and .
2. Run one of the Python scripts:
   - For ElevenLabs speech synthesis:
     ```bash
     python gradio_elevenlabs_video_to_speech_and_text.py
     ```
   - For Google Text-to-Speech:
     ```bash
     python gradio_google_video_to_speech_and_text.py
     ```
3. Access the web interface by clicking on the link provided in the terminal or by navigating to `http://localhost:7860` in your web browser.

### For `speech2sign`:

1. Activate the virtual environment and navigate to the `speech2sign` folder.
2. Run the Python script:
   ```bash
   python gradio_speech_to_text_and_video.py
   ```
3. Access the web interface by clicking on the link provided in the terminal or by navigating to `http://localhost:7860` in your web browser.

## Dependencies

Make sure you have the following installed:

- ElevenLabs API key (for `gradio_elevenlabs_video_to_speech_and_text.py`)
- Google Cloud Speech-to-Text and Text-to-Speech API keys (for `gradio_speech_to_text_and_video.py` and `gradio_google_video_to_speech_and_text.py`)


## Deployment

Explore the live demos of the projects on Hugging Face Spaces:

- [Sahla-speech2sign](https://huggingface.co/spaces/AdelShousha/Sahla-speech2sign)
- [Sahla-sign2speech](https://huggingface.co/spaces/AdelShousha/Sahla-sign2speech)

This allows you to experience the functionality without setting up the environment locally.

