# Sign Detection

Real-time sign language detection using MediaPipe pose/hand tracking and a two-stream TensorFlow/Keras model. Recognizes gestures for: **hello**, **iloveyou**, and **thankyou**.

## Features

- Real-time webcam detection
- Two-stream architecture (global body + local hands)
- MediaPipe-based pose and hand landmark detection
- Temporal smoothing for stable predictions

## Requirements

- Python 3.8+
- Webcam
- Model file: `models/two_stream_best.h5` (see [models/README.md](models/README.md))

## Installation

```bash
# Clone the repository
git clone https://github.com/Madhusudhan1706/Sign_detection.git
cd Sign_detection

# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Obtain the model file** (555MB) - see [models/README.md](models/README.md) for instructions
2. Place `two_stream_best.h5` in the `models/` directory
3. Ensure your webcam is connected

## Usage

```bash
python sign_detection.py
```

Press `q` to quit the application.

## Project Structure

```
Sign_detection/
├── sign_detection.py      # Main detection script
├── classes.npy            # Class labels
├── models/
│   ├── README.md          # Model download instructions
│   └── two_stream_best.h5 # Trained model (not in repo - too large)
└── dataset/               # Training/validation/test videos
    ├── train/
    ├── val/
    └── test/
```

## Notes

- Model files (`.h5`) are excluded from git due to size constraints
- Dataset videos are included for reference/retraining
- The script displays debug windows showing the cropped regions fed to the model

## License

See repository for license information.
