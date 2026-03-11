# Sign Detection

This repository contains a real-time sign language detection demo using MediaPipe and a two-stream TensorFlow model.

Quick start

```bash
# create virtualenv and activate (optional)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run
python sign_detection.py
```

Notes
- Model files and large binaries are excluded from the repository by `.gitignore`.
- To create a GitHub remote automatically, the maintainer should be authenticated with the `gh` CLI.
