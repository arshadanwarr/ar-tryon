# AR Try-On

A web app that places virtual accessories (glasses, hat, mustache) on a face in a photo, using facial landmarks detected by MediaPipe.

**Live app:** [your Streamlit link]

![Demo](screenshots/demo.png)

## Features
- Take a photo with the camera or upload one
- Choose glasses, hat and mustache, and adjust the size of each
- Option to add space above the head so hats are not cut off
- Side-by-side original and result, with PNG download

## How it works
1. MediaPipe Face Landmarker detects facial landmarks in the photo.
2. Each accessory is anchored to its own landmarks:
   - Glasses: outer eye corners (width from their distance)
   - Hat: top of the forehead (size from face width)
   - Mustache: between the nose and upper lip (size from mouth width)
3. Each accessory is resized, rotated by the head roll (angle of the eye line) and blended onto the photo using its transparency channel.

## Tech stack
Python, Streamlit, MediaPipe, OpenCV, NumPy, Pillow

## Run locally
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
app.py            Streamlit interface
tryon.py          Landmark detection and accessory placement
assets/           Transparent PNGs (glasses, hat, mustache)
models/           MediaPipe face_landmarker.task
requirements.txt  Python dependencies
packages.txt      System packages for Streamlit Cloud
```

## Limitations
- The overlay is 2D, so accessories do not follow large head turns (yaw)
- One face per photo
- Works on photos (camera snapshot or upload), not continuous live video

## Possible improvements
- Use 3D head pose so accessories rotate with the head
- Live video with streamlit-webrtc
- Multiple faces and more accessories
