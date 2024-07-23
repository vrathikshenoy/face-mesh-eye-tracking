# Face Mesh and Eye Tracking

This project uses computer vision techniques to perform real-time face mesh detection, eye tracking, and head pose estimation using a webcam.

## Features

- Face mesh detection and rendering
- Eye tracking with iris angle calculation
- Head pose estimation
- Real-time processing with FPS display

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/vrathikshenoy/face-mesh-eye-tracking.git
   cd face-mesh-eye-tracking
   ```

2. Install the required packages:
   ```
   pip install opencv-python mediapipe numpy
   ```

## Usage

Run the script with:

```
python face_mesh_eye_tracking.py
```

- Press 'Esc' to exit the application.

## How it works

1. **Face Mesh Detection**: The script uses MediaPipe's Face Mesh solution to detect and render a 3D face mesh in real-time.

2. **Eye Tracking**: 
   - Detects the position of the irises
   - Calculates the angle of each eye relative to the eye corners
   - Visualizes eye movement with vectors

3. **Head Pose Estimation**:
   - Uses selected facial landmarks to estimate the 3D pose of the head
   - Determines the direction the person is looking (Left, Right, Up, Down, or Forward)
   - Visualizes head direction with a line projecting from the nose

4. **Performance**: 
   - Displays the real-time frames per second (FPS) to monitor performance

## Customization

You can adjust the following parameters in the script:

- `max_num_faces`: Maximum number of faces to detect
- `min_detection_confidence`: Minimum confidence value for face detection
- `min_tracking_confidence`: Minimum confidence value for landmark tracking

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/vrathikshenoy/face-mesh-eye-tracking/issues) if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the Face Mesh solution
- [OpenCV](https://opencv.org/) for computer vision utilities

## Referance 

- https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py

