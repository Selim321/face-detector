# Face Detection Web App

This is a web application for real-time face detection using either Haar Cascade or Deep Neural Network (DNN) methods. It allows users to upload images or use their webcam to detect faces.

## Features

* **Image Upload:** Upload an image to detect faces.
* **Live Video:** Use your webcam for real-time face detection.
* **Detection Methods:** Choose between Haar Cascade (fast) and DNN (more accurate) methods.
* **Confidence Threshold:** Adjust the confidence level for DNN detection.
* **Camera Testing:** A built-in camera test to verify camera functionality.
* **WebRTC and Simple Fallback:** WebRTC for better performance, with a fallback to OpenCV for compatibility.
* **Troubleshooting Tips:** Detailed instructions for resolving webcam issues.

## How to Use

1.  **Image Upload Mode:**
    * Select "Image Upload" from the sidebar.
    * Upload an image using the file uploader.
    * Choose the detection method (Haar Cascade or DNN).
    * Adjust the confidence threshold (for DNN).
    * Click "Detect Faces" to see the results.

2.  **Live Video Mode:**
    * Select "Live Video" from the sidebar.
    * Click "Test Camera" to verify your camera is working.
    * If the camera test works, the webcam feature should also work.
    * If WebRTC does not work, the fallback method is used.
    * Choose the detection method (Haar Cascade or DNN).
    * Adjust the confidence threshold (for DNN).
    * The live video feed will display the detected faces.
    * To stop the webcam when using the fallback method, click the "Stop Camera" button.

## Access the Web App

The web application is deployed on Streamlit Cloud and can be accessed at:

[https://face-detector-jltdxezjcqcgpm7pv5q7og.streamlit.app/](https://face-detector-jltdxezjcqcgpm7pv5q7og.streamlit.app/)

## Installation (for local development)

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd face-detector
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download DNN model files:**

    * Download `opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt`.
    * Place them in the same directory as `app.py`.

5.  **Run the application:**

    ```bash
    streamlit run app.py
    ```

## Dependencies

* streamlit
* streamlit-webrtc
* opencv-python
* numpy
* pillow
* av

## Troubleshooting

* **Webcam Issues:**
    * Ensure your browser has camera permissions.
    * Check if another application is using your camera.
    * Try using Chrome or Firefox.
    * Refresh the page.
    * Check your system's camera settings.
* **DNN Model Issues:**
    * Ensure that the DNN model files are in the same directory as the python application.
* **libGL.so.1 error on streamlit cloud:**
    * Ensure that the packages.txt file contains libgl1.

## Author

Selim321