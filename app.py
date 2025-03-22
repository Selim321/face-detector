import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import logging
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    st.title("Face Detection App")
    
    # Sidebar for configurations
    st.sidebar.title("Settings")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Image Upload", "Live Video"])
    
    # Detection method selection
    detection_method = st.sidebar.radio(
        "Choose Detection Method",
        ("Haar Cascade (Fast)", "DNN (More Accurate)")
    )
    
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)
    
    if app_mode == "Image Upload":
        image_mode(detection_method, confidence_threshold)
    else:
        video_mode(detection_method, confidence_threshold)


def image_mode(detection_method, confidence_threshold):
    st.subheader("Image Mode")
    st.write("Upload an image to detect faces")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display original image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Detect faces based on chosen method
        if st.button("Detect Faces"):
            if detection_method == "Haar Cascade (Fast)":
                with st.spinner("Detecting faces using Haar Cascade..."):
                    result_img, face_count = detect_faces_haar(img_array)
            else:
                with st.spinner("Detecting faces using DNN (this may take a moment)..."):
                    result_img, face_count = detect_faces_dnn(img_array, confidence_threshold)
            
            # Display result
            st.subheader("Detection Result")
            st.image(result_img, caption=f"Detected {face_count} faces", use_column_width=True)
            
            if face_count == 0:
                st.info("No faces detected. Try adjusting the confidence threshold or use a different image.")


def video_mode(detection_method, confidence_threshold):
    st.subheader("Live Video Mode")
    st.write("Using webcam for real-time face detection")
    
    # Create a simple video capture test button
    if st.button("Test Camera"):
        test_camera()
    
    # Debug info expander
    with st.expander("Troubleshooting Info"):
        st.write("""
        If you're having issues with the webcam:
        
        1. **Browser Permissions**: Ensure your browser has permission to access the camera
        2. **Multiple Applications**: Check if another application is using your camera
        3. **Browser Compatibility**: Try using Chrome or Firefox for best compatibility
        4. **Refresh the Page**: Sometimes refreshing the page resolves connection issues
        5. **System Camera Settings**: Check if your camera is enabled in system settings
        """)
        
        # Display available cameras (OpenCV)
        st.subheader("Available Camera Devices")
        camera_indexes = []
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_indexes.append(i)
                ret, frame = cap.read()
                if ret:
                    st.write(f"Camera index {i}: Available")
                    # Display small preview if possible
                    small_frame = cv2.resize(frame, (320, 240))
                    st.image(small_frame, channels="BGR", caption=f"Camera {i} Preview")
                else:
                    st.write(f"Camera index {i}: Connected but no frame")
                cap.release()
        
        if not camera_indexes:
            st.error("No cameras detected by OpenCV")
    
    class VideoProcessor(VideoProcessorBase):
        def __init__(self, detection_method, confidence_threshold):
            self.detection_method = detection_method
            self.confidence_threshold = confidence_threshold
            self.face_count = 0
            self.frame_counter = 0
            self.last_log_time = time.time()
            
            # Load Haar cascade if needed
            if self.detection_method == "Haar Cascade (Fast)":
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                logger.info("Haar cascade loaded successfully")
            
            # Load DNN model if needed
            if self.detection_method == "DNN (More Accurate)":
                try:
                    model_file = "opencv_face_detector_uint8.pb"
                    config_file = "opencv_face_detector.pbtxt"
                    self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                    logger.info("DNN model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading DNN model: {e}")
                    st.warning("DNN model files not found. Using Haar Cascade instead.")
                    self.detection_method = "Haar Cascade (Fast)"
                    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def recv(self, frame):
            try:
                self.frame_counter += 1
                img = frame.to_ndarray(format="bgr24")
                
                # Log frame info occasionally
                current_time = time.time()
                if current_time - self.last_log_time > 5:  # Log every 5 seconds
                    logger.info(f"Processing frame {self.frame_counter}, shape: {img.shape}")
                    self.last_log_time = current_time
                
                # Process based on detection method
                if self.detection_method == "Haar Cascade (Fast)":
                    processed_img, self.face_count = self._process_haar(img)
                else:
                    processed_img, self.face_count = self._process_dnn(img, self.confidence_threshold)
                
                # Add face count text to image
                cv2.putText(processed_img, f"Faces: {self.face_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add frame counter (for debugging)
                cv2.putText(processed_img, f"Frame: {self.frame_counter}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            except Exception as e:
                logger.error(f"Error in frame processing: {e}")
                # Return original frame if there's an error
                return frame
        
        def _process_haar(self, img):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Draw rectangles around faces
            img_with_faces = img.copy()
            face_count = len(faces)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            return img_with_faces, face_count
        
        def _process_dnn(self, img, conf_threshold):
            frameHeight = img.shape[0]
            frameWidth = img.shape[1]
            
            # Create a blob and pass it through the model
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Create a copy of the image to draw on
            img_with_faces = img.copy()
            face_count = 0
            
            # Process detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > conf_threshold:
                    face_count += 1
                    
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    
                    # Draw rectangle
                    cv2.rectangle(img_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence label
                    label = f"{confidence:.2f}"
                    cv2.putText(img_with_faces, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return img_with_faces, face_count
    
    # WebRTC configuration - using updated parameters 
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    
    # Status placeholder
    status_placeholder = st.empty()
    status_placeholder.info("Starting webcam...")
    
    # Add a radio button for selecting the webcam method
    webrtc_method = st.radio(
        "Choose Webcam Method:",
        ("WebRTC (Recommended)", "Simple Fallback")
    )
    
    if webrtc_method == "WebRTC (Recommended)":
        # Create WebRTC streamer with updated parameters
        try:
            ctx = webrtc_streamer(
                key="face-detection",
                mode=WebRtcMode.SENDRECV,
                frontend_rtc_configuration={"iceServers": ice_servers},
                server_rtc_configuration={"iceServers": ice_servers},
                video_processor_factory=lambda: VideoProcessor(detection_method, confidence_threshold),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            if ctx.state.playing:
                status_placeholder.success("Webcam started successfully!")
            else:
                status_placeholder.warning("Waiting for webcam to start...")
                
            # Display info text
            st.info("""
            Start the webcam using the button above. 
            
            Troubleshooting tips:
            1. Check if the browser is requesting camera permissions
            2. Try refreshing the page if the video doesn't appear
            3. Click the "Test Camera" button to check if your camera is working
            4. Try the "Simple Fallback" method if WebRTC is not working
            """)
            
        except Exception as e:
            st.error(f"Error initializing WebRTC: {e}")
            logger.error(f"WebRTC error: {e}")
            st.warning("Please try the Simple Fallback method instead.")
    else:
        # Simple OpenCV fallback
        simple_webcam(detection_method, confidence_threshold)


def simple_webcam(detection_method, confidence_threshold):
    """Fallback simple webcam implementation using OpenCV"""
    st.write("Using simple webcam method")
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    # Add stop button
    stop = st.button("Stop Camera")
    
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Failed to open camera. Check your camera connection and permissions.")
        return
    
    # Prepare face detection
    if detection_method == "Haar Cascade (Fast)":
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        detector = lambda img: detect_faces_haar(img)
    else:
        try:
            # Try to load DNN model
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            detector = lambda img: detect_faces_dnn(img, confidence_threshold)
        except:
            st.warning("DNN model not found, using Haar Cascade instead")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            detector = lambda img: detect_faces_haar(img)
    
    # Video loop
    try:
        frame_count = 0
        while not stop:
            frame_count += 1
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to get frame from camera")
                break
            
            # Process frame for face detection
            if frame_count % 2 == 0:  # Process every other frame for better performance
                processed_frame, face_count = detector(frame)
                
                # Add face count
                cv2.putText(processed_frame, f"Faces: {face_count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # Short sleep to keep UI responsive
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"Error in webcam processing: {e}")
        logger.error(f"Webcam error: {e}")
    finally:
        # Release resources
        cap.release()


def test_camera():
    """Function to test if camera is working"""
    st.write("Testing camera...")
    
    try:
        # Create a placeholder for the test image
        test_placeholder = st.empty()
        
        # Try to open the camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Failed to open camera. Please check your camera connection and settings.")
            return
        
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            st.error("Could not read frame from camera. Camera might be in use by another application.")
            return
        
        # Display the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_placeholder.image(rgb_frame, caption="Camera Test - If you see this image, your camera is working!", use_column_width=True)
        
        # Release the camera
        cap.release()
        
        st.success("Camera test successful! Your camera is working.")
        
    except Exception as e:
        st.error(f"Error during camera test: {e}")
        logger.error(f"Camera test error: {e}")


def detect_faces_haar(img):
    """Detect faces using Haar Cascade classifier"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    img_with_faces = img.copy()
    face_count = len(faces)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img_with_faces, face_count


def detect_faces_dnn(img, conf_threshold=0.5):
    """Detect faces using DNN model"""
    # Load the DNN face detector model
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    
    # Check if model files exist, if not, inform user
    try:
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    except:
        st.error("""
        DNN model files not found. You need to download them:
        1. opencv_face_detector_uint8.pb
        2. opencv_face_detector.pbtxt
        
        Place them in the same directory as this script.
        
        For now, reverting to Haar Cascade method.
        """)
        return detect_faces_haar(img)
    
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    
    # Create a copy of the image to draw on
    img_with_faces = img.copy()
    face_count = 0
    
    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf_threshold:
            face_count += 1
            
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            
            # Draw rectangle
            cv2.rectangle(img_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"{confidence:.2f}"
            cv2.putText(img_with_faces, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_with_faces, face_count


if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Detection App",
        page_icon="👤",
        layout="wide"
    )
    main()