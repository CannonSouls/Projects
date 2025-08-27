import cv2  # Import the OpenCV library for computer vision tasks

# Load pre-trained Haar cascades for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load face detector
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')  # Load smile detector

# Start webcam
cap = cv2.VideoCapture(0)  # Initialize video capture from the default webcam

while True:  # Continuously capture frames from the webcam
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:  # If frame not read correctly, exit loop
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces in the grayscale frame

    for (x, y, w, h) in faces:  # Iterate over all detected faces
        roi_gray = gray[y:y+h, x:x+w]  # Region of interest in grayscale (face area)
        roi_color = frame[y:y+h, x:x+w]  # Region of interest in color (face area)
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)  # Detect smiles in the face region

        if len(smiles) > 0:  # If at least one smile is detected
            cv2.putText(frame, 'Smile!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)  # Display 'Smile!' above the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)  # Draw a rectangle around the face

    cv2.imshow('Smile Detection', frame)  # Show the frame with detections
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' key is pressed
        break

cap.release()  # Release the webcam resource
cv2.destroyAllWindows()  # Close all OpenCV windows