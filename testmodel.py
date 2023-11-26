import cv2

# Initialize the video capture object
video = cv2.VideoCapture(0)  # 0 means the default webcam, you can change it to a video file path if needed

# Load the trained model
trained_model = cv2.face.LBPHFaceRecognizer_create()
trained_model.read("Trainer.yml")

# Load the face cascade
face_cascade = cv2.CascadeClassifier("C:/Users/kamle/OneDrive/Documents/EDI/haarcascade_frontalface_default.xml")

# Adjusted name list to start from index 0, where label 0 corresponds to "Unknown"
name_list = ["", "Varad", "Anish", "Kamlesh", ""]

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Perform face recognition
        label, confidence = trained_model.predict(face_roi)

        # Determine the name based on the label and confidence threshold
        if confidence >= 50.0:
            name = name_list[label]
        else:
            name = "Unknown"

        # Print the detected name and confidence level
        print(f"{name}={confidence:.2f}")

        # Display the recognized name and confidence level on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {name}", (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the recognized faces
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and destroy OpenCV windows
video.release()
cv2.destroyAllWindows()
