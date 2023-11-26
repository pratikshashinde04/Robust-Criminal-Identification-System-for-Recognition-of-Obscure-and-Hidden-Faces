
import os
import cv2

# Load the cascade classifiers for face, eyes, nose, and mouth
face_cascade = cv2.CascadeClassifier("C:/Users/kamle/OneDrive/Documents/EDI/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/kamle/OneDrive/Documents/EDI/haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("C:/Users/kamle/OneDrive/Documents/EDI/haarcascade_frontalface_default.xml")
mouth_cascade = cv2.CascadeClassifier("C:/Users/kamle/OneDrive/Documents/EDI/haarcascade_frontalface_default.xml")

def process_and_save_image(image, criminal_id, count):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_filename = f'User_{criminal_id}_Face_{count}.jpg'
        face_path = os.path.join('C:/Users/kamle/OneDrive/Documents/EDI/datasets', face_filename)
        cv2.imwrite(face_path, gray[y:y + h, x:x + w])

        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f"Frame Count: {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Detect eyes and save them
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            eye_filename = f'User_{criminal_id}_Eye_{count}_{i}.jpg'
            eye_path = os.path.join('C:/Users/kamle/OneDrive/Documents/EDI/datasets', eye_filename)
            cv2.imwrite(eye_path, roi_gray[ey:ey + eh, ex:ex + ew])
            # Draw a rectangle around the eyes and display frame count
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(roi_color, f"Frame Count: {count}", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detect nose and save it
        noses = nose_cascade.detectMultiScale(roi_gray)
        for i, (nx, ny, nw, nh) in enumerate(noses):
            nose_filename = f'User_{criminal_id}_Nose_{count}_{i}.jpg'
            nose_path = os.path.join('C:/Users/kamle/OneDrive/Documents/EDI/datasets', nose_filename)
            cv2.imwrite(nose_path, roi_gray[ny:ny + nh, nx:nx + nw])
            # Draw a rectangle around the nose and display frame count
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
            cv2.putText(roi_color, f"Frame Count: {count}", (nx, ny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Detect mouth and save it
        mouths = mouth_cascade.detectMultiScale(roi_gray)
        for i, (mx, my, mw, mh) in enumerate(mouths):
            mouth_filename = f'User_{criminal_id}_Mouth_{count}_{i}.jpg'
            mouth_path = os.path.join('C:/Users/kamle/OneDrive/Documents/EDI/datasets', mouth_filename)
            cv2.imwrite(mouth_path, roi_gray[my:my + mh, mx:mx + mw])
            # Draw a rectangle around the mouth and display frame count
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
            cv2.putText(roi_color, f"Frame Count: {count}", (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Capture from Webcam", image)
    cv2.waitKey(1)

    return count

def main():
    criminal_id = input("Enter Criminal's ID: ")

    print("Select an option:")
    print("1. Capture image from webcam")
    print("2. Upload an existing image")
    print("3. Use an existing video")
    choice = input("Enter choice: ")

    count = 0
    capture_limit = 300  # Set the limit to 300 frames

    if choice == "1":
        video = cv2.VideoCapture(0)  # Open the
        while count < capture_limit:
            ret, frame = video.read()
            if not ret:
                break
            count = process_and_save_image(frame, criminal_id, count)
        video.release()
    elif choice == "2":
        image_path = input("Enter the path of the existing image: ")
        image = cv2.imread(image_path)
        if image is not None:
            count = process_and_save_image(image, criminal_id, count)
        else:
            print("Error: Unable to read the image.")
    elif choice == "3":
        video_path = input("Enter the path of the existing video: ")
        video = cv2.VideoCapture(video_path)
        while count < capture_limit:
            ret, frame = video.read()
            if not ret:
                break
            count = process_and_save_image(frame, criminal_id, count)
        video.release()
    else:
        print("Invalid choice.")

    print("Dataset Collection Done for Criminal", criminal_id)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
