
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# -------------------------------
# 1. Load trained mask detection model
# -------------------------------
try:
    model = load_model("mask_detector.h5", compile=False)
    print("Mask detection model loaded successfully.")

except Exception as e:
    print("ERROR: Could not load mask_detector.h5")
    print(e)
    raise SystemExit(1)


# -------------------------------
# 2. Load face detector
# -------------------------------
face_cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(face_cascade_path)

if face_detector.empty():
    print("ERROR: Could not load the face detection classifier.")
    raise SystemExit(1)


# -------------------------------
# 3. Open camera
# -------------------------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    camera.release()
    print("ERROR: Camera could not be opened.")
    print("Check Windows camera permissions and close other camera apps.")
    raise SystemExit(1)

# Request a reasonable camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened successfully.")
print("Press Q to quit.")


# -------------------------------
# 4. Face detection and prediction
# -------------------------------
frame_errors = 0

try:
    while True:
        success, frame = camera.read()

        if not success or frame is None:
            frame_errors += 1
            print("WARNING: Unable to read camera frame.")

            if frame_errors >= 10:
                print("ERROR: Camera is not providing frames.")
                break

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

            continue

        frame_errors = 0

        # Mirror the image for a natural webcam preview
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:

            # Extract face region
            face = frame[y:y + h, x:x + w]

            if face.size == 0:
                continue

            # OpenCV uses BGR; Keras image datasets commonly use RGB.
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Resize to model input size
            face = cv2.resize(face, (224, 224))

            # Normalize pixel values
            face = face.astype("float32") / 255.0

            # Add batch dimension: (1, 224, 224, 3)
            face = np.expand_dims(face, axis=0)

            # Predict
            prediction = model.predict(face, verbose=0)[0]

            if len(prediction) < 2:
                print("ERROR: Model must return two class scores.")
                break

            mask_score = float(prediction[0])
            no_mask_score = float(prediction[1])

            if mask_score >= no_mask_score:
                label = "Mask"
                confidence = mask_score * 100
                color = (0, 255, 0)
            else:
                label = "No Mask"
                confidence = no_mask_score * 100
                color = (0, 0, 255)

            text = f"{label}: {confidence:.1f}%"

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Keep label inside the frame
            text_y = max(y - 10, 25)

            cv2.putText(
                frame,
                text,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

        # Display status and camera image
        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow("Face Mask Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nDetection stopped by user.")

except Exception as e:
    print("ERROR during detection:", e)

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera released. Program closed.")
