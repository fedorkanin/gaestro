import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


def create_gesture_recognizer():
    base_options = python.BaseOptions(
        model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO)
    return vision.GestureRecognizer.create_from_options(options)


def initialize_webcam():
    return cv2.VideoCapture(0)


def process_frame(frame, recognizer, mp_hands, mp_drawing, mp_drawing_styles, frame_count, fps):
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_obj = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    timestamp = int(frame_count * 1000 / fps)

    recognition_result = recognizer.recognize_for_video(image_obj, timestamp)

    annotated_image = image_rgb.copy()

    if recognition_result.gestures:
        annotate_gesture(recognition_result, annotated_image,
                         mp_hands, mp_drawing, mp_drawing_styles, frame)

    return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


def annotate_gesture(recognition_result, annotated_image, mp_hands, mp_drawing, mp_drawing_styles, frame):
    top_gesture = recognition_result.gestures[0][0]
    gesture_name = top_gesture.category_name
    gesture_score = top_gesture.score

    for hand_landmarks in recognition_result.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        cv2.putText(annotated_image, f"{gesture_name} ({gesture_score:.2f})",
                    (int(hand_landmarks[0].x * frame.shape[1]),
                     int(hand_landmarks[0].y * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)


def main():
    recognizer = create_gesture_recognizer()
    cap = initialize_webcam()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        annotated_image_bgr = process_frame(
            frame, recognizer, mp_hands, mp_drawing, mp_drawing_styles, frame_count, fps)

        frame_count += 1

        cv2.imshow('Gesture Recognition', annotated_image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
