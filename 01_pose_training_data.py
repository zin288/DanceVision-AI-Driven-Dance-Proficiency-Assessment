import mediapipe as mp
import cv2
import argparse
import csv
import numpy as np
from pathlib import Path
import time
from uielements.uielements import DisplayValueLabel

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose

_cap = None


def _release_video():
    _cap.release()


def _initialize_video_capture():
    global _cap
    _cap = cv2.VideoCapture(0)


def _read_frame():
    ret, image = _cap.read()

    return ret, image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", required=True, help="target name for captured data")
    ap.add_argument("--collect-for", type=int, required=False, default=30, help="number of seconds to collect data")
    ap.add_argument("--start-delay", type=int, required=False, default=5,
                    help="number of seconds to wait before collecting data")
    ap.add_argument("--file-name", type=str, required=False, default='pose_training.csv',
                    help="name of the training data file")
    ap.add_argument("--dry-run", action='store_true', help="[Optional: False] is set then do NOT store data")

    args = vars(ap.parse_args())

    class_name = args['class_name']
    collect_for = args['collect_for']
    start_delay = args['start_delay']
    file_name = args['file_name']
    dry_run = args['dry_run']

    _initialize_video_capture()

    # Initiate holistic model
    start_delay_time = time.time() + start_delay
    collect_for_time = start_delay_time + collect_for

    display_label = DisplayValueLabel(3,3, 200, 40, "Start in")
    print(f"***** Start capture in {start_delay} seconds ")

    # save an image  of the pose to so we can overlay points
    saved_image = False
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, image = _read_frame()

            # Wait 'start_delay_time' before starting to collect data points to give the person
            # time to get into position
            if time.time() > start_delay_time:
                # have we saved an initial image yet?
                if not saved_image:
                    if not dry_run:
                        print(f"Image Shape: {image.shape}")
                        cv2.imwrite(f'data/{class_name}.png', image)
                        saved_image = True
                    display_label.label = 'Seconds'

                # Recolor Feed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detections
                results = pose.process(image)

                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # display remaining collection time
                display_label.set_value(int(collect_for_time-time.time()))
                display_label.draw(image)


                # 4. Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    # We are only interested in the arm landmarks
                    landmarks = results.pose_landmarks.landmark
                    arm_landmarks = []
                    pose_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                    arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y ]

                    pose_index = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                    arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                    pose_index = mp_pose.PoseLandmark.LEFT_ELBOW.value
                    arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                    pose_index = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                    arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                    pose_index = mp_pose.PoseLandmark.LEFT_WRIST.value
                    arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y ]

                    pose_index = mp_pose.PoseLandmark.RIGHT_WRIST.value
                    arm_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                    row = np.around(arm_landmarks, decimals=9).tolist()

                    # Append class name
                    row.insert(0, class_name)

                    # Export to CSV
                    if not dry_run:
                        Path(f'data').mkdir(parents=True, exist_ok=True)
                        with open(f'data/{file_name}', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row)

                except:
                    pass

            else:
                display_label.set_value(int(start_delay_time-time.time()))
                display_label.draw(image)

            cv2.imshow('Pose Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            if time.time() > collect_for_time:
                break

    _release_video()

    cv2.destroyAllWindows()
