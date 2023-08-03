import mediapipe as mp
import cv2
import argparse
import csv
import numpy as np
from pathlib import Path
import os
import random

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose

_cap = None
_current_video_file = None


def _release_video():
    _cap.release()


def _initialize_video_capture(video_file):
    global _cap, _current_video_file
    _cap = cv2.VideoCapture(video_file)
    _current_video_file = video_file


def _read_frame():
    ret, image = _cap.read()

    return ret, image

# Eg. python 01_training_data.py --class-name Hit --video-folder vid/raw --start-frame 38 --end-frame 56 --file-name pose_training_12pt.csv --select-random

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", required=True, help="target name for captured data")
    ap.add_argument("--video-folder", required=True, help="folder containing videos to process")
    ap.add_argument("--start-frame", type=int, required=True, help="frame to start collecting data")
    ap.add_argument("--end-frame", type=int, required=True, help="frame to end collecting data")
    ap.add_argument("--file-name", type=str, required=False, default='pose_training.csv',
                    help="name of the training data file")
    ap.add_argument("--dry-run", action='store_true', help="[Optional: False] is set then do NOT store data")
    ap.add_argument("--select-random", action='store_true', help="[Optional: False] if set, select random frames")

    args = vars(ap.parse_args())

    class_name = args['class_name']
    video_folder = args['video_folder']
    start_frame = args['start_frame']
    end_frame = args['end_frame']
    file_name = args['file_name']
    dry_run = args['dry_run']
    select_random = args['select_random']


    n = end_frame - start_frame + 1

    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

    vid_num = 0

    for video_file in video_files:
        _initialize_video_capture(os.path.join(video_folder, video_file))

        vid_num +=1

        num_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT  ))
        if not _cap.isOpened(): 
            print(f"Failed to open {_current_video_file}")
            continue


        # Initiate holistic model
        frame_count = 0
        print(f"Processing {video_file}")

        num_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_select = random.sample(list(set(range(1, num_frames + 1)) - set(range(start_frame, end_frame + 1))), min(n, len(list(set(range(1, num_frames + 1)) - set(range(start_frame, end_frame + 1)))))) if select_random else []



        random.shuffle(frames_to_select)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ret, image = _read_frame()

                if not ret:
                    break  # End of video

                frame_count += 1

                if start_frame <= frame_count <= end_frame or (select_random and frame_count in frames_to_select):
                    if select_random and frame_count in frames_to_select and vid_num % 2 != 0:
                        label = 'Miss'
                    elif start_frame <= frame_count <= end_frame: 
                        label = class_name

                    # Recolor Feed
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    # Make Detections
                    results = pose.process(image)

                    # Recolor image back to BGR for rendering
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # 4. Pose Detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                                )

                    # Export coordinates
                    try:
                        # Extract Pose landmarks
                        landmarks = results.pose_landmarks.landmark
                        body_landmarks = []
                        pose_index = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.LEFT_ELBOW.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.LEFT_WRIST.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.RIGHT_WRIST.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.LEFT_HIP.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.RIGHT_HIP.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.LEFT_KNEE.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.RIGHT_KNEE.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.LEFT_ANKLE.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        pose_index = mp_pose.PoseLandmark.RIGHT_ANKLE.value
                        body_landmarks += [landmarks[pose_index].x, landmarks[pose_index].y]

                        row = np.around(body_landmarks, decimals=9).tolist()

                        # Append class name
                        row.insert(0, label)

                        # Export to CSV
                        if not dry_run:
                            Path(f'data').mkdir(parents=True, exist_ok=True)
                            with open(f'data/{file_name}', mode='a', newline='') as f:
                                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(row)

                    except:
                        pass

                cv2.imshow('Pose Detection', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        _release_video()

    cv2.destroyAllWindows()
