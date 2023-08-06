import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import imutils
import argparse
import copy
import pygame
import os

DEFAULT_IMAGE_WIDTH = 1200

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose

"""
Usage:

python 03_pose_predictions.py --model-name model1 --input-video input.avi

python 03_pose_predictions.py --input-video input.avi
"""

# Initialize the mixer module
pygame.mixer.init()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("--model-name", type=str, required=False, default='model1',
                    help="name of the saved pickled model [no suffix]")
    ap.add_argument("--suppress-landmarks", action='store_true',
                    help="[Optional: False] if present do not show landmarks on yourself ")
    ap.add_argument("--image-width", type=int, required=False, default=1200,
                    help="Image width")
    ap.add_argument("--add-counters", action='store_true',
                    help="[Optional: False] if present should the pose counts ")
    ap.add_argument("--mode", type=str, required=False, default='default',
                    help="Mode of operation. Choose 'default' or 'challenge'.")
    ap.add_argument("--input-video", type=str, required=True,
                    help="Input video file name. The video should be located in 'vid/raw' directory.")

    args = vars(ap.parse_args())

    # Load the audio file
    pygame.mixer.music.load(f"vid/raw/{args['input_video'].replace('.avi', '.mp3')}")

    DEFAULT_IMAGE_WIDTH = args['image_width']
    model_name = args['model_name']
    suppress_landmarks = args['suppress_landmarks']
    add_counters = args['add_counters']

    # Create a directory to store the frames
    frames_dir = f"vid/frames/{args['input_video'].replace('.avi', '')}"
    os.makedirs(frames_dir, exist_ok=True)

    # Load key poses and models if in 'challenge' mode
    if args['mode'] == 'challenge':
        # Load key poses
        key_poses_df = pd.read_excel('data/key_poses.xlsx')

        # Load all models
        models = {}
        for i in range(1, 11):
            with open(f'model/model{i}.pkl', 'rb') as f:
                models[f'model{i}'] = joblib.load(f)

    with open(f'model/{model_name}.pkl', 'rb') as f:
        model = joblib.load(f)

    # Open the video file
    cap = cv2.VideoCapture(f'vid/raw/{args["input_video"]}')

    # Get the video's width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the VideoWriter (replace 'output.mp4' with your desired output file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'XVID'
    out = cv2.VideoWriter(f'vid/annotated/{args["input_video"]}', fourcc,30.0, (width, height))

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5) as pose:
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = imutils.resize(frame, width=DEFAULT_IMAGE_WIDTH)

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Pose Detections
            if not suppress_landmarks:
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

                if args['mode'] == 'challenge':
                    # Convert frame number to integer

                    current_class_name = '---'

                    # Get the model for the current frame
                    filtered_df = key_poses_df[(key_poses_df['start_frame'] <= frame_num) & (key_poses_df['end_frame'] >= frame_num)]
                    if not filtered_df.empty:
                        current_model_row = filtered_df.iloc[0]
                        model_name = current_model_row['model_name']
                        current_class_name  = current_model_row['class_name']

                        # Select the model
                        model = models[model_name]
                            
                        # Make Detections
                        X = pd.DataFrame([row])
                        body_language_class = model.predict(X)[0]
                        body_language_prob = model.predict_proba(X)[0]
                        print(body_language_class, np.around(body_language_prob, decimals=3))

                        # Calculate the width of the text box
                        text_width, _ = cv2.getTextSize(current_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        

                        class1 = body_language_class.split(' ')[0]
                        prob1 = str(round(body_language_prob[np.argmax(body_language_prob)], 2))

                        if class1 != '-' and class1!='Miss':
                            if float(prob1) >= 0.8:
                                # Get status box
                                status_width = 250
                                cv2.rectangle(image, (0, 0), (1250, 60), (134, 240, 125), -1)                                
                            else:
                                # Get status box
                                status_width = 250
                                cv2.rectangle(image, (0, 0), (1250, 60), (130,245, 231), -1)  
                            cv2.putText(image, f'Prob:{prob1} Pose:{class1}', (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        elif class1 == 'Miss':
                                # Get status box
                                status_width = 250
                                cv2.rectangle(image, (0, 0), (1250, 60), (130, 137, 245), -1)                                                             
                                cv2.putText(image, f'Pose:{current_class_name}', (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


                    else:
                        print(f"No model available for frame {frame_num}")
                        class1 = '-'
                        prob1 = '-'
                
                if args['mode'] == 'default':
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    print(body_language_class, np.around(body_language_prob, decimals=3))

                    class1 = body_language_class.split(' ')[0]
                    prob1 = str(round(body_language_prob[np.argmax(body_language_prob)], 2))
               

                

                frame_num += 1

            except Exception as exc:
                print(f"{exc}")
 
            # Save the frame as an image file
            cv2.imwrite(f"{frames_dir}/frame2_{frame_num}.png", image)

            # Write the frame to the output file
            out.write(frame)
            cv2.imshow('Pose Prediction', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release the video file and the output file
    cap.release()
    out.release()
    cv2.destroyAllWindows()





# # Combining frames into a video as the code above is not working.

#     import cv2
#     import os
#     import glob
#     import re

#     # Specify the directory containing your images
#     img_dir = 'vid/frames/output3a'
#     output_file = 'vid/annotated/output3a.avi'

#     # Define a function for natural sort order
#     def atoi(text):
#         return int(text) if text.isdigit() else text

#     def natural_keys(text):
#         return [atoi(c) for c in re.split(r'(\d+)', text)]

#     # Get the list of image file names
#     img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

#     # Sort the files in natural order
#     img_files.sort(key=natural_keys)

#     # Read the first image to get the shape
#     img = cv2.imread(img_files[0])
#     height, width, _ = img.shape

#     # Define the codec and create a VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))

#     for img_file in img_files:
#         # Read each image
#         img = cv2.imread(img_file)

#         # Write the image to the output file
#         out.write(img)

#     # Release the output file
#     out.release()


