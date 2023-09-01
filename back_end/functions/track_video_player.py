import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import torch


def track_video_player(path, Model):
    Model += '.yaml'

    # Set color for each id, pre-generate 100 colors
    num_colors = 100
    colors = []
    for i in range(num_colors):
        red = np.random.randint(0, 256)
        green = np.random.randint(0, 256)
        blue = np.random.randint(0, 256)
        colors.append((red, green, blue))

    track_colors = {}
    track_ids_counter = 0

    # Load the model
    model = YOLO('./runs/detect/best-model/weights/best.onnx')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Open the video file
    video_path = path
    cap = cv2.VideoCapture(video_path)

    # Get video frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    scaling_factor = frame_height / 1080

    # Create a VideoWriter object to save the new video
    output_path = "./results/videos/results.mp4"  # 保存视频的路径
    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Format text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1 * scaling_factor
    font_thickness = int(3 * scaling_factor)
    text_color = (0, 0, 255)
    text_position = (0, int(40 * scaling_factor))

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Traverse the video frame by frame and make predictions
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, vid_stride=True, device=device,
                                  tracker=str(Model))  # tracker is used to specify the tracker

            # Calculate scaling factor based on frame dimensions
            if results is not None and len(results) > 0 and results[0].boxes is not None and results[
                0].boxes.id is not None:

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # record.append(len(track_ids))
                # Visualize the results on the frame
                # annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y), time.time()))  # x, y center point
                    # if len(track) > 30:  # retain 90 tracks for 90 frames
                    #   track.pop(0)

                    # Adjust the thickness of the tracks using the scaling factor
                    track_thickness = int(10 * scaling_factor)

                    if track_id not in track_colors:
                        track_colors[track_id] = colors[track_ids_counter % len(colors)]
                        track_ids_counter += 1
                    # Draw the tracking lines
                    color = track_colors[track_id]
                    # Prepare points for drawing the track
                    points = []
                    for point in track:
                        x, y, timestamp = point
                        points.append((int(x), int(y)))
                    points = np.array(points).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=track_thickness)

                    # Calculate speed and acceleration of the tracked object
                    if len(track) > 2:
                        pprev_x, pprev_y, pprev_time = track[-3]
                        prev_x, prev_y, prev_time = track[-2]
                        now_x, now_y, now_time = track[-1]

                        prev_time_interval = prev_time - pprev_time
                        now_time_interval = now_time - prev_time

                        prev_distance = np.sqrt((pprev_x - prev_x) ** 2 + (pprev_y - prev_y) ** 2)
                        curr_distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

                        prev_speed = prev_distance / prev_time_interval
                        curr_speed = curr_distance / now_time_interval

                        acceleration = (curr_speed - prev_speed) / now_time_interval

                        speed_text = f"Speed: {curr_speed:.2f}"
                        acceleration_text = f"Acc: {acceleration:.2f}"
                        # Displays current velocity and acceleration on each target's unknown
                        cv2.putText(frame, speed_text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale,
                                    (0, 0, 255), font_thickness, cv2.LINE_AA)
                        cv2.putText(frame, acceleration_text, (int(x), int(y) + 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale,
                                    (0, 0, 255), font_thickness, cv2.LINE_AA)

                # Write the annotated frame to the output video
                out.write(frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                out.write(frame)
                continue
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
