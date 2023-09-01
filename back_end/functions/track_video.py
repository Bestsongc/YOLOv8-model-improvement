import cv2
import numpy as np
from ultralytics import YOLO
from math import atan2, degrees
from collections import defaultdict
import torch


def track_video(path, Model):
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

    # Store the IDs of reverse-moving objects
    reverse_moving_objects = {}

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

    # Set the correct direction
    direction_angle = -22.5
    direction_vector = np.array([np.cos(np.radians(direction_angle)), np.sin(np.radians(direction_angle))])

    # Format text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 * scaling_factor
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

            if results is not None and len(results) > 0 and results[0].boxes is not None and results[
                0].boxes.id is not None:

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                # annotated_frame = results[0].plot()

                # Clear the reverse moving objects dictionary for each frame
                reverse_moving_objects.clear()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # Adjust the thickness of the tracks using the scaling factor
                    track_thickness = int(10 * scaling_factor)

                    if track_id not in track_colors:
                        track_colors[track_id] = colors[track_ids_counter % len(colors)]
                        track_ids_counter += 1
                    # Draw the tracking lines
                    # color = track_colors[track_id]
                    # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(frame, [points], isClosed=False, color=color, thickness=track_thickness)

                    # Calculate direction angle of the tracked object's movement
                    if len(track) > 1:
                        prev_x, prev_y = track[-2]
                        curr_x, curr_y = track[-1]
                        direction_vector_obj = np.array([curr_x - prev_x, curr_y - prev_y])
                        angle = degrees(atan2(np.cross(direction_vector_obj, direction_vector),
                                              np.dot(direction_vector_obj, direction_vector)))
                        if angle > 180:  # Ensure angle is within [-180, 180]
                            angle -= 360
                        if abs(angle) > 90:  # If angle > 90 degrees (i.e., moving away from the set direction)
                            reverse_moving_objects[track_id] = (curr_x, curr_y)

                # Traverse each retrograde id
                for reverse_id, (x, y) in reverse_moving_objects.items():
                    track = track_history[reverse_id]
                    color = track_colors[reverse_id]
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    # Draw the trajectory of the retrograde object and mark the id
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=track_thickness)
                    cv2.putText(frame, f"id:{reverse_id}", (int(x), int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

                # Render the ids of all retrograde targets in the upper left
                ids_text = "IDs of the wrong direction: " + ", ".join([str(id) for id in reverse_moving_objects.keys()])
                cv2.putText(frame, ids_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                            font_thickness, cv2.LINE_AA)

                # Write the annotated frame to the output video
                out.write(frame)

                # Display the annotated frame
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                cv2.putText(frame, f"ID of the wrong direction:", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness,
                            cv2.LINE_AA)  # Use cv2.LINE_AA for better font rendering
                out.write(frame)
                continue
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    out.release()
