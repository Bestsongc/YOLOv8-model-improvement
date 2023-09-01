import cv2
from ultralytics import YOLO
import time
# Load the YOLOv8 model
model = YOLO('/yolov8n.pt')


# Open the video file
video_path = "D:/Download/alley_-_42696 (360p).mp4"
cap = cv2.VideoCapture(video_path)
#要保存视频的话
#output_path = "runs/track_result_video/output_video5.avi" #保存视频的路径
#fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
#out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


total_time=0
total_frames=0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        total_frames+=1
        inference_start = time.time()

        results = model.predict(frame,device=0)

        inference_end = time.time()

        inference_time = inference_end - inference_start
        total_time+=inference_time
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        #out.write(annotated_frame)

        # Display the annotated frame
        #cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:

        # Break the loop if the end of the video is reached
        break

average_fps=total_frames/total_time
print(average_fps)
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()