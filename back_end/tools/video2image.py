import cv2
import os

def video_to_image(image_folder, video_name):
    cap = cv2.VideoCapture(video_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        image_name = f"frame_{frame_count:04d}.jpg"
        image_path = os.path.join(image_folder, image_name)

        cv2.imwrite(image_path, frame)



    cap.release()
    cv2.destroyAllWindows()

# 使用方法示例
image_folder = 'D:/ultralytics-main/pictures'  # 图片文件夹路径
video_name = 'D:/微信文件/WeChat Files/wxid_jmmnslqz5lwl22/FileStorage/File/2023-08/video.avi'  # 输出视频文件名


video_to_image(image_folder, video_name)