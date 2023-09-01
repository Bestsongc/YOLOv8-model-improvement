import cv2
import os

def images_to_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")]
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# 使用方法示例
image_folder = 'D:/BaiduNetdiskDownload/MOT15/MOT15/train/ADL-Rundle-6/img1'  # 图片文件夹路径
video_name = 'D:/BaiduNetdiskDownload/MOT15/MOT15/train/ADL-Rundle-6/img1/output_video.avi'  # 输出视频文件名
fps = 30  # 每秒帧数

images_to_video(image_folder, video_name, fps)
