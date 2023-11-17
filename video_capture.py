import cv2
import os

def video_to_frames(video_path, output_path):
    vc = cv2.VideoCapture(video_path)
    cnt = 0
    while(vc.isOpened):
        rval, frame = vc.read()
        if rval:
            cv2.imwrite(os.path.join(output_path,f"frame{cnt:05d}.jpg"),frame)
            cnt += 1
        else:
            break
    vc.release()

video_path = 'your_video_path'
output_path = 'save_video_path'

video_to_frames(video_path, output_path)
