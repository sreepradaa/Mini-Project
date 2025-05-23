import cv2
import os

video_path = "C:\Users\Sreeprada\Downloads\gettyimages-1164849900-640_adpp.mp4"  # Replace with your video path
output_dir = "C:/MiniP/test_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 30 == 0:  # Save every 30th frame
        cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)
    frame_count += 1
cap.release()
print(f"Extracted frames saved to {output_dir}")