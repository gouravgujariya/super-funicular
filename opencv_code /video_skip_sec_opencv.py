import cv2
import os

# Path to the folder containing video files
video_folder = 'C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\videos\\NonViolence'
# Directory to save the extracted frames
output_dir = 'NonViolence'
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

# Iterate through each file in the video folder
for video_filename in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_filename)

    if not os.path.isfile(video_path):
        continue

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 if fps is not available

    # Calculate the number of frames to skip
    frames_to_skip = int(fps * 3)  # Skip approximately every 3 seconds

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the current frame as an image file
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:08d}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

        # Skip frames
        for _ in range(frames_to_skip):
            ret = video_capture.grab()
            if not ret:
                break

    # Release the video capture object
    video_capture.release()

    print(f"Extracted frames from video '{video_filename}'.")

print("Finished extracting frames from all videos.")
