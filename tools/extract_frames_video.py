import cv2
import argparse
import os
from math import ceil

def extract_frames(video_paths, output_folder, frame_rate, ds_name, width, height):
    for idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = ceil(fps / frame_rate)
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                # Resize with aspect ratio preservation and cropping if necessary
                h, w = frame.shape[:2]
                r = min(height/h, width/w)
                new_h, new_w = int(h * r), int(w * r)
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Calculate cropping
                x_center = new_w / 2
                y_center = new_h / 2
                x_start = max(0, int(x_center - (width / 2)))
                y_start = max(0, int(y_center - (height / 2)))

                cropped = resized[y_start:y_start+height, x_start:x_start+width]

                frame_filename = f"{ds_name}_{idx:05d}_{count // interval:05d}_leftImg8bit.png"
                cv2.imwrite(os.path.join(output_folder, frame_filename), cropped)
            
            count += 1
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MP4 files.")
    parser.add_argument("video_paths", nargs='+', help="Paths to the video files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the extracted frames.")
    parser.add_argument("--frame_rate", type=float, default=6, help="Frame extraction rate (frames per second).")
    parser.add_argument("--ds_name", type=str, default="ntu", help="Dataset name prefix for the output files.")
    parser.add_argument("--width", type=int, default=2048, help="Width of the output images.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the output images.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    extract_frames(args.video_paths, args.output_folder, args.frame_rate, args.ds_name, args.width, args.height)
