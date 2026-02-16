import subprocess
import os

def video_to_frames(video_path, fps=30, resolution=(640, 480)):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_full_path = os.path.join(script_dir, video_path)
    
    print(f"Started processing of {video_full_path}")
    
    folder_name = os.path.join(script_dir, "Images")
    os.makedirs(folder_name, exist_ok=True)

    command = [
        'ffmpeg',
        '-i', video_full_path,
        '-vf', f'fps={fps},scale={resolution[0]}:{resolution[1]},format=gray',
        os.path.join(folder_name, 'img%04d.png')
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Success! Frames are in: {folder_name}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")


video_to_frames("video.mp4", 10)