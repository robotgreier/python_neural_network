import subprocess
import os

def video_to_frames(video_path, fps=30, resolution=(640, 480)):
    # Create a folder based on the video filename (e.g., "my_video_frames")
    folder_name = os.path.splitext(os.path.basename(video_path))[0] + "_frames"
    os.makedirs(folder_name, exist_ok=True)

    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps},scale={resolution[0]}:{resolution[1]},format=gray',
        f'{folder_name}/img%04d.png'
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Success! Frames are in: {folder_name}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")