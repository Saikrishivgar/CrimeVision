import os
import subprocess

def generate_clip(video_path, clip_start, clip_end, output_path):
    """
    Extracts a clip from the video using FFmpeg.
    """
    if os.path.exists(output_path):
        return output_path
        
    duration = clip_end - clip_start
    
    # First attempt: stream copy (fast)
    cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-ss", str(clip_start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Stream copy failed, trying re-encoding: {e}")
        # Second attempt: re-encode (slower, but more reliable for precise cuts)
        cmd = [
            "/opt/homebrew/bin/ffmpeg",
            "-y",
            "-ss", str(clip_start),
            "-i", video_path,
            "-t", str(duration),
            "-vcodec", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e2:
            print(f"FFmpeg re-encoding failed: {e2}")
            return None
            
    return output_path
