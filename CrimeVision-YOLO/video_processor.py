import cv2

class VideoProcessor:
    """
    Handles frame-by-frame extraction and timestamp generation for video streams.
    """
    def __init__(self, video_path):
        self.video_path = video_path
        
        # Convert string '0' to int for webcam capture
        if isinstance(video_path, str) and video_path.isdigit():
            video_path = int(video_path)
            
        # Critical fix for Apple Silicon macOS thread deadlocks in FastAPI:
        cv2.setNumThreads(0)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Avoid division by zero/uninitialized FPS (e.g. for webcams)
        if self.fps <= 0:
            self.fps = 30.0
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames < 0:
            self.total_frames = 0
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0.0

    def get_timestamp(self, frame_idx):
        """
        Converts frame index to standard timestamp string HH:MM:SS.MS
        """
        total_seconds = frame_idx / self.fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def read_frames(self):
        """
        Generator yielding (frame_index, timestamp, frame) for each frame in the video.
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            timestamp = self.get_timestamp(frame_idx)
            yield frame_idx, timestamp, frame
            frame_idx += 1

    def release(self):
        """
        Releases the OpenCV video capture resource.
        """
        if self.cap.isOpened():
            self.cap.release()
