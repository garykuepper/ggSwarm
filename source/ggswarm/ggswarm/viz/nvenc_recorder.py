"""Video recorder using ffmpeg with NVENC H.264 hardware encoding.

Replaces gym.wrappers.RecordVideo for faster, phone-compatible video output.
"""

from __future__ import annotations

import os
import subprocess

import gymnasium as gym
import numpy as np


class NvencRecorder(gym.Wrapper):
    """Records environment renders to H.264 MP4 using ffmpeg + NVENC.

    Args:
        env: The gymnasium environment.
        video_folder: Directory to save videos.
        video_length: Number of steps to record.
        name_prefix: Filename prefix for the output video.
        fps: Frames per second in the output video.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        video_length: int = 500,
        name_prefix: str = "ggswarm",
        fps: int = 30,
    ):
        super().__init__(env)
        self.video_folder = video_folder
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.fps = fps

        self._step_count = 0
        self._recording = False
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._video_path: str | None = None
        self._width: int | None = None
        self._height: int | None = None

        os.makedirs(video_folder, exist_ok=True)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._step_count = 0
        self._start_recording()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1

        if self._recording:
            self._capture_frame()
            if self._step_count >= self.video_length:
                self._stop_recording()

        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop_recording()
        super().close()

    def _start_recording(self):
        if self._recording:
            self._stop_recording()

        # Capture first frame to get dimensions
        frame = self.render()
        if frame is None:
            return
        if isinstance(frame, np.ndarray):
            self._height, self._width = frame.shape[:2]
        else:
            return

        self._video_path = os.path.join(
            self.video_folder, f"{self.name_prefix}-episode-{self._step_count}.mp4"
        )

        # Launch ffmpeg with NVENC H.264 encoder
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self._width}x{self._height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "23",
            "-pix_fmt", "yuv420p",
            self._video_path,
        ]

        print(f"[INFO] ffmpeg command: {' '.join(cmd)}")
        self._ffmpeg_proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self._recording = True

        # Write first frame
        self._write_frame(frame)

    def _capture_frame(self):
        frame = self.render()
        if frame is not None and isinstance(frame, np.ndarray):
            self._write_frame(frame)

    def _write_frame(self, frame: np.ndarray):
        if self._ffmpeg_proc is not None and self._ffmpeg_proc.stdin is not None:
            try:
                self._ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                self._recording = False

    def _stop_recording(self):
        if self._ffmpeg_proc is not None:
            if self._ffmpeg_proc.stdin is not None:
                self._ffmpeg_proc.stdin.close()
            _, stderr = self._ffmpeg_proc.communicate()
            if stderr:
                stderr_str = stderr.decode(errors="replace")
                # Check if NVENC was actually used
                if "h264_nvenc" in stderr_str:
                    print("[INFO] Video encoder: h264_nvenc (GPU)")
                elif "libx264" in stderr_str:
                    print("[WARN] Video encoder: libx264 (CPU fallback)")
                # Print any errors
                for line in stderr_str.splitlines():
                    if "error" in line.lower() or "cannot" in line.lower():
                        print(f"[ffmpeg] {line.strip()}")
            self._ffmpeg_proc = None
        self._recording = False
        if self._video_path and os.path.exists(self._video_path):
            size_mb = os.path.getsize(self._video_path) / (1024 * 1024)
            print(f"[INFO] Video saved: {self._video_path} ({size_mb:.1f} MB)")
