"""Gymnasium RecordVideo subclass with ffmpeg codec fallback for Isaac Sim captures."""

from __future__ import annotations

import os

import gymnasium as gym


class EncodingRecordVideo(gym.wrappers.RecordVideo):
    """RecordVideo wrapper with controllable ffmpeg codec settings."""

    def __init__(
        self,
        *args,
        preferred_codec: str,
        bitrate: str | None,
        preset: str | None,
        ffmpeg_params: list[str] | None,
        disable_logger: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, disable_logger=disable_logger, **kwargs)
        self.preferred_codec = preferred_codec
        self.bitrate = bitrate
        self.preset = preset
        self.ffmpeg_params = ffmpeg_params
        self.disable_logger = disable_logger

    def _codec_fallback_chain(self) -> list[str]:
        chain = [self.preferred_codec]
        if self.preferred_codec.endswith("_nvenc"):
            chain.extend(["libx265", "libx264"])
        else:
            chain.extend(["libx265", "libx264"])
        deduped: list[str] = []
        for codec in chain:
            if codec not in deduped:
                deduped.append(codec)
        return deduped

    def stop_recording(self) -> None:
        """Stop recording and save video with codec fallback."""
        assert self.recording, "stop_recording was called, but no recording was started"

        if len(self.recorded_frames) == 0:
            gym.logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise gym.error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            path = os.path.join(self.video_folder, f"{self._video_name}.mp4")

            codecs = self._codec_fallback_chain()
            last_error: Exception | None = None
            for codec in codecs:
                try:
                    codec_preset = self.preset
                    if codec_preset is None:
                        codec_preset = "p5" if codec.endswith("_nvenc") else "slow"
                    print(
                        "[INFO] Video encode attempt: "
                        f"codec={codec} preset={codec_preset} bitrate={self.bitrate}"
                    )
                    clip.write_videofile(
                        path,
                        codec=codec,
                        bitrate=self.bitrate,
                        preset=codec_preset,
                        ffmpeg_params=self.ffmpeg_params,
                        audio=False,
                        logger=moviepy_logger,
                        pixel_format="yuv420p",
                    )
                    last_error = None
                    print(f"[INFO] Video encoded successfully with codec={codec}")
                    break
                except Exception as exc:  # pragma: no cover
                    last_error = exc
                    print(f"[WARN] Video encode failed with codec={codec}: {exc}")
                    continue

            del clip
            if last_error is not None:
                raise RuntimeError(
                    "All video encoding attempts failed "
                    f"(tried: {', '.join(codecs)})."
                ) from last_error

        del self.recorded_frames
        self.recorded_frames = []
        self.recording = False
        self._video_name = None

        if self.gc_trigger and self.gc_trigger(self.episode_id):
            import gc

            gc.collect()
