import logging
import os
import threading
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

from optimus1.util.thread import MultiThreadServerAPI
from optimus1.util.utils import get_time, save_bin
from optimus1.util.video import count_video_frames, write_video

from .mod import Mod


class RecorderMod(Mod):
    video_frames = []
    export_video: bool = True
    output_video_path: str = ""
    output_video_name: str = ""
    with_prompt: bool = False

    video_sub_task: bool = False
    video_sub_task_frames = []

    action_frames = []
    export_action: bool = True

    action_sub_task: bool = False
    action_sub_task_frames = []

    _lock: threading.Lock = threading.Lock()

    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        super().__init__(cfg)
        self.logger = logger
        self.reset()

    def reset(self):
        self.export_video = self.cfg["video"]["save"]
        self.video_frames = []

        self.export_action = self.cfg["action"]["save"]
        self.action_frames = []

        if self.export_video:
            self.video_sub_task = self.cfg["video"]["sub_task"]
            self.video_sub_task_frames = []

            self.output_video_path = self.cfg["video"]["path"]
            self.output_video_name = self.cfg["video"]["name"]

            os.makedirs(self.output_video_path, exist_ok=True)
            self.with_prompt = False

        if self.export_action:
            self.action_sub_task = self.cfg["action"]["sub_task"]
            self.action_sub_task_frames = []

    def step(
        self,
        obs: Dict[str, Any],
        prompt: str | None = None,
        action: Dict[str, Any] | None = None,
    ):
        with self._lock:
            if self.export_video:
                frame = np.asarray(obs["pov"], dtype=np.uint8).copy()
                self.with_prompt = False
                self.video_frames.append(frame)
                self.video_sub_task_frames.append(frame)

            if self.export_action:
                self.action_frames.append(action)
                self.action_sub_task_frames.append(action)

    def _snapshot_frames(self, is_sub_task: bool) -> tuple[list, list]:
        with self._lock:
            video_frames = self.video_frames if is_sub_task is False else self.video_sub_task_frames
            action_frames = self.action_frames if is_sub_task is False else self.action_sub_task_frames
            return list(video_frames), list(action_frames)

    def _frame_shapes(self, frames: list) -> Dict[str, Any]:
        shapes = {}
        if frames:
            shapes["first"] = list(getattr(frames[0], "shape", ()))
            shapes["last"] = list(getattr(frames[-1], "shape", ()))
        unique = []
        seen = set()
        for frame in frames:
            shape = tuple(getattr(frame, "shape", ()))
            if shape and shape not in seen:
                seen.add(shape)
                unique.append(list(shape))
            if len(unique) >= 8:
                break
        shapes["unique_sample"] = unique
        return shapes

    def _log_video_integrity(
        self,
        output_video_filepath: str,
        video_frames: list,
        action_frames: list,
    ) -> None:
        file_size = os.path.getsize(output_video_filepath) if os.path.exists(output_video_filepath) else 0
        decoded_frames = count_video_frames(output_video_filepath)
        self.logger.info(
            "Video integrity: "
            f"expected_frames={len(video_frames)}, decoded_frames={decoded_frames}, "
            f"actions={len(action_frames)}, file_size={file_size}, "
            f"frame_shapes={self._frame_shapes(video_frames)}, raw_pov_only=True"
        )
        if decoded_frames >= 0 and decoded_frames != len(video_frames):
            self.logger.warning(
                "Video frame count mismatch after export: "
                f"expected={len(video_frames)}, decoded={decoded_frames}, file={output_video_filepath}"
            )

    def _save(
        self,
        task: str,
        status: str,
        is_sub_task: bool = False,
        actual_done_final_task: bool = "",
        biome: str = "",
        run_uuid: str = "",
        video_frames_snapshot: list | None = None,
        action_frames_snapshot: list | None = None,
    ) -> str | None:
        if self.export_video:
            # dir/{task}/{status}/{time}.mp4
            task = task.replace(" ", "_")
            actual_done_final_task = (actual_done_final_task or "").replace(" ", "_")
            video_dir = os.path.join(self.output_video_path, task, biome, status)
            os.makedirs(video_dir, exist_ok=True)
            time = get_time()

            # uid = str(uuid.uuid4())[:5]

            # output_video_filepath = os.path.join(video_dir, f"{run_uuid}_{actual_done_final_task}_{time}.mp4")
            output_video_filepath = os.path.join(video_dir, f"{time}_{actual_done_final_task}_{run_uuid}.mp4")

            # output_action_filepath = os.path.join(video_dir, f"{run_uuid}_{actual_done_final_task}_{time}.pkl")
            output_action_filepath = os.path.join(video_dir, f"{time}_{actual_done_final_task}_{run_uuid}.pkl")

            self.logger.info(
                f"[dark_violet]save video&action to {output_video_filepath}[/dark_violet]"
            )

            if video_frames_snapshot is not None:
                video_frames = video_frames_snapshot
            else:
                video_frames = self.video_frames if is_sub_task is False else self.video_sub_task_frames
            if action_frames_snapshot is not None:
                action_frames = action_frames_snapshot
            else:
                action_frames = self.action_frames if is_sub_task is False else self.action_sub_task_frames
            if not video_frames:
                self.logger.warning("No video frames recorded; skip video export.")
                return None

            with self._lock:
                if self.export_action:
                    save_bin(action_frames, output_action_filepath)

                first_shape = getattr(video_frames[0], "shape", None)
                self.logger.info(
                    "Writing episode video: "
                    f"frames={len(video_frames)}, actions={len(action_frames)}, first_shape={first_shape}"
                )
                write_video(output_video_filepath, video_frames)
                self._log_video_integrity(
                    output_video_filepath,
                    video_frames,
                    action_frames,
                )

                # if is_sub_task:
                #     self.video_sub_task_frames = []
                #     self.action_sub_task_frames = []
            return output_video_filepath

    def save(self, task: str, status: str, is_sub_task: bool = False, actual_done_final_task: bool = "", biome: str = "", run_uuid: str = ""):
        video_frames, action_frames = self._snapshot_frames(is_sub_task)
        thread = MultiThreadServerAPI(
            self._save,
            args=(task, status, is_sub_task, actual_done_final_task, biome, run_uuid, video_frames, action_frames),
        )
        thread.start()
        return thread
