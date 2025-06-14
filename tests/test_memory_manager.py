import importlib
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("cv2") is None or importlib.util.find_spec("torch") is None:
    pytest.skip("required deps not available", allow_module_level=True)

from detectors import Config, MemoryManager


def test_resize_if_needed(tmp_path: Path) -> None:
    cfg = Config()
    mem_mgr = MemoryManager(cfg)
    large_image = np.ones((5000, 5000, 3), dtype=np.uint8) * 255
    resized, scale = mem_mgr.resize_if_needed(large_image)
    assert resized.shape[0] <= cfg.config["system"]["max_image_size"][1]
    assert resized.shape[1] <= cfg.config["system"]["max_image_size"][0]
    assert scale <= 1.0
