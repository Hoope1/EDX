import importlib

import pytest

if importlib.util.find_spec("cv2") is None or importlib.util.find_spec("torch") is None:
    pytest.skip("required deps not available", allow_module_level=True)

from detectors import Config


def test_default_config_has_system_section() -> None:
    cfg = Config()
    assert "system" in cfg.config
    assert "use_gpu" in cfg.config["system"]
