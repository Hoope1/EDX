import streamlit as st
from pathlib import Path
import cv2
from PIL import Image

from detectors import (
    Config,
    MemoryManager,
    run_hed,
    run_structured,
    run_kornia,
    run_bdcn,
    run_fixed,
)

METHODS = [
    ("HED", run_hed),
    ("StructuredForests", run_structured),
    ("Kornia", run_kornia),
    ("BDCN", run_bdcn),
    ("FixedEdgeCNN", run_fixed),
]

st.set_page_config(page_title="Edge Detection Toolkit", layout="wide")

cfg = Config()
memory_mgr = MemoryManager(cfg)

input_dir = st.text_input("Input directory", value=str(Path("images").resolve()))
input_path = Path(input_dir)

if not input_path.exists():
    st.stop()

formats = cfg.config.get("supported_formats", [".jpg", ".png"])
images = [p for p in input_path.rglob("*") if p.suffix.lower() in formats]

st.write(f"Found {len(images)} images")

selected_methods = st.multiselect(
    "Methods", [m[0] for m in METHODS], default=[m[0] for m in METHODS]
)

if st.button("Run"):
    output_dir = input_path / "edges"
    output_dir.mkdir(exist_ok=True)
    for img_path in images:
        img = Image.open(img_path)
        st.image(img, caption=img_path.name)
        for name, func in METHODS:
            if name not in selected_methods:
                continue
            edges = func(img_path, memory_mgr)
            result_path = output_dir / f"{img_path.stem}_{name}.png"
            cv2.imwrite(str(result_path), edges)
            st.image(Image.open(result_path), caption=name)
