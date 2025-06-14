#!/usr/bin/env python3
"""
Edge Detection Toolkit - Detector Module
Unterstützt HED, Structured Forests, Kornia, BDCN und Fixed CNN
Mit GPU-Support, Memory Management und robusten Downloads
"""

import argparse
import gzip
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import psutil
import requests
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Basis-Verzeichnisse
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
HED_DIR = MODEL_DIR / "hed"
STRUCT_DIR = MODEL_DIR / "structured"
CONFIG_FILE = BASE_DIR / "config.yaml"
BDCN_REPO = BASE_DIR / "bdcn_repo"
BDCN_MODEL = BDCN_REPO / "pretrained" / "bdcn_pretrained.pth"

# ------------------------------------------------------
# Configuration Management
# ------------------------------------------------------


class Config:
    """Zentrale Konfigurationsverwaltung"""

    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config = self._load_config(config_path)
        self._setup_gpu()

    def _load_config(self, path: Path) -> dict:
        """Lade Konfiguration aus YAML"""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            print(f"[warning] Config nicht gefunden: {path}, verwende Defaults")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Standard-Konfiguration wenn keine config.yaml existiert"""
        return {
            "system": {
                "use_gpu": "auto",
                "gpu_memory_fraction": 0.8,
                "max_workers": 4,
                "chunk_size": 10,
                "max_image_size": [4096, 4096],
                "memory_limit_mb": 8192,
            },
            "downloads": {
                "timeout": 120,
                "max_retries": 3,
                "retry_delay": 5,
                "verify_checksums": True,
                "chunk_size": 8192,
            },
            "edge_detection": {
                "hed": {
                    "crop_size": 500,
                    "scale": 1.0,
                    "mean": [104.00699, 116.66877, 122.67891],
                },
                "kornia": {
                    "low_threshold": 0.1,
                    "high_threshold": 0.2,
                    "kernel_size": 5,
                },
                "bdcn_fallback": {
                    "blur_kernel": 5,
                    "canny_low": 50,
                    "canny_high": 150,
                    "morph_kernel": 3,
                },
                "fixed_cnn": {"kernel": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]},
            },
            "output": {
                "save_format": "png",
                "jpg_quality": 95,
                "png_compression": 6,
                "preserve_structure": True,
                "skip_existing": False,
            },
            "supported_formats": [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".webp",
                ".jp2",
            ],
        }

    def _setup_gpu(self):
        """GPU/CUDA Setup mit Memory Management"""
        use_gpu = self.config["system"]["use_gpu"]

        if use_gpu == "auto":
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = use_gpu and torch.cuda.is_available()

        if self.use_cuda:
            # GPU Memory Fraction setzen
            fraction = self.config["system"]["gpu_memory_fraction"]
            torch.cuda.set_per_process_memory_fraction(fraction)

            # Device Info ausgeben
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[GPU] Verwende {device_name} mit {total_memory:.1f}GB Speicher")
            print(f"[GPU] Memory Fraction: {fraction * 100:.0f}%")
        else:
            print("[CPU] Keine GPU gefunden oder deaktiviert")

    @property
    def device(self):
        return torch.device("cuda" if self.use_cuda else "cpu")


# Globale Config-Instanz
config = None

# ------------------------------------------------------
# Download Management mit Progress und Checksums
# ------------------------------------------------------


class DownloadManager:
    """Robuster Download-Manager mit Progress-Bar und Checksums"""

    def __init__(self, cfg: Config):
        self.config = cfg.config["downloads"]

    def download_with_progress(
        self, url: str, dst: Path, checksum: Optional[str] = None
    ) -> bool:
        """Download mit Progress-Bar und optionaler Checksum-Verifizierung"""
        if dst.exists():
            if checksum and self.config["verify_checksums"]:
                if self._verify_checksum(dst, checksum):
                    print(f"[exists] {dst.name} bereits vorhanden und verifiziert")
                    return True
                else:
                    print(f"[warning] {dst.name} Checksum falsch, lade neu")
                    dst.unlink()
            else:
                return True

        # Download mit Retries
        for attempt in range(self.config["max_retries"]):
            try:
                print(
                    f"[download] {dst.name} (Versuch {attempt + 1}/{self.config['max_retries']})"
                )

                response = requests.get(
                    url, stream=True, timeout=self.config["timeout"]
                )
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                # Progress Bar
                with open(dst, "wb") as f:
                    with tqdm(
                        total=total_size, unit="B", unit_scale=True, desc=dst.name
                    ) as pbar:
                        for chunk in response.iter_content(
                            chunk_size=self.config["chunk_size"]
                        ):
                            f.write(chunk)
                            pbar.update(len(chunk))

                # Checksum verifizieren
                if checksum and self.config["verify_checksums"]:
                    if self._verify_checksum(dst, checksum):
                        print(
                            f"[success] {dst.name} erfolgreich heruntergeladen und verifiziert"
                        )
                        return True
                    else:
                        print(f"[error] {dst.name} Checksum-Fehler")
                        dst.unlink()
                        if attempt < self.config["max_retries"] - 1:
                            time.sleep(self.config["retry_delay"])
                            continue
                        return False

                return True

            except Exception as e:
                print(f"[error] Download fehlgeschlagen: {e}")
                if dst.exists():
                    dst.unlink()
                if attempt < self.config["max_retries"] - 1:
                    time.sleep(self.config["retry_delay"])
                else:
                    return False

        return False

    def download_google_drive(self, file_id: str, dst: Path) -> bool:
        """Spezieller Handler für Google Drive Downloads"""
        try:
            import gdown
        except ImportError:
            print("[info] Installiere gdown für Google Drive Support...")
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown

        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(dst), quiet=False)
            return dst.exists()
        except Exception as e:
            print(f"[error] Google Drive Download fehlgeschlagen: {e}")
            return False

    def _verify_checksum(self, file_path: Path, expected: str) -> bool:
        """MD5 Checksum Verifizierung"""
        if not expected:
            return True

        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)

        actual = md5.hexdigest()
        return actual == expected


# ------------------------------------------------------
# Memory Management
# ------------------------------------------------------


class MemoryManager:
    """Speicherverwaltung für große Bilder"""

    def __init__(self, cfg: Config):
        self.config = cfg.config["system"]
        self.max_size = tuple(self.config["max_image_size"])
        self.memory_limit = self.config["memory_limit_mb"] * 1024 * 1024  # in Bytes

    def check_memory(self) -> Tuple[float, float]:
        """Prüfe verfügbaren Speicher"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        used_percent = memory.percent
        return available_mb, used_percent

    def resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize Bild wenn zu groß für Speicher"""
        h, w = image.shape[:2]
        scale = 1.0

        # Check Größe
        if h > self.max_size[1] or w > self.max_size[0]:
            scale = min(self.max_size[0] / w, self.max_size[1] / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"[resize] Bild zu groß ({w}x{h}), resize auf {new_w}x{new_h}")
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Check Memory
        available_mb, used_percent = self.check_memory()
        if used_percent > 80:
            print(f"[warning] Hohe Speicherauslastung: {used_percent:.1f}%")

        return image, scale


# ------------------------------------------------------
# Model Initialization
# ------------------------------------------------------


def init_models() -> None:
    """Initialisiere alle Modelle mit robusten Downloads"""
    global config
    config = Config()

    # Verzeichnisse erstellen
    HED_DIR.mkdir(parents=True, exist_ok=True)
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    downloader = DownloadManager(config)

    # HED Model URLs (aktualisiert und erweitert)
    hed_urls = [
        {
            "proto": "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt",
            "weights": "https://vcl.ucsd.edu/hed/hed-pretrained-bsds.caffemodel",
            "weights_alt": [
                "https://github.com/s9xie/hed/raw/master/examples/hed/hed_pretrained_bsds.caffemodel",
                "https://www.dropbox.com/s/3jv2kket4iy9dqh/hed_pretrained_bsds.caffemodel?dl=1",
            ],
        }
    ]

    # Download HED
    proto_path = HED_DIR / "deploy.prototxt"
    weights_path = HED_DIR / "hed.caffemodel"

    proto_success = False
    weights_success = False

    for urls in hed_urls:
        if not proto_success:
            proto_success = downloader.download_with_progress(urls["proto"], proto_path)

        if not weights_success:
            # Versuche Haupt-URL
            weights_success = downloader.download_with_progress(
                urls["weights"], weights_path
            )

            # Fallbacks
            if not weights_success:
                for alt_url in urls.get("weights_alt", []):
                    if "drive.google.com" in alt_url or "dropbox.com" in alt_url:
                        # Spezialbehandlung für Cloud-Storage
                        print("[info] Versuche Cloud-Download...")
                        if "drive.google.com" in alt_url:
                            file_id = alt_url.split("/d/")[1].split("/")[0]
                            weights_success = downloader.download_google_drive(
                                file_id, weights_path
                            )
                        else:
                            # Dropbox: dl=1 Parameter wichtig
                            weights_success = downloader.download_with_progress(
                                alt_url, weights_path
                            )
                    else:
                        weights_success = downloader.download_with_progress(
                            alt_url, weights_path
                        )

                    if weights_success:
                        break

        if proto_success and weights_success:
            break

    if not proto_success or not weights_success:
        print("[warning] HED Model konnte nicht vollständig geladen werden")

    # Structured Forests
    gz_path = STRUCT_DIR / "model.yml.gz"
    yml_path = STRUCT_DIR / "model.yml"

    struct_url = "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz"
    if downloader.download_with_progress(struct_url, gz_path):
        if not yml_path.exists() and gz_path.exists():
            try:
                with gzip.open(gz_path, "rb") as fi:
                    with open(yml_path, "wb") as fo:
                        fo.write(fi.read())
                print("[success] Structured Forests Model entpackt")
            except Exception as e:
                print(f"[error] Entpacken fehlgeschlagen: {e}")

    # BDCN Weights (optional)
    if BDCN_REPO.exists():
        BDCN_MODEL.parent.mkdir(parents=True, exist_ok=True)
        downloader.download_with_progress(
            "https://github.com/zijundeng/BDCN/releases/download/v1.0.0/bdcn_pretrained.pth",
            BDCN_MODEL,
        )


# ------------------------------------------------------
# BDCN Integration
# ------------------------------------------------------


def load_bdcn_model(device: torch.device) -> nn.Module:
    """Lade das BDCN-Netzwerk und die Gewichte."""
    sys.path.insert(0, str(BDCN_REPO / "model"))
    from bdcn import BDCN
    from utils import load_checkpoint

    net = BDCN()
    checkpoint = torch.load(BDCN_MODEL, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    load_checkpoint(net, checkpoint)
    net.to(device).eval()
    return net


# ------------------------------------------------------
# Edge Detection Methods (optimiert)
# ------------------------------------------------------


def run_hed(
    image_path: Union[str, Path], memory_mgr: Optional[MemoryManager] = None
) -> np.ndarray:
    """HED Edge Detection mit Memory Management"""
    proto = HED_DIR / "deploy.prototxt"
    weights = HED_DIR / "hed.caffemodel"

    if not proto.exists() or not weights.exists():
        raise RuntimeError("HED Model nicht verfügbar")

    # Lade Bild
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    # Memory Management
    if memory_mgr:
        img, scale = memory_mgr.resize_if_needed(img)

    H, W = img.shape[:2]

    # Lade Netzwerk
    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

    # GPU Support für OpenCV DNN
    if config and config.use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Konfiguration aus config.yaml
    cfg = config.config["edge_detection"]["hed"] if config else {}
    mean = cfg.get("mean", [104.00699, 116.66877, 122.67891])

    # Verarbeitung
    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), mean, swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()[0, 0]
    out = cv2.resize(out, (W, H))

    return (out * 255).astype("uint8")


def run_structured(
    image_path: Union[str, Path], memory_mgr: Optional[MemoryManager] = None
) -> np.ndarray:
    """Structured Forests Edge Detection"""
    model_path = STRUCT_DIR / "model.yml"

    if not model_path.exists():
        raise RuntimeError("Structured Forests Model nicht verfügbar")

    # Model laden
    detector = cv2.ximgproc.createStructuredEdgeDetection(str(model_path))

    # Bild laden
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    # Memory Management
    if memory_mgr:
        img, scale = memory_mgr.resize_if_needed(img)

    # Normalisierung
    img_norm = img.astype("float32") / 255.0

    # Edge Detection
    edges = detector.detectEdges(img_norm)

    return (edges * 255).astype("uint8")


def run_kornia(
    image_path: Union[str, Path], memory_mgr: Optional[MemoryManager] = None
) -> np.ndarray:
    """Kornia Canny Edge Detection mit GPU Support"""
    import kornia

    # Lade als Grayscale
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    # Memory Management
    if memory_mgr:
        gray, scale = memory_mgr.resize_if_needed(gray)

    # Konfiguration
    cfg = config.config["edge_detection"]["kornia"] if config else {}

    # Konvertiere zu Tensor
    tensor = torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # GPU wenn verfügbar
    if config:
        tensor = tensor.to(config.device)

    # Canny Edge Detection
    edges_mag, edges_dir = kornia.filters.canny(
        tensor,
        low_threshold=cfg.get("low_threshold", 0.1),
        high_threshold=cfg.get("high_threshold", 0.2),
        kernel_size=(cfg.get("kernel_size", 5), cfg.get("kernel_size", 5)),
    )

    # Zurück zu CPU und numpy
    edges = edges_mag[0, 0].cpu().numpy()

    return (edges * 255).astype("uint8")


def run_bdcn(
    image_path: Union[str, Path], memory_mgr: Optional[MemoryManager] = None
) -> np.ndarray:
    """BDCN Edge Detection mit verbessertem Fallback"""
    if BDCN_REPO.exists() and BDCN_MODEL.exists():
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

        if memory_mgr:
            img, _ = memory_mgr.resize_if_needed(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tensor = (
            torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        device = config.device if config else torch.device("cpu")
        tensor = tensor.to(device)
        net = load_bdcn_model(device)

        with torch.no_grad():
            edge = net(tensor)[0].squeeze().cpu().numpy()
        return (edge * 255).astype("uint8")

    # Fallback: Advanced Canny
    print("[info] BDCN nicht verfügbar, verwende erweiterten Canny-Algorithmus")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    if memory_mgr:
        img, scale = memory_mgr.resize_if_needed(img)

    # Konfiguration
    cfg = config.config["edge_detection"]["bdcn_fallback"] if config else {}

    # Bilateral Filter für bessere Kanten
    img_filtered = cv2.bilateralFilter(img, 9, 75, 75)

    # Adaptive Threshold für lokale Anpassung
    adaptive = cv2.adaptiveThreshold(
        img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Canny auf gefiltertem Bild
    edges_canny = cv2.Canny(
        img_filtered, cfg.get("canny_low", 50), cfg.get("canny_high", 150)
    )

    # Kombiniere beide Methoden
    edges_combined = cv2.bitwise_or(edges_canny, adaptive)

    # Morphologische Operationen
    kernel = np.ones((cfg.get("morph_kernel", 3), cfg.get("morph_kernel", 3)), np.uint8)
    edges_final = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)

    return edges_final


def run_fixed(
    image_path: Union[str, Path], memory_mgr: Optional[MemoryManager] = None
) -> np.ndarray:
    """Fixed Edge CNN (Sobel) mit GPU Support"""
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    if memory_mgr:
        gray, scale = memory_mgr.resize_if_needed(gray)

    # Konfiguration
    cfg = config.config["edge_detection"]["fixed_cnn"] if config else {}
    kernel_data = cfg.get("kernel", [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Sobel Kernel
    kernel = torch.tensor(kernel_data, dtype=torch.float32)

    # Conv2D Layer
    conv_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    conv_x.weight.data = kernel.unsqueeze(0).unsqueeze(0)

    conv_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    conv_y.weight.data = kernel.t().unsqueeze(0).unsqueeze(0)

    # GPU wenn verfügbar
    if config:
        conv_x = conv_x.to(config.device)
        conv_y = conv_y.to(config.device)

    # Verarbeitung
    with torch.no_grad():
        tensor = (
            torch.tensor(gray / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )

        if config:
            tensor = tensor.to(config.device)

        # Gradients
        gx = conv_x(tensor)
        gy = conv_y(tensor)

        # Magnitude
        magnitude = torch.sqrt(gx**2 + gy**2)

        # Zurück zu CPU
        edges = magnitude.squeeze().cpu().numpy()

    return (edges * 255).astype("uint8")


# ------------------------------------------------------
# CLI Interface
# ------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Edge Detection Toolkit - Model Management"
    )
    parser.add_argument(
        "--init-models",
        action="store_true",
        help="Download und initialisiere alle Modelle",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verifiziere installierte Modelle"
    )
    parser.add_argument(
        "--gpu-info", action="store_true", help="Zeige GPU Informationen"
    )

    args = parser.parse_args()

    if args.gpu_info:
        config = Config()
        if torch.cuda.is_available():
            print(f"CUDA verfügbar: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Speicher: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            )
        else:
            print("Keine CUDA-fähige GPU gefunden")

    elif args.verify:
        config = Config()
        print("\n=== Model Verification ===")

        # HED
        hed_proto = HED_DIR / "deploy.prototxt"
        hed_weights = HED_DIR / "hed.caffemodel"
        print(f"HED Proto: {'✓' if hed_proto.exists() else '✗'}")
        print(f"HED Weights: {'✓' if hed_weights.exists() else '✗'}")

        # Structured Forests
        struct_model = STRUCT_DIR / "model.yml"
        print(f"Structured Forests: {'✓' if struct_model.exists() else '✗'}")

        # BDCN
        print(f"BDCN Repository: {'✓' if BDCN_REPO.exists() else '✗'}")
        print(f"BDCN Weights: {'✓' if BDCN_MODEL.exists() else '✗'}")

    elif args.init_models:
        init_models()
    else:
        parser.print_help()
