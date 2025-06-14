#!/usr/bin/env python3
"""
create.py – Erstellt die komplette optimierte Projektstruktur
für das Edge-Detection-Toolkit mit allen Features.

Features:
- GPU/CUDA Support mit Memory Management
- Multiprocessing für parallele Verarbeitung
- Progress Bars und robuste Downloads
- Config-File Support (YAML)
- Erweiterte Bildformate
- Unterordner-Verarbeitung

Aufruf:
    python create.py            # legt ./edge_detection_tool/ an
    python create.py --force    # löscht vorhandenen Ordner vorher
    python create.py --minimal  # ohne Beispielbilder
"""

import argparse
import shutil
from pathlib import Path
from textwrap import dedent

# ------------------------------------------------------
# Projektstruktur
# ------------------------------------------------------

BASE_DIR = Path("edge_detection_tool")

DIRS = [
    BASE_DIR,
    BASE_DIR / "models" / "hed",
    BASE_DIR / "models" / "structured",
    BASE_DIR / "images",
    BASE_DIR / "images" / "samples",  # Für Beispielbilder
    BASE_DIR / "results",
    BASE_DIR / "logs",  # Für Log-Dateien
]

# ------------------------------------------------------
# Datei-Inhalte
# ------------------------------------------------------

# requirements.txt
REQUIREMENTS = dedent(
    """
# Core Computer Vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0   # StructuredForests API

# Deep Learning Frameworks
torch>=2.0.0       # Auto-detect CUDA, falls verfügbar
torchvision>=0.15.0
kornia>=0.7.0      # GPU-beschleunigte CV-Operationen

# Image Processing
Pillow>=10.0.0     # Erweiterte Bildformat-Unterstützung
imageio>=2.31.0    # TIFF, WebP Support

# Utilities
requests>=2.31.0
tqdm>=4.66.0       # Progress Bars
pyyaml>=6.0        # Config File Support
psutil>=5.9.0      # Memory Management
numpy>=1.24.0

# Optional aber empfohlen für Performance
numba>=0.58.0      # JIT Compilation für CPU-intensive Ops
scikit-image>=0.21.0  # Zusätzliche Bildverarbeitung

# Development (optional)
pytest>=7.4.0      # Für Tests
black>=23.0.0      # Code Formatting
"""
).strip()

# config.yaml
CONFIG_YAML = dedent(
    """
# Edge Detection Toolkit Configuration
# Alle Werte können hier zentral angepasst werden

# System Settings
system:
  # GPU/CUDA Einstellungen
  use_gpu: auto  # auto, true, false
  gpu_memory_fraction: 0.8  # Maximal 80% des GPU-Speichers nutzen (wichtig bei 4GB)
  
  # Multiprocessing
  max_workers: 4  # Anzahl paralleler Prozesse (0 = auto-detect)
  chunk_size: 10  # Bilder pro Batch
  
  # Memory Management (für 16GB RAM System)
  max_image_size: [4096, 4096]  # Maximale Bildgröße vor Resize
  memory_limit_mb: 8192  # Max 8GB für Bildverarbeitung
  
# Download Settings
downloads:
  timeout: 120  # Sekunden
  max_retries: 3
  retry_delay: 5  # Sekunden zwischen Versuchen
  verify_checksums: true
  chunk_size: 8192  # Download chunk size in bytes
  
# Model URLs und Checksums
models:
  hed:
    proto_urls:
      - url: "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_edge_detection/deploy.prototxt"
        checksum: ""
      - url: "https://github.com/s9xie/hed/raw/master/examples/hed/deploy.prototxt"
        checksum: ""
    weights_urls:
      - url: "https://vcl.ucsd.edu/hed/hed-pretrained-bsds.caffemodel"
        checksum: ""
      - url: "https://github.com/s9xie/hed/raw/master/examples/hed/hed_pretrained_bsds.caffemodel"
        checksum: ""
        
  structured_forests:
    model_url: "https://github.com/opencv/opencv_extra/raw/master/testdata/cv/ximgproc/model.yml.gz"
    checksum: ""
    
# Edge Detection Parameters
edge_detection:
  # HED
  hed:
    crop_size: 500  # Für GPU Memory Management
    scale: 1.0
    mean: [104.00699, 116.66877, 122.67891]
    
  # Structured Forests
  structured_forests:
    scale: 1.0
    
  # Kornia Canny
  kornia:
    low_threshold: 0.1
    high_threshold: 0.2
    kernel_size: 5
    
  # BDCN Fallback (Canny)
  bdcn_fallback:
    blur_kernel: 5
    canny_low: 50
    canny_high: 150
    morph_kernel: 3
    
  # Fixed Edge CNN (Sobel)
  fixed_cnn:
    kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]
    
# Output Settings
output:
  # Bildformate
  save_format: "png"  # png, jpg, tiff, webp
  jpg_quality: 95
  png_compression: 6  # 0-9
  
  # Verarbeitung
  preserve_structure: true  # Unterordner beibehalten
  skip_existing: false  # Bereits verarbeitete Bilder überspringen
  
# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  file: "edge_detection.log"
  console: true
  
# Unterstützte Bildformate
supported_formats:
  - ".jpg"
  - ".jpeg"
  - ".png"
  - ".bmp"
  - ".tiff"
  - ".tif"
  - ".webp"
  - ".jp2"  # JPEG 2000
"""
).strip()

# run.bat
RUN_BAT = dedent(
    r"""
@echo off
REM Edge Detection Toolkit - Windows Batch Runner
REM Optimiert für Windows 11 mit GPU Support

echo ===============================================
echo Edge Detection Toolkit - Automatisches Setup
echo ===============================================
echo.

REM 1) Python-Version prüfen
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python nicht gefunden!
    echo Bitte Python 3.8+ installieren: https://www.python.org/
    pause
    exit /b 1
)

REM 2) Virtuelle Umgebung anlegen (falls nicht vorhanden)
IF NOT EXIST venv (
    echo [1/6] Erstelle virtuelle Umgebung...
    python -m venv venv
    
    REM Aktiviere und upgrade pip einmalig
    call venv\Scripts\activate
    python -m pip install --upgrade pip wheel setuptools >nul 2>&1
) ELSE (
    echo [1/6] Virtuelle Umgebung gefunden
    call venv\Scripts\activate
)

REM 3) Requirements installieren
echo [2/6] Installiere Pakete...
python -m pip install -r requirements.txt

REM 4) GPU/CUDA Check
echo [3/6] Pruefe GPU/CUDA Support...
python -c "import torch; print('[GPU]', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Keine CUDA GPU gefunden')"

REM 5) Modelle herunterladen
echo [4/6] Lade Modelle herunter...
python detectors.py --init-models

REM 6) Modelle verifizieren
echo [5/6] Verifiziere Installation...
python detectors.py --verify

REM 7) Beispielbilder check
IF NOT EXIST images\*.* (
    echo.
    echo [WARNING] Keine Bilder im 'images' Ordner gefunden!
    echo Bitte Bilder in den 'images' Ordner kopieren.
    echo.
    pause
    exit /b 0
)

REM 8) Edge Detection starten
echo [6/6] Starte Kantenerkennung...
echo.
echo Verwende folgende Einstellungen:
echo - Input:  images\
echo - Output: results\
echo - GPU:    auto-detect
echo - Worker: auto (CPU Kerne)
echo.

REM Hauptverarbeitung mit erweiterten Optionen
python run_edge_detectors.py --input_dir images --output_dir results

echo.
echo ===============================================
echo Verarbeitung abgeschlossen!
echo Ergebnisse in: results\
echo ===============================================
echo.

REM Optional: Ergebnisse anzeigen
choice /C YN /M "Moechten Sie den Ergebnis-Ordner oeffnen"
IF ERRORLEVEL 2 GOTO END
IF ERRORLEVEL 1 start "" "results"

:END
pause
"""
).strip()

# README.md (gekürzte Version)
README_MD = dedent(
    """
# Edge Detection Toolkit

Ein GPU-beschleunigtes Toolkit für Batch-Kantenerkennung mit 5 verschiedenen Algorithmen, Multiprocessing-Support und intelligenter Speicherverwaltung.

## Features

- **5 Edge Detection Methoden**: HED, Structured Forests, Kornia, BDCN (Fallback), Fixed CNN
- **GPU/CUDA Support**: Automatische Erkennung und Optimierung
- **Multiprocessing**: Parallele Verarbeitung mehrerer Bilder
- **Memory Management**: Automatisches Resizing großer Bilder
- **Erweiterte Bildformate**: JPG, PNG, TIFF, WebP, BMP, JPEG2000
- **Progress Tracking**: Fortschrittsbalken für alle Operationen
- **YAML-Konfiguration**: Zentrale Einstellungsverwaltung

## Installation

### Windows Schnellstart

```bash
python create.py
cd edge_detection_tool
run.bat
```

### Manuelle Installation

```bash
python create.py
cd edge_detection_tool
python -m venv venv
# Windows: venv\\Scripts\\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python detectors.py --init-models
```

## Verwendung

```bash
# Basis
python run_edge_detectors.py --input_dir images --output_dir results

# Erweiterte Optionen
python run_edge_detectors.py -i images -o results --methods HED Kornia
python run_edge_detectors.py -i images -o results --skip-existing --workers 8
```

## Konfiguration

Siehe `config.yaml` für alle Einstellungen. Wichtige Parameter:

- `system.use_gpu`: GPU-Nutzung (auto/true/false)
- `system.max_workers`: Anzahl paralleler Prozesse
- `output.save_format`: Ausgabeformat (png/jpg/tiff/webp)

## Lizenz

MIT License
"""
).strip()

# ------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------


def create_sample_image(path: Path, size: tuple = (512, 512)):
    """Erstelle ein Beispielbild mit Kanten zum Testen"""
    try:
        import cv2
        import numpy as np

        # Erstelle Bild mit geometrischen Formen
        img = np.ones(size + (3,), dtype=np.uint8) * 255

        # Rechteck
        cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), 3)

        # Kreis
        cv2.circle(img, (350, 150), 80, (0, 255, 0), 3)

        # Linien
        cv2.line(img, (50, 300), (450, 450), (255, 0, 0), 2)
        cv2.line(img, (450, 300), (50, 450), (255, 0, 0), 2)

        # Text
        cv2.putText(img, "Edge Detection Test", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Speichern
        cv2.imwrite(str(path), img)
        return True

    except ImportError:
        print("[info] OpenCV nicht verfügbar - Beispielbild wird später erstellt")
        return False


def create_file(path: Path, content: str):
    """Erstelle Datei mit Inhalt"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"[created] {path}")


# ------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Edge Detection Toolkit - Projekt-Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent(
            """
        Dieses Script erstellt die komplette Projektstruktur für das
        Edge Detection Toolkit mit allen Optimierungen.
        
        Nach der Erstellung:
          cd edge_detection_tool
          run.bat  # Windows
          # oder manuell für Linux/Mac - siehe README.md
        """
        ),
    )

    parser.add_argument("--force", action="store_true", help="Lösche vorhandenen Ordner vor Erstellung")
    parser.add_argument("--minimal", action="store_true", help="Erstelle keine Beispielbilder")

    args = parser.parse_args()

    # Bei --force alten Ordner löschen
    if args.force and BASE_DIR.exists():
        print(f"[delete] Lösche {BASE_DIR}")
        shutil.rmtree(BASE_DIR)

    # Verzeichnisse erstellen
    print("\n=== Erstelle Verzeichnisstruktur ===")
    for directory in DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[mkdir] {directory}")

    # Dateien erstellen
    print("\n=== Erstelle Projektdateien ===")

    files = [
        (BASE_DIR / "requirements.txt", REQUIREMENTS),
        (BASE_DIR / "config.yaml", CONFIG_YAML),
        (BASE_DIR / "run.bat", RUN_BAT),
        (BASE_DIR / "README.md", README_MD),
    ]

    for file_path, content in files:
        create_file(file_path, content)

    # Info über große Dateien
    print("\n[INFO] Die folgenden Dateien müssen aus den Artifacts kopiert werden:")
    print("  - detectors.py (aus detectors_optimized)")
    print("  - run_edge_detectors.py (aus run_edge_detectors_optimized)")

    # Beispielbilder
    if not args.minimal:
        print("\n=== Erstelle Beispielbilder ===")
        sample_paths = [
            BASE_DIR / "images" / "samples" / "test_shapes.png",
        ]

        for sample_path in sample_paths:
            if create_sample_image(sample_path):
                print(f"[created] {sample_path}")

    # Abschluss
    print(f"\n✅ Projekt erfolgreich erstellt in: {BASE_DIR.absolute()}")
    print("\nNächste Schritte:")
    print(f"  1. cd {BASE_DIR}")
    print("  2. Kopiere detectors.py und run_edge_detectors.py aus den Artifacts")
    print("  3. git submodule update --init --recursive  # BDCN")
    print("  4. run.bat  # Für Windows")
    print("\nOder manuell:")
    print("  python -m venv venv")
    print("  venv\\Scripts\\activate  # Windows")
    print("  source venv/bin/activate  # Linux/Mac")
    print("  pip install -r requirements.txt")
    print("  python detectors.py --init-models")
    print("  python run_edge_detectors.py -i images -o results")


if __name__ == "__main__":
    main()
