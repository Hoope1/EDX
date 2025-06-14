# Edge Detection Toolkit - Umfassende Dokumentation

Ein hochoptimiertes, GPU-beschleunigtes Toolkit für verschiedene Kantenerkennungs-Algorithmen mit Multiprocessing-Support, Memory-Management und erweiterten Features für professionelle Bildverarbeitung.

## Inhaltsverzeichnis

1. [Überblick](#überblick)
2. [Features im Detail](#features-im-detail)
3. [Systemanforderungen](#systemanforderungen)
4. [Installation](#installation)
5. [Konfiguration](#konfiguration)
6. [Verwendung](#verwendung)
7. [Edge Detection Methoden](#edge-detection-methoden)
8. [Kommandozeilen-Referenz](#kommandozeilen-referenz)
9. [Performance-Optimierung](#performance-optimierung)
10. [Troubleshooting](#troubleshooting)
11. [Entwickler-Dokumentation](#entwickler-dokumentation)
12. [FAQ](#faq)
13. [Changelog](#changelog)
14. [Lizenz](#lizenz)

---

## Überblick

Das Edge Detection Toolkit ist eine professionelle Lösung zur Batch-Verarbeitung von Bildern mit verschiedenen Kantenerkennungs-Algorithmen. Es wurde speziell für maximale Performance und Flexibilität entwickelt und unterstützt sowohl CPU- als auch GPU-basierte Verarbeitung.

### Kernmerkmale

- **5 verschiedene Edge-Detection-Algorithmen** mit unterschiedlichen Stärken
- **Automatische GPU/CUDA-Erkennung** und -Optimierung
- **Multiprocessing** für parallele Verarbeitung mehrerer Bilder
- **Intelligentes Memory-Management** für große Bilder
- **Robuste Download-Mechanismen** mit Fallbacks und Checksums
- **Umfangreiche Bildformat-Unterstützung** inkl. RAW-Formate
- **Progress-Tracking** für alle Operationen
- **Zentrale Konfiguration** über YAML-Datei

---

## Features im Detail

### 🎯 Edge Detection Algorithmen

1. **HED (Holistically-Nested Edge Detection)**
   - Deep Learning basiert (Caffe)
   - Beste Ergebnisse für natürliche Bilder
   - GPU-beschleunigt über OpenCV DNN

2. **Structured Forests**
   - Machine Learning basiert
   - Schnell und robust
   - Gute Balance zwischen Geschwindigkeit und Qualität

3. **Kornia Canny**
   - GPU-optimierte Canny-Implementation
   - PyTorch-basiert
   - Ideal für Echtzeit-Anwendungen

4. **BDCN (Bi-Directional Cascade Network)**
   - Modernster Deep Learning Ansatz
   - Fallback auf erweiterten Canny wenn nicht verfügbar
   - Beste Ergebnisse für komplexe Szenen

5. **Fixed Edge CNN (Sobel)**
   - GPU-beschleunigter Sobel-Filter
   - Sehr schnell
   - Ideal für einfache Kantenerkennung

### 🚀 Performance-Features

- **GPU-Beschleunigung**
  - Automatische CUDA-Erkennung
  - Konfigurierbares Memory-Management
  - Unterstützung für Multi-GPU-Systeme (experimentell)

- **Multiprocessing**
  - Parallele Verarbeitung mehrerer Bilder
  - Automatische CPU-Kern-Erkennung
  - Konfigurierbares Chunking

- **Memory-Management**
  - Automatisches Resizing großer Bilder
  - Speicher-Monitoring
  - Out-of-Memory-Prävention

### 📊 Erweiterte Features

- **Bildformat-Support**
  - Standard: JPG, PNG, BMP
  - Erweitert: TIFF, WebP, JPEG2000
  - RAW: CR2, NEF, ARW (über Pillow)

- **Batch-Processing**
  - Unterordner-Unterstützung
  - Struktur-Erhaltung
  - Skip-Existing-Modus

- **Progress-Tracking**
  - Download-Progress mit Geschwindigkeit
  - Batch-Progress mit ETA
  - Memory-Usage-Anzeige

---

## Systemanforderungen

### Minimum
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8 oder höher
- **RAM**: 8 GB
- **CPU**: Quad-Core (4 Threads)
- **Festplatte**: 2 GB freier Speicher
- **GPU**: Optional (beliebige CUDA-fähige GPU)

### Empfohlen
- **OS**: Windows 11, Ubuntu 22.04+
- **Python**: 3.10 oder 3.11
- **RAM**: 16 GB oder mehr
- **CPU**: 6+ Cores (12+ Threads)
- **Festplatte**: SSD mit 10 GB freiem Speicher
- **GPU**: NVIDIA mit 4+ GB VRAM (z.B. GTX 1650, Quadro T1000)

### Optimal
- **RAM**: 32 GB
- **CPU**: 8+ Cores (16+ Threads)
- **GPU**: NVIDIA RTX 3060 oder besser (8+ GB VRAM)
- **Festplatte**: NVMe SSD

### GPU-Kompatibilität
- **NVIDIA**: Alle CUDA-fähigen GPUs (Compute Capability 3.5+)
- **AMD**: Experimenteller Support über ROCm (Linux)
- **Intel**: Nicht unterstützt

---

## Installation

### Schnellstart (Windows)

```bash
# 1. Repository klonen oder create.py herunterladen
python create.py

# 2. In Projektordner wechseln
cd edge_detection_tool

# 3. Automatisches Setup ausführen
run.bat
```

### Manuelle Installation (alle Plattformen)

#### 1. Projekt erstellen

```bash
# Mit allen Features
python create.py

# Minimale Installation (ohne Beispielbilder)
python create.py --minimal

# Vorhandenes Projekt überschreiben
python create.py --force
```

#### 2. Virtuelle Umgebung einrichten

**Windows:**
```cmd
cd edge_detection_tool
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
cd edge_detection_tool
python3 -m venv venv
source venv/bin/activate
```

#### 3. Dependencies installieren

```bash
# Basis-Installation
pip install -r requirements.txt

# Mit GPU-Support (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Mit GPU-Support (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Modelle herunterladen

```bash
# Alle Modelle herunterladen
python detectors.py --init-models

# Installation verifizieren
python detectors.py --verify

# GPU-Info anzeigen
python detectors.py --gpu-info
```

### Docker-Installation (experimentell)

```dockerfile
FROM python:3.10-slim

# System-Dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    wget git && rm -rf /var/lib/apt/lists/*

# Projekt kopieren
COPY . /app
WORKDIR /app

# Python-Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Modelle herunterladen
RUN python detectors.py --init-models

ENTRYPOINT ["python", "run_edge_detectors.py"]
```

---

## Konfiguration

### config.yaml Struktur

Die zentrale Konfigurationsdatei `config.yaml` steuert alle Aspekte des Toolkits:

#### System-Einstellungen

```yaml
system:
  # GPU/CUDA Einstellungen
  use_gpu: auto        # auto|true|false - GPU-Nutzung
  gpu_memory_fraction: 0.8  # 0.1-1.0 - Max GPU-Speicher-Nutzung
  
  # Multiprocessing
  max_workers: 4       # Anzahl paralleler Prozesse (0=auto)
  chunk_size: 10       # Bilder pro Batch
  
  # Memory Management
  max_image_size: [4096, 4096]  # [Breite, Höhe] - Max Bildgröße
  memory_limit_mb: 8192         # Max RAM-Nutzung in MB
```

#### Download-Einstellungen

```yaml
downloads:
  timeout: 120         # Timeout in Sekunden
  max_retries: 3       # Anzahl Wiederholungsversuche
  retry_delay: 5       # Wartezeit zwischen Versuchen
  verify_checksums: true  # MD5-Verifizierung
  chunk_size: 8192     # Download-Chunk-Größe in Bytes
```

#### Edge Detection Parameter

```yaml
edge_detection:
  # HED Einstellungen
  hed:
    crop_size: 500     # Kachel-Größe für GPU-Processing
    scale: 1.0         # Skalierungsfaktor
    mean: [104.00699, 116.66877, 122.67891]  # BGR Mean-Values
    
  # Kornia Canny
  kornia:
    low_threshold: 0.1    # Unterer Schwellwert (0-1)
    high_threshold: 0.2   # Oberer Schwellwert (0-1)
    kernel_size: 5        # Gauss-Kernel-Größe
    
  # BDCN Fallback (erweiterter Canny)
  bdcn_fallback:
    blur_kernel: 5        # Bilateral-Filter-Kernel
    canny_low: 50         # Canny unterer Schwellwert
    canny_high: 150       # Canny oberer Schwellwert
    morph_kernel: 3       # Morphologie-Kernel
    
  # Fixed CNN (Sobel)
  fixed_cnn:
    kernel: [[-1,0,1],[-2,0,2],[-1,0,1]]  # Sobel-X Kernel
```

#### Output-Einstellungen

```yaml
output:
  save_format: "png"    # png|jpg|tiff|webp - Ausgabeformat
  jpg_quality: 95       # JPEG-Qualität (1-100)
  png_compression: 6    # PNG-Kompression (0-9)
  preserve_structure: true   # Unterordner beibehalten
  skip_existing: false       # Existierende überspringen
```

#### Erweiterte Einstellungen

```yaml
# Logging
logging:
  level: "INFO"         # DEBUG|INFO|WARNING|ERROR
  file: "edge_detection.log"
  console: true         # Konsolen-Output
  
# Unterstützte Formate
supported_formats:
  - ".jpg"
  - ".jpeg"
  - ".png"
  - ".bmp"
  - ".tiff"
  - ".tif"
  - ".webp"
  - ".jp2"
```

### Umgebungsvariablen

```bash
# Alternative Config-Datei
export EDGE_CONFIG=/path/to/custom/config.yaml

# GPU-Device festlegen
export CUDA_VISIBLE_DEVICES=0,1  # Nutze GPU 0 und 1

# Thread-Limits
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

---

## Verwendung

### Basis-Verwendung

```bash
# Standard: Alle Bilder im 'images' Ordner verarbeiten
python run_edge_detectors.py --input_dir images --output_dir results

# Kurzform
python run_edge_detectors.py -i images -o results
```

### Erweiterte Optionen

#### Methoden-Auswahl

```bash
# Nur bestimmte Methoden verwenden
python run_edge_detectors.py -i images -o results --methods HED Kornia

# Alle außer einer Methode (via Config-Edit)
```

#### Performance-Optionen

```bash
# Sequentielle Verarbeitung (debugging/wenig RAM)
python run_edge_detectors.py -i images -o results --sequential

# Spezifische Worker-Anzahl
python run_edge_detectors.py -i images -o results --workers 8

# GPU explizit ein/aus
python run_edge_detectors.py -i images -o results --gpu on
python run_edge_detectors.py -i images -o results --gpu off
```

#### Batch-Optionen

```bash
# Existierende Dateien überspringen
python run_edge_detectors.py -i dataset -o output --skip-existing

# Mit Unterordnern (Standard wenn preserve_structure: true)
python run_edge_detectors.py -i photos -o edges

# Zusätzliche Formate
python run_edge_detectors.py -i raw -o processed --formats .cr2 .nef .arw
```

### Praktische Beispiele

#### 1. Große Foto-Sammlung verarbeiten

```bash
# Optimale Einstellungen für große Sammlungen
python run_edge_detectors.py \
    -i "D:/Fotos/2024" \
    -o "D:/Fotos/2024_edges" \
    --methods HED StructuredForests \
    --skip-existing \
    --workers 6
```

#### 2. Einzelnes Bild mit allen Methoden

```bash
# Temporären Ordner für einzelnes Bild erstellen
mkdir temp_image
copy "path/to/image.jpg" temp_image/
python run_edge_detectors.py -i temp_image -o temp_results
```

#### 3. Wissenschaftliche Auswertung

```bash
# Höchste Qualität, alle Methoden, TIFF-Output
# Erst config.yaml anpassen:
# output:
#   save_format: "tiff"
#   preserve_structure: true

python run_edge_detectors.py \
    -i microscopy_data \
    -o analysis_results \
    --sequential  # Für reproduzierbare Ergebnisse
```

#### 4. Real-Time Preview (experimentell)

```python
# preview.py - Live-Preview-Script
import cv2
from detectors import run_kornia, Config, MemoryManager

config = Config()
mem_mgr = MemoryManager(config)

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Speichere Frame temporär
    cv2.imwrite('temp.jpg', frame)
    
    # Edge Detection
    edges = run_kornia('temp.jpg', mem_mgr)
    
    # Anzeigen
    cv2.imshow('Original', frame)
    cv2.imshow('Edges', edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Edge Detection Methoden

### HED (Holistically-Nested Edge Detection)

**Beschreibung**: Deep Learning basierter Ansatz, der hierarchische Features nutzt.

**Stärken**:
- Exzellent für natürliche Bilder
- Erkennt Objektgrenzen sehr gut
- Robust gegen Texturen

**Schwächen**:
- Benötigt viel Speicher
- Langsamer als klassische Methoden
- Kann bei technischen Zeichnungen übersehen

**Optimale Anwendung**:
- Naturfotografie
- Porträts
- Komplexe Szenen

**Parameter-Tuning**:
```yaml
hed:
  crop_size: 500    # Kleinere Werte bei wenig GPU-RAM
  mean: [104, 117, 123]  # Alternative Mean-Values
```

### Structured Forests

**Beschreibung**: Random Forest basierter Ansatz mit strukturiertem Output.

**Stärken**:
- Gute Balance Geschwindigkeit/Qualität
- Trainiert auf Berkeley Dataset
- CPU-freundlich

**Schwächen**:
- Keine GPU-Beschleunigung
- Mittelmäßig bei sehr feinen Details

**Optimale Anwendung**:
- Allzweck-Kantenerkennung
- Wenn GPU nicht verfügbar
- Batch-Processing

### Kornia Canny

**Beschreibung**: GPU-optimierte Implementation des Canny-Algorithmus.

**Stärken**:
- Sehr schnell auf GPU
- Klassischer, bewährter Algorithmus
- Geringe False-Positive-Rate

**Schwächen**:
- Kann wichtige schwache Kanten verpassen
- Parameter-sensitiv

**Optimale Anwendung**:
- Technische Zeichnungen
- Klare Kanten
- Real-Time-Anwendungen

**Parameter-Tuning**:
```yaml
kornia:
  low_threshold: 0.05   # Sensibler
  high_threshold: 0.15  # Weniger Rauschen
  kernel_size: 7        # Glattere Kanten
```

### BDCN (Bi-Directional Cascade Network)

**Beschreibung**: State-of-the-Art Deep Learning mit bidirektionalem Ansatz.

**Stärken**:
- Beste Qualität
- Multi-Scale-Features
- Lernt Kantenhierarchien

**Schwächen**:
- Benötigt Git für Installation
- Sehr speicherintensiv
- Fallback oft ausreichend

**Optimale Anwendung**:
- Forschung
- Höchste Qualitätsansprüche
- Komplexe Texturen

### Fixed Edge CNN (Sobel)

**Beschreibung**: GPU-beschleunigter Sobel-Operator als CNN.

**Stärken**:
- Extrem schnell
- Minimaler Speicherbedarf
- Deterministisch

**Schwächen**:
- Nur Gradient-Magnitude
- Keine Kantenverdünnung
- Mehr Rauschen

**Optimale Anwendung**:
- Vorverarbeitung
- Wenn Geschwindigkeit kritisch
- Einfache Kantenerkennung

---

## Kommandozeilen-Referenz

### run_edge_detectors.py

```bash
usage: run_edge_detectors.py [-h] -i INPUT_DIR -o OUTPUT_DIR
                            [--methods {HED,StructuredForests,Kornia,BDCN,FixedEdgeCNN} ...]
                            [--sequential] [--skip-existing]
                            [--workers N] [--gpu {auto,on,off}]
                            [--config CONFIG] [--formats EXT ...]

Edge Detection Toolkit - Batch Processing

required arguments:
  -i INPUT_DIR, --input_dir INPUT_DIR
                        Eingabe-Verzeichnis mit Bildern
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Ausgabe-Verzeichnis für Ergebnisse

optional arguments:
  -h, --help            Hilfe anzeigen
  --methods {HED,StructuredForests,Kornia,BDCN,FixedEdgeCNN} ...
                        Nur bestimmte Methoden verwenden
  --sequential          Sequentielle statt parallele Verarbeitung
  --skip-existing       Überspringe bereits verarbeitete Bilder
  --workers N           Anzahl paralleler Prozesse (default: auto)
  --gpu {auto,on,off}   GPU-Nutzung (default: auto)
  --config CONFIG       Alternative Config-Datei
  --formats EXT ...     Zusätzliche Bildformate (z.B. .webp .jp2)
```

### detectors.py

```bash
usage: detectors.py [-h] [--init-models] [--verify] [--gpu-info]

Edge Detection Toolkit - Model Management

optional arguments:
  -h, --help     Hilfe anzeigen
  --init-models  Download und initialisiere alle Modelle
  --verify       Verifiziere installierte Modelle
  --gpu-info     Zeige GPU Informationen
```

---

## Performance-Optimierung

### GPU-Optimierung

#### Memory Management für verschiedene GPUs

**4GB VRAM (z.B. GTX 1650, Quadro T1000)**:
```yaml
system:
  gpu_memory_fraction: 0.7  # 70% = ~2.8GB
edge_detection:
  hed:
    crop_size: 400  # Kleinere Patches
```

**8GB VRAM (z.B. RTX 3060 Ti)**:
```yaml
system:
  gpu_memory_fraction: 0.85
edge_detection:
  hed:
    crop_size: 600
```

**16GB+ VRAM (z.B. RTX 3090)**:
```yaml
system:
  gpu_memory_fraction: 0.9
edge_detection:
  hed:
    crop_size: 800
```

#### Multi-GPU Setup

```bash
# Nutze GPU 0 und 1
export CUDA_VISIBLE_DEVICES=0,1

# Nur GPU 1 nutzen
export CUDA_VISIBLE_DEVICES=1
```

### CPU-Optimierung

#### Thread-Optimierung

```yaml
system:
  max_workers: 8  # Für 8-Core CPU
```

**Faustregel**: 
- `max_workers = CPU_Cores - 2` für Systeme mit GUI
- `max_workers = CPU_Cores` für Server

#### Memory-Optimierung

```bash
# Für 16GB RAM System
system:
  memory_limit_mb: 12288  # 12GB für Processing
  max_workers: 4          # Weniger parallele Prozesse
```

### Festplatten-Optimierung

#### SSD vs HDD

- **SSD**: Verwende mehr Worker (`max_workers: 8-16`)
- **HDD**: Reduziere Worker (`max_workers: 2-4`)

#### RAID/NAS

```yaml
# Für Netzwerkspeicher
system:
  max_workers: 2  # I/O-Engpass vermeiden
downloads:
  chunk_size: 65536  # Größere Chunks
```

### Batch-Größen-Optimierung

```python
# Optimale Batch-Größe berechnen
import psutil

ram_gb = psutil.virtual_memory().total / (1024**3)
cpu_cores = psutil.cpu_count()

# Empfohlene Worker
workers = min(cpu_cores - 2, int(ram_gb / 2))
print(f"Empfohlene Worker: {workers}")
```

---

## Troubleshooting

### Häufige Probleme

#### 1. CUDA/GPU nicht erkannt

**Symptom**: `Keine CUDA GPU gefunden`

**Lösungen**:
```bash
# 1. CUDA-Version prüfen
nvidia-smi

# 2. PyTorch mit CUDA neu installieren
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Environment-Variablen prüfen
echo %CUDA_PATH%  # Windows
echo $CUDA_PATH   # Linux
```

#### 2. Out of Memory Fehler

**Symptom**: `CUDA out of memory` oder `MemoryError`

**Lösungen**:
```yaml
# In config.yaml
system:
  gpu_memory_fraction: 0.5  # Reduzieren
  max_image_size: [2048, 2048]  # Kleinere Bilder
edge_detection:
  hed:
    crop_size: 300  # Kleinere Patches
```

#### 3. Download-Fehler

**Symptom**: `Download fehlgeschlagen`

**Lösungen**:
```yaml
# In config.yaml
downloads:
  timeout: 300  # Erhöhen
  max_retries: 5
  retry_delay: 10
```

```bash
# Manueller Download
wget https://example.com/model.caffemodel -O models/hed/hed.caffemodel
```

#### 4. Import-Fehler

**Symptom**: `ModuleNotFoundError`

**Lösungen**:
```bash
# Virtuelle Umgebung aktiviert?
which python  # Sollte venv/... zeigen

# Alle Dependencies installiert?
pip install -r requirements.txt --force-reinstall

# OpenCV-Probleme?
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

#### 5. Keine Bilder gefunden

**Symptom**: `Keine Bilder gefunden in ...`

**Lösungen**:
```bash
# Formate prüfen
python run_edge_detectors.py -i images -o results --formats .jpeg .JPG .JPEG

# Unterordner?
# Stelle sicher dass preserve_structure: true in config.yaml
```

### Performance-Probleme

#### Langsame Verarbeitung

1. **GPU-Nutzung prüfen**:
```bash
# Während der Verarbeitung
nvidia-smi -l 1  # Update jede Sekunde
```

2. **CPU-Bottleneck**:
```yaml
system:
  max_workers: 16  # Erhöhen wenn CPU nicht ausgelastet
```

3. **I/O-Bottleneck**:
- Auf SSD verschieben
- Weniger Worker bei HDD
- Output-Format optimieren (JPEG statt PNG)

#### Speicherlecks

**Symptome**: RAM-Nutzung steigt kontinuierlich

**Lösungen**:
```bash
# Sequentielle Verarbeitung nutzen
python run_edge_detectors.py -i images -o results --sequential

# Oder in kleineren Batches
python run_edge_detectors.py -i images_part1 -o results
python run_edge_detectors.py -i images_part2 -o results
```

### Plattform-spezifische Probleme

#### Windows

**Problem**: `WinError 5` bei pip
```bash
# Als Administrator ausführen oder:
python -m pip install --user -r requirements.txt
```

**Problem**: Lange Pfade
```bash
# In Registry aktivieren oder Git Bash nutzen
git config --system core.longpaths true
```

#### Linux

**Problem**: `libGL.so.1` nicht gefunden
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# CentOS/RHEL
sudo yum install mesa-libGL
```

#### macOS

**Problem**: `SSL: CERTIFICATE_VERIFY_FAILED`
```bash
# Zertifikate installieren
pip install --upgrade certifi
# Oder in Python Installation:
# /Applications/Python 3.x/Install Certificates.command
```

---

## Entwickler-Dokumentation

### Architektur

```
edge_detection_tool/
├── config.yaml          # Zentrale Konfiguration
├── detectors.py         # Edge Detection Implementierungen
├── run_edge_detectors.py # Batch Processing Engine
├── models/              # Vortrainierte Modelle
│   ├── hed/            # HED Caffe Modell
│   └── structured/     # Structured Forests
├── images/             # Input-Verzeichnis
├── results/            # Output-Verzeichnis
└── logs/               # Log-Dateien
```

### Erweiterung um neue Methoden

#### 1. Detector-Funktion hinzufügen

```python
# In detectors.py
def run_my_method(image_path: Union[str, Path], 
                  memory_mgr: Optional[MemoryManager] = None) -> np.ndarray:
    """Meine neue Edge Detection Methode"""
    
    # Bild laden
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")
    
    # Memory Management
    if memory_mgr:
        img, scale = memory_mgr.resize_if_needed(img)
    
    # Konfiguration laden
    cfg = config.config['edge_detection']['my_method'] if config else {}
    
    # Edge Detection
    # ... Ihre Implementierung ...
    
    return edges.astype('uint8')
```

#### 2. Methode registrieren

```python
# In run_edge_detectors.py
METHODS = [
    ('HED', run_hed),
    ('StructuredForests', run_structured),
    ('Kornia', run_kornia),
    ('BDCN', run_bdcn),
    ('FixedEdgeCNN', run_fixed),
    ('MyMethod', run_my_method),  # NEU
]
```

#### 3. Konfiguration erweitern

```yaml
# In config.yaml
edge_detection:
  my_method:
    param1: value1
    param2: value2
```

### API-Nutzung

#### Als Python-Modul

```python
from pathlib import Path
from detectors import Config, MemoryManager, run_hed

# Setup
config = Config()
memory_mgr = MemoryManager(config)

# Einzelbild verarbeiten
image_path = Path("test.jpg")
edges = run_hed(image_path, memory_mgr)

# Speichern
import cv2
cv2.imwrite("edges.png", edges)
```

#### Batch-Processing API

```python
from run_edge_detectors import BatchProcessor

# Processor erstellen
processor = BatchProcessor(
    input_dir=Path("input"),
    output_dir=Path("output")
)

# Custom Config
processor.config.config['system']['max_workers'] = 8

# Verarbeitung starten
processor.run(sequential=False)
```

### Testing

#### Unit Tests

```python
# test_detectors.py
import pytest
import numpy as np
from pathlib import Path
from detectors import run_fixed, Config, MemoryManager

def test_fixed_edge():
    config = Config()
    mem_mgr = MemoryManager(config)
    
    # Test-Bild erstellen
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    cv2.imwrite("test.jpg", test_img)
    
    # Edge Detection
    edges = run_fixed(Path("test.jpg"), mem_mgr)
    
    # Assertions
    assert edges.shape == (100, 100)
    assert edges.dtype == np.uint8
    assert 0 <= edges.min() <= edges.max() <= 255
```

#### Performance-Benchmarks

```python
# benchmark.py
import time
from pathlib import Path
from detectors import METHODS, Config, MemoryManager

config = Config()
mem_mgr = MemoryManager(config)
test_image = Path("benchmark.jpg")

for name, func in METHODS:
    start = time.time()
    result = func(test_image, mem_mgr)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.3f}s")
```

### Debugging

#### Verbose Mode

```python
# Debug-Logging aktivieren
import logging
logging.basicConfig(level=logging.DEBUG)

# Oder in config.yaml
logging:
  level: "DEBUG"
```

#### Memory-Profiling

```python
# memory_profile.py
import tracemalloc
tracemalloc.start()

# Verarbeitung
from run_edge_detectors import main
main()

# Stats
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

#### GPU-Profiling

```python
# gpu_profile.py
import torch.cuda

# Vor Verarbeitung
torch.cuda.reset_peak_memory_stats()

# Verarbeitung durchführen
# ...

# Stats ausgeben
print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

---

## FAQ

### Allgemeine Fragen

**F: Welche Methode soll ich verwenden?**
A: Für die meisten Anwendungen empfehlen wir HED (beste Qualität) oder Structured Forests (gute Balance). Für Echtzeitanwendungen nutzen Sie Kornia.

**F: Kann ich das Toolkit ohne GPU nutzen?**
A: Ja! Alle Methoden funktionieren auf CPU, sind aber langsamer. Structured Forests ist CPU-optimiert.

**F: Unterstützt das Toolkit Videos?**
A: Nicht direkt, aber Sie können Frames extrahieren und verarbeiten:
```bash
# Frames extrahieren
ffmpeg -i video.mp4 frames/frame_%04d.png

# Verarbeiten
python run_edge_detectors.py -i frames -o edges

# Video erstellen
ffmpeg -r 30 -i edges/HED/frame_%04d.png -c:v libx264 edges_video.mp4
```

**F: Wie kann ich die Ausgabequalität verbessern?**
A: 
1. Höhere Eingangsauflösung verwenden
2. Output-Format auf TIFF oder PNG setzen
3. Parameter der jeweiligen Methode anpassen
4. Mehrere Methoden kombinieren

### Technische Fragen

**F: Warum ist meine GPU-Auslastung niedrig?**
A: Mögliche Gründe:
- CPU-Bottleneck (erhöhen Sie `max_workers`)
- Kleine Bilder (GPU arbeitet ineffizient)
- I/O-Bottleneck (SSD verwenden)

**F: Kann ich Remote-Processing nutzen?**
A: Ja, über SSH:
```bash
ssh user@server "cd /path/to/tool && python run_edge_detectors.py -i remote_images -o remote_results"
```

**F: Wie erstelle ich einen Docker-Container?**
A: Siehe Docker-Installation oben oder nutzen Sie:
```bash
docker build -t edge-detection .
docker run -v /local/images:/app/images edge-detection -i images -o results
```

### Lizenz-Fragen

**F: Kann ich das Toolkit kommerziell nutzen?**
A: Ja, unter MIT-Lizenz. Beachten Sie aber Lizenzen der Modelle (HED, BDCN).

**F: Muss ich Attributierung angeben?**
A: Nicht erforderlich, aber geschätzt. Die einzelnen Modelle haben eigene Anforderungen.

---

## Changelog

### Version 2.0.0 (2024)
- ✨ Komplette Überarbeitung mit GPU-Support
- ✨ Multiprocessing implementiert
- ✨ Config-System hinzugefügt
- ✨ Progress-Bars und robuste Downloads
- ✨ Memory-Management
- ✨ Erweiterte Bildformat-Unterstützung
- 🐛 Viele Bugfixes

### Version 1.0.0 (Original)
- Basis-Implementation
- 5 Edge-Detection-Methoden
- Einfache Batch-Verarbeitung

---

## Lizenz

MIT License

Copyright (c) 2024 Edge Detection Toolkit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Credits & Referenzen

### Modelle & Algorithmen

- **HED**: S. Xie and Z. Tu, "Holistically-Nested Edge Detection", ICCV 2015
- **Structured Forests**: P. Dollár and C. L. Zitnick, "Structured Forests for Fast Edge Detection", ICCV 2013
- **BDCN**: J. He et al., "Bi-Directional Cascade Network for Perceptual Edge Detection", CVPR 2019
- **Kornia**: E. Riba et al., "Kornia: an Open Source Differentiable Computer Vision Library", WACV 2020

### Entwicklung

Entwickelt mit ❤️ für die Computer Vision Community.

Besonderer Dank an:
- OpenCV Community
- PyTorch Team
- Alle Contributor der genutzten Open-Source-Projekte

---

## Kontakt & Support

- **Issues**: GitHub Issues (wenn Repository vorhanden)
- **Dokumentation**: Diese README.md
- **Updates**: Check GitHub für neueste Version

Für kommerzielle Support-Anfragen oder Custom-Entwicklung kontaktieren Sie uns über GitHub.

---

*Letzte Aktualisierung: 2024*
