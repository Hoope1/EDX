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

## Systemanforderungen

### Minimum
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8+
- **RAM**: 8 GB
- **CPU**: Quad-Core
- **GPU**: Optional (NVIDIA mit CUDA-Support)

### Empfohlen
- **RAM**: 16 GB
- **CPU**: 6+ Cores
- **GPU**: NVIDIA mit 4+ GB VRAM (z.B. GTX 1650, Quadro T1000)

## Installation

### Windows Schnellstart

```bash
# 1. Projekt erstellen
python create.py

# 2. In Ordner wechseln
cd edge_detection_tool

# 3. Setup ausführen
run.bat
```

### Manuelle Installation

```bash
# 1. Projekt erstellen
python create.py

# 2. Virtuelle Umgebung
cd edge_detection_tool
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Dependencies installieren
pip install -r requirements.txt

# 4. Modelle herunterladen
python detectors.py --init-models
```

### GPU-Support (Optional)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Verwendung

### Basis-Verwendung

```bash
# Alle Bilder im 'images' Ordner verarbeiten
python run_edge_detectors.py --input_dir images --output_dir results
```

### Erweiterte Optionen

```bash
# Nur bestimmte Methoden
python run_edge_detectors.py -i images -o results --methods HED Kornia

# Sequentielle Verarbeitung (weniger RAM-Nutzung)
python run_edge_detectors.py -i images -o results --sequential

# Existierende Dateien überspringen
python run_edge_detectors.py -i images -o results --skip-existing

# Anzahl paralleler Prozesse festlegen
python run_edge_detectors.py -i images -o results --workers 8

# GPU explizit steuern
python run_edge_detectors.py -i images -o results --gpu off
```

## Konfiguration (config.yaml)

### Wichtige Einstellungen

```yaml
system:
  use_gpu: auto              # auto|true|false
  gpu_memory_fraction: 0.8   # GPU-Speicher-Limit (0.1-1.0)
  max_workers: 4             # Parallele Prozesse (0=auto)
  max_image_size: [4096, 4096]  # Max Bildgröße [Breite, Höhe]
  memory_limit_mb: 8192      # RAM-Limit in MB

output:
  save_format: "png"         # png|jpg|tiff|webp
  jpg_quality: 95           # JPEG-Qualität (1-100)
  preserve_structure: true   # Unterordner beibehalten
  skip_existing: false      # Existierende überspringen
```

### GPU-Memory-Einstellungen

**Für 4GB VRAM (z.B. Quadro T1000):**
```yaml
system:
  gpu_memory_fraction: 0.7
edge_detection:
  hed:
    crop_size: 400
```

**Für 8GB+ VRAM:**
```yaml
system:
  gpu_memory_fraction: 0.85
edge_detection:
  hed:
    crop_size: 600
```

## Edge Detection Methoden

### HED (Holistically-Nested Edge Detection)
- **Beste für**: Natürliche Bilder, Objektgrenzen
- **Vorteile**: Höchste Qualität, Deep Learning
- **Nachteile**: Speicherintensiv, langsamer

### Structured Forests
- **Beste für**: Allzweck, CPU-Processing
- **Vorteile**: Gute Balance Geschwindigkeit/Qualität
- **Nachteile**: Keine GPU-Beschleunigung

### Kornia Canny
- **Beste für**: Technische Zeichnungen, klare Kanten
- **Vorteile**: GPU-beschleunigt, sehr schnell
- **Nachteile**: Kann schwache Kanten verpassen

### BDCN (Fallback)
- **Beste für**: Wenn Original-BDCN nicht verfügbar
- **Vorteile**: Erweiterte Canny-Implementation
- **Nachteile**: Nicht die volle BDCN-Qualität

### Fixed Edge CNN (Sobel)
- **Beste für**: Schnelle Vorverarbeitung
- **Vorteile**: Extrem schnell, minimal Speicher
- **Nachteile**: Einfachste Methode

## Troubleshooting

### GPU nicht erkannt

```bash
# CUDA-Version prüfen
nvidia-smi

# PyTorch mit CUDA neu installieren
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

In `config.yaml` anpassen:
```yaml
system:
  gpu_memory_fraction: 0.5      # Reduzieren
  max_image_size: [2048, 2048]  # Kleinere Bilder
edge_detection:
  hed:
    crop_size: 300              # Kleinere Patches
```

### Download-Fehler

In `config.yaml`:
```yaml
downloads:
  timeout: 300      # Timeout erhöhen
  max_retries: 5    # Mehr Versuche
```

### Keine Bilder gefunden

```bash
# Formate explizit angeben
python run_edge_detectors.py -i images -o results --formats .jpg .JPG .jpeg .JPEG
```

### Windows-spezifisch

**"WinError 5" bei pip:**
```bash
# Als User installieren
python -m pip install --user -r requirements.txt
```

### Linux-spezifisch

**"libGL.so.1 nicht gefunden":**
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# CentOS/RHEL
sudo yum install mesa-libGL
```

## Kommandozeilen-Referenz

### run_edge_detectors.py

```
-i, --input_dir      Input-Verzeichnis (erforderlich)
-o, --output_dir     Output-Verzeichnis (erforderlich)
--methods            Methoden auswählen [HED, StructuredForests, Kornia, BDCN, FixedEdgeCNN]
--sequential         Sequentielle statt parallele Verarbeitung
--skip-existing      Existierende Dateien überspringen
--workers N          Anzahl Worker (default: auto)
--gpu {auto,on,off}  GPU-Nutzung (default: auto)
--formats            Zusätzliche Bildformate (z.B. --formats .webp .tiff)
```

### detectors.py

```
--init-models    Modelle herunterladen
--verify         Installation verifizieren  
--gpu-info       GPU-Informationen anzeigen
```

## Performance-Tipps

### CPU-Optimierung
- `max_workers` = CPU-Kerne - 2 (für Desktop)
- `max_workers` = CPU-Kerne (für Server)

### GPU-Optimierung
- Bei 4GB VRAM: `gpu_memory_fraction: 0.7`
- Bei OOM-Errors: `crop_size` reduzieren

### Speicher-Optimierung
- Bei 16GB RAM: `memory_limit_mb: 8192`
- Große Bilder: `max_image_size` reduzieren
- Wenig RAM: `--sequential` nutzen

## Projekt-Struktur

```
edge_detection_tool/
├── config.yaml              # Konfiguration
├── detectors.py            # Edge Detection Algorithmen
├── run_edge_detectors.py   # Batch-Processing
├── requirements.txt        # Python-Dependencies
├── run.bat                 # Windows-Starter
├── models/                 # Vortrainierte Modelle
│   ├── hed/               # HED-Modell
│   └── structured/        # Structured Forests
├── images/                 # Input-Bilder
└── results/               # Output-Bilder
    ├── HED/
    ├── StructuredForests/
    ├── Kornia/
    ├── BDCN/
    └── FixedEdgeCNN/
```

## Beispiel-Workflows

### Große Foto-Sammlung
```bash
python run_edge_detectors.py \
    -i "D:/Fotos/2024" \
    -o "D:/Fotos/2024_edges" \
    --methods HED StructuredForests \
    --skip-existing \
    --workers 6
```

### Wissenschaftliche Auswertung
```yaml
# In config.yaml:
output:
  save_format: "tiff"
  
# Dann:
python run_edge_detectors.py -i data -o analysis --sequential
```

### Schnelle Vorschau
```bash
python run_edge_detectors.py \
    -i test_images \
    -o quick_results \
    --methods Kornia FixedEdgeCNN \
    --workers 8
```

## FAQ

**Welche Methode soll ich nutzen?**
- Beste Qualität: HED
- Beste Balance: Structured Forests  
- Schnellste: Kornia oder Fixed CNN

**Kann ich ohne GPU arbeiten?**
Ja, alle Methoden funktionieren auf CPU. Structured Forests ist CPU-optimiert.

**Wie verbessere ich die Qualität?**
1. Höhere Input-Auflösung
2. PNG oder TIFF als Output
3. HED-Methode verwenden
4. Parameter in config.yaml anpassen

**Warum ist die GPU-Auslastung niedrig?**
- Erhöhen Sie `max_workers`
- Prüfen Sie CPU-Bottleneck
- Nutzen Sie SSD statt HDD

## Lizenz

MIT License - Siehe LICENSE Datei

Die verwendeten Modelle (HED, Structured Forests) haben eigene Lizenzen.

---

*Entwickelt für die Computer Vision Community*
