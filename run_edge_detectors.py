#!/usr/bin/env python3
"""
Edge Detection Toolkit - Batch Processing
Unterstützt parallele Verarbeitung, Unterordner, erweiterte Formate
Mit Progress-Tracking und Memory-Management
"""

import argparse
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch
from PIL import Image
from tqdm import tqdm

# Detectors importieren
# fmt: off
# isort: off
from detectors import (
    Config,
    MemoryManager,
    run_bdcn,
    run_fixed,
    run_hed,
    run_kornia,
    run_structured,
)
# isort: on

# fmt: on

# Warnungen für Pillow unterdrücken
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# ------------------------------------------------------
# Edge Detection Methods Registry
# ------------------------------------------------------

METHODS = [
    ("HED", run_hed),
    ("StructuredForests", run_structured),
    ("Kornia", run_kornia),
    ("BDCN", run_bdcn),
    ("FixedEdgeCNN", run_fixed),
]

# ------------------------------------------------------
# Image Processing
# ------------------------------------------------------


class ImageProcessor:
    """Verarbeitet einzelne Bilder mit allen Edge-Detection-Methoden"""

    def __init__(self, config: Config, memory_mgr: MemoryManager):
        self.config = config
        self.memory_mgr = memory_mgr
        self.output_config = config.config["output"]

    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Lade Bild mit erweitertem Format-Support"""
        try:
            # Versuche mit OpenCV
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img

            # Fallback auf Pillow für exotische Formate
            pil_img = Image.open(image_path)

            # Konvertiere zu RGB wenn nötig
            if pil_img.mode not in ["RGB", "L"]:
                if pil_img.mode == "RGBA":
                    # Handle Alpha Channel
                    background = Image.new("RGB", pil_img.size, (255, 255, 255))
                    background.paste(pil_img, mask=pil_img.split()[3])
                    pil_img = background
                else:
                    pil_img = pil_img.convert("RGB")

            # Zu numpy array
            img_array = np.array(pil_img)

            # BGR für OpenCV (wenn RGB)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            return img_array

        except Exception as e:
            print(f"[error] Kann Bild nicht laden {image_path}: {e}")
            return None

    def save_image(self, image: np.ndarray, output_path: Path) -> bool:
        """Speichere Bild mit konfigurierbarem Format"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Format aus Config oder behalte Original
            fmt = self.output_config["save_format"].lower()
            if fmt != output_path.suffix[1:].lower():
                output_path = output_path.with_suffix(f".{fmt}")

            # Format-spezifische Optionen
            if fmt in ["jpg", "jpeg"]:
                quality = self.output_config.get("jpg_quality", 95)
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif fmt == "png":
                compression = self.output_config.get("png_compression", 6)
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            elif fmt == "webp":
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_WEBP_QUALITY, 95])
            elif fmt in ["tiff", "tif"]:
                # Pillow für TIFF
                Image.fromarray(image).save(str(output_path), compression="tiff_lzw")
            else:
                # Standard
                cv2.imwrite(str(output_path), image)

            return True

        except Exception as e:
            print(f"[error] Kann Bild nicht speichern {output_path}: {e}")
            return False

    def process_single_method(
        self, image_path: Path, method_name: str, method_func, output_base: Path
    ) -> Tuple[str, bool, str]:
        """Verarbeite ein Bild mit einer Methode"""
        try:
            # Output-Pfad bestimmen
            if self.output_config["preserve_structure"]:
                # Behalte komplette Unterordner-Struktur
                rel_path = image_path.relative_to(self.input_dir)
                output_path = output_base / method_name / rel_path
            else:
                output_path = output_base / method_name / image_path.name

            # Skip wenn bereits existiert
            if self.output_config["skip_existing"] and output_path.exists():
                return (method_name, True, "skipped")

            # Edge Detection ausführen
            edges = method_func(image_path, self.memory_mgr)

            # Speichern
            success = self.save_image(edges, output_path)

            return (method_name, success, "processed" if success else "failed")

        except Exception as e:
            return (method_name, False, f"error: {str(e)}")


# ------------------------------------------------------
# Batch Processing
# ------------------------------------------------------


def process_image_batch(args: Tuple[Path, Path, Dict, List[Tuple[str, Callable]]]) -> Dict[str, str]:
    """Verarbeite ein Bild mit allen Methoden (für Multiprocessing)"""
    image_path, output_dir, config_dict, methods = args

    # Rekonstruiere Config-Objekt
    config = Config()
    config.config = config_dict

    # Memory Manager
    memory_mgr = MemoryManager(config)

    # Processor
    processor = ImageProcessor(config, memory_mgr)

    results = {}

    for method_name, method_func in methods:
        _, success, status = processor.process_single_method(image_path, method_name, method_func, output_dir)
        results[method_name] = status

    return results


class BatchProcessor:
    """Hauptklasse für Batch-Verarbeitung mit Multiprocessing"""

    def __init__(self, input_dir: Path, output_dir: Path, methods: Optional[List[Tuple[str, Callable]]] = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = Config()
        self.methods = methods or METHODS

        # Unterstützte Formate aus Config
        self.supported_formats = self.config.config.get(
            "supported_formats", [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]
        )

        # Worker-Anzahl
        self.max_workers = self.config.config["system"]["max_workers"]
        if self.max_workers <= 0:
            self.max_workers = mp.cpu_count()

    def find_images(self) -> List[Path]:
        """Finde alle Bilder (inkl. Unterordner wenn aktiviert)"""
        images = []

        if self.config.config["output"]["preserve_structure"]:
            # Rekursiv mit Unterordnern
            for fmt in self.supported_formats:
                images.extend(self.input_dir.rglob(f"*{fmt}"))
                images.extend(self.input_dir.rglob(f"*{fmt.upper()}"))
        else:
            # Nur Top-Level
            for fmt in self.supported_formats:
                images.extend(self.input_dir.glob(f"*{fmt}"))
                images.extend(self.input_dir.glob(f"*{fmt.upper()}"))

        return sorted(set(images))  # Duplikate entfernen und sortieren

    def create_output_structure(self):
        """Erstelle Output-Verzeichnisstruktur"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for method_name, _ in self.methods:
            method_dir = self.output_dir / method_name
            method_dir.mkdir(exist_ok=True)

    def process_sequential(self, images: List[Path]):
        """Sequentielle Verarbeitung (für Debugging)"""
        print("[info] Verwende sequentielle Verarbeitung")

        memory_mgr = MemoryManager(self.config)
        processor = ImageProcessor(self.config, memory_mgr)

        # Progress Bar für alle Bilder
        total_ops = len(images) * len(self.methods)
        with tqdm(total=total_ops, desc="Verarbeitung") as pbar:
            for image_path in images:
                pbar.set_postfix({"file": image_path.name})

                for method_name, method_func in self.methods:
                    _, success, status = processor.process_single_method(
                        image_path, method_name, method_func, self.output_dir
                    )

                    if not success and "error" in status:
                        tqdm.write(f"[error] {image_path.name} - {method_name}: {status}")

                    pbar.update(1)

    def process_parallel(self, images: List[Path]):
        """Parallele Verarbeitung mit Multiprocessing"""
        print(f"[info] Verwende parallele Verarbeitung mit {self.max_workers} Prozessen")

        # Vorbereite Argumente
        args_list = [(img, self.output_dir, self.config.config, self.methods) for img in images]

        # Progress Bars
        total_images = len(images)
        completed = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Starte alle Jobs
            future_to_image = {executor.submit(process_image_batch, args): args[0] for args in args_list}

            # Progress Bar
            with tqdm(total=total_images, desc="Bilder verarbeitet") as pbar:
                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]

                    try:
                        results = future.result()

                        # Zeige Fehler
                        for method, status in results.items():
                            if "error" in status:
                                tqdm.write(f"[error] {image_path.name} - {method}: {status}")

                    except Exception as e:
                        tqdm.write(f"[error] {image_path.name}: {e}")

                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix({"current": image_path.name, "memory": f"{psutil.virtual_memory().percent:.0f}%"})

    def show_summary(self, images: List[Path]):
        """Zeige Verarbeitungs-Zusammenfassung"""
        print("\n" + "=" * 60)
        print("VERARBEITUNGS-ZUSAMMENFASSUNG")
        print("=" * 60)
        print(f"Input-Verzeichnis:  {self.input_dir}")
        print(f"Output-Verzeichnis: {self.output_dir}")
        print(f"Gefundene Bilder:   {len(images)}")
        print(f"Edge-Methoden:      {len(self.methods)}")
        print(f"Gesamt-Operationen: {len(images) * len(self.methods)}")
        print(f"CPU-Kerne:          {mp.cpu_count()} (verwendet: {self.max_workers})")

        if self.config.use_cuda:
            print(f"GPU:                {torch.cuda.get_device_name(0)}")
        else:
            print("GPU:                Nicht verfügbar/deaktiviert")

        print("=" * 60 + "\n")

    def run(self, sequential: bool = False):
        """Hauptverarbeitungs-Funktion"""
        # Finde Bilder
        images = self.find_images()

        if not images:
            print(f"[warning] Keine Bilder gefunden in {self.input_dir}")
            print(f"Unterstützte Formate: {', '.join(self.supported_formats)}")
            return

        # Zeige Zusammenfassung
        self.show_summary(images)

        # Erstelle Output-Struktur
        self.create_output_structure()

        # Verarbeitung
        if sequential or self.max_workers == 1:
            self.process_sequential(images)
        else:
            self.process_parallel(images)

        print("\n[success] Verarbeitung abgeschlossen!")

        # Statistiken
        total_output = sum(1 for _ in self.output_dir.rglob("*") if _.is_file())
        print(f"[info] {total_output} Dateien erstellt in {self.output_dir}")


# ------------------------------------------------------
# CLI Interface
# ------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Edge Detection Toolkit - Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  %(prog)s --input_dir images --output_dir results
  %(prog)s -i photos -o edges --sequential
  %(prog)s -i . -o ./output --methods HED Kornia
  %(prog)s -i dataset -o processed --skip-existing
        """,
    )

    # Pflicht-Argumente
    parser.add_argument("-i", "--input_dir", required=True, type=Path, help="Eingabe-Verzeichnis mit Bildern")
    parser.add_argument("-o", "--output_dir", required=True, type=Path, help="Ausgabe-Verzeichnis für Ergebnisse")

    # Optionale Argumente
    parser.add_argument(
        "--methods", nargs="+", choices=[m[0] for m in METHODS], help="Nur bestimmte Methoden verwenden"
    )
    parser.add_argument("--sequential", action="store_true", help="Sequentielle statt parallele Verarbeitung")
    parser.add_argument("--skip-existing", action="store_true", help="Überspringe bereits verarbeitete Bilder")
    parser.add_argument("--workers", type=int, metavar="N", help="Anzahl paralleler Prozesse (default: auto)")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="GPU-Nutzung (default: auto)")
    parser.add_argument("--config", type=Path, help="Alternative Config-Datei")
    parser.add_argument("--formats", nargs="+", metavar="EXT", help="Zusätzliche Bildformate (z.B. .webp .jp2)")

    args = parser.parse_args()

    # Validierung
    if not args.input_dir.exists():
        print(f"[error] Input-Verzeichnis existiert nicht: {args.input_dir}")
        sys.exit(1)

    # Config überschreiben wenn nötig
    if args.config and args.config.exists():
        os.environ["EDGE_CONFIG"] = str(args.config)

    # Temporäre Config-Anpassungen
    temp_config = {}

    if args.skip_existing:
        temp_config.setdefault("output", {})["skip_existing"] = True

    if args.workers:
        temp_config.setdefault("system", {})["max_workers"] = args.workers

    if args.gpu != "auto":
        temp_config.setdefault("system", {})["use_gpu"] = args.gpu == "on"

    if args.formats:
        temp_config["supported_formats"] = [f if f.startswith(".") else f".{f}" for f in args.formats]

    # Wenn Methoden-Filter aktiv
    if args.methods:
        # Filter METHODS basierend auf Auswahl
        filtered_methods = [(name, func) for name, func in METHODS if name in args.methods]
    else:
        filtered_methods = METHODS

    # Processor erstellen und ausführen
    processor = BatchProcessor(args.input_dir, args.output_dir, filtered_methods)

    # Temporäre Config anwenden
    if temp_config:
        processor.config.config.update(temp_config)

    # Verarbeitung starten
    try:
        processor.run(sequential=args.sequential)
    except KeyboardInterrupt:
        print("\n[interrupted] Verarbeitung abgebrochen")
        sys.exit(1)
    except Exception as e:
        print(f"\n[error] Unerwarteter Fehler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Multiprocessing-Kompatibilität für Windows
    if sys.platform.startswith("win"):
        mp.freeze_support()

    main()
