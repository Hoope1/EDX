## Inhalt

1. [Projekt­struktur für Codex](#projektstruktur-für-codex)
2. [Coding-Konventionen](#coding-konventionen)
3. [Edge-Detection-API & Module](#edge-detection-api--module)
4. [Konfiguration `config.yaml`](#konfiguration-configyaml)
5. [Tests & CI-Checks](#tests--ci-checks)
6. [CLI / Script-Nutzung](#cli--script-nutzung)
7. [Performance-Richtlinien](#performance-richtlinien)
8. [Pull-Request-Checkliste](#pull-request-checkliste)
9. [Agent Playbook (Quick Steps)](#agent-playbook-quick-steps)

---

## Projekt­struktur für Codex

| Pfad                         | Zweck für den AI-Agent                                               |
|------------------------------|----------------------------------------------------------------------|
| `/src/`                      | Haupt-Python-Package **`edge_detection_tool`**                      |
| &nbsp;&nbsp;`__init__.py`    | Exportiert Package-API                                               |
| &nbsp;&nbsp;`detectors.py`   | **Implementierung aller Edge-Algorithmen** (siehe [3](#edge-detection-api--module)) |
| &nbsp;&nbsp;`run_edge_detectors.py` | **Batch-Engine & CLI-Entry-Point**                           |
| &nbsp;&nbsp;`memory.py`      | Memory-Management-Hilfen                                             |
| &nbsp;&nbsp;`config.py`      | YAML-Loader & Defaults                                               |
| `/models/`                   | Vorge­trainierte Netz­werke (wird von `detectors.py` geladen)        |
| `/tests/`                    | PyTest-Suite (Unit + Smoke)                                          |
| `/examples/`                 | Mini-Notebooks & Snippets                                           |
| `/docker/`                   | Dockerfile + Runtime-Helper                                          |
| `config.yaml`                | **Zentrale Konfiguration** (darf von Codex gelesen / editiert werden) |
| `requirements.txt`           | Pip-Dependencies (GPU & CPU-Optionen)                                |

> **Codex‐Hinweis:** Es existiert **kein** UI-Code.  
> Änderungen an `/public` oder anderen statischen Path ≈ ❌ nicht zulässig.

---

## Coding-Konventionen

| Thema                              | Vorgabe für Codex                                                                              |
|------------------------------------|------------------------------------------------------------------------------------------------|
| **Sprache**                        | Python ≥ 3.10, strikte Typannotationen (`from __future__ import annotations`)                  |
| **Formatierung**                   | `black --line-length 120` + `isort`                                                            |
| **Linting**                        | `ruff --select ALL --ignore ANN101`                                                            |
| **Logging**                        | Verwende `logging.getLogger(__name__)`; **kein** `print` in Bibliotheks­code                   |
| **GPU-Handling**                   | Prüfe `torch.cuda.is_available()`; fallback auf CPU ohne Abbruch                               |
| **Dateinamen**                     | `snake_case.py`; Functions `snake_case`; Klassen `PascalCase`                                  |
| **Dokstrings**                     | Google-Style (1-Zeiler + ausführliche Beschreibung, Args, Returns, Raises)                    |
| **Neue Abhängigkeiten**            | Müssen zuerst in `requirements.txt` (CPU + CUDA Pin) aufgenommen und in PR-Beschreibung erklärt werden |

---

## Edge Detection API & Module

### 3.1 Namen & Registrierung

Jede Methode hat:

```python
# detectors.py
METHODS: list[tuple[str, Callable[[Path, MemoryManager], np.ndarray]]]

1. Funktions-Prefix run_<method>()


2. Name-String exakt wie CLI-Flag (HED, Kornia, usw.)


3. Registrierung am Ende der Datei in METHODS



> Codex‐Pflicht: Neue Algorithmen immer hier registrieren und in config.yaml eintragen.



3.2 Aktuelle Implementierungen

Name (Flag)	Datei / Funktion	GPU-Pfad	Besonderheiten

HED	run_hed	✅	Benötigt Caffe-Model im /models/hed/
Structured Forests	run_structured	❌	CPU-optimiert, keine CUDA
Kornia Canny	run_kornia	✅	PyTorch, batch-fähig
BDCN	run_bdcn	✅	Model-Download via Git LFS
Fixed Edge CNN Sobel	run_fixed	✅	Deterministisch, Low Memory



---

Konfiguration config.yaml

Wird beim Import von Config() geladen.

Änderungen gelten prozess­weit (Singleton).

Codex darf Felder hinzufügen oder existente ändern, niemals löschen.


system:
  use_gpu: auto            # auto|true|false
  max_workers: 0           # 0 = auto (CPU-Cores-2)
  gpu_memory_fraction: 0.8 # 0–1
edge_detection:
  hed:
    crop_size: 500
output:
  save_format: png
logging:
  level: INFO

> Codex‐Hinweis: Immer yaml.safe_load verwenden; boolean / ints casten.




---

Tests & CI-Checks

Befehl	Zweck	Muss grün sein

pytest -q	Unit- & Smoke-Tests	✅
ruff check .	Statisches Linting	✅
black --check .	Formatierung	✅
mypy src/	Typ­prüfung	✅ (keine error-Level)
python detectors.py --verify	MD5-Checksum der Modelle	✅
python detectors.py --gpu-info	Darf nicht crashen, auch ohne GPU	✅


CI-Workflow befindet sich in .github/workflows/ci.yml.
Codex soll bei neuen Features entsprechende Tests in /tests/ ergänzen.


---

CLI / Script-Nutzung

Script / Entry Point	Hauptaufgabe	Wichtig für Codex-Änderungen

run_edge_detectors.py	Stapelverarbeitung (Batch)	Flags synchron halten mit README.md
detectors.py --init-models	Modelle laden (Download)	Checksums stets aktualisieren
create.py (root)	Projekt-Gerüst / Boilerplate (Setup)	Nur Windows Auto-Installer


Flag-Parsing via argparse.

Neue Flags müssen in --help erscheinen und in der Dokumentation (README.md) ergänzt werden.



---

Performance-Richtlinien

Szenario	Codex-Empfehlung

VRAM < 4 GB	system.gpu_memory_fraction ≤ 0.6, edge_detection.hed.crop_size ≤ 400
Headless Server (nur CPU)	--sequential oder max_workers = CPU_CORES; StructuredForests bevorzugen
SSD I/O-Engpass	Worker-Zahl ≤ 8; Output-Format JPEG 80 Qualität
Memory-Leak Debugging	Verwende MemoryManager.debug=True und tracemalloc; keine globalen Tensor-Referenzen speichern
Multi-GPU (Forschung)	Setze CUDA_VISIBLE_DEVICES und verteile Batches manuell; Codex soll kein automatisches DDP einbauen



---

Pull-Request-Checkliste

1. Beschreibung: Kurz + präzise (Was | Warum).


2. Issue-Verlinkung: Closes #⟨ID⟩ falls vorhanden.


3. Tests: Neu oder angepasst, alle CI-Checks bestehen.


4. Screenshots / Logs: Bei CLI-Flag-Erweiterungen beispielhafte Ausgabe.


5. Docs: README.md + ggf. config.yaml kommentiert.


6. Kompaktheit: Kein Funktions-Mix; Single Responsibility PR.



> Codex‐Reminder: Automatisch generierte PR-Titel im Imperativ, ≤ 60 Zeichen.




---

Agent Playbook (Quick Steps)

1. Scanne Repo → Lese AGENTS.md vollständig.


2. Lade Config via Config(), prüfe use_gpu.


3. Wähle Methode (METHODS-Liste) oder registriere neue → Tests anlegen.


4. Nutze Memory-Manager für alle Bilder ≥ system.max_image_size.


5. Erstelle Batch CLI Flag parallel zu Funktions-Namen, halte Help-Text aktuell.


6. Führe lokale Checks (Tabelle oben) aus, anschließend PR mit Checkliste öffnen.




---

Ende der AGENTS.md – Stand 2024-12-01



