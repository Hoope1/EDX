AGENTS.md – Edge Detection Toolkit


---

lang: python version: "1.0"

1  Projektübersicht

Ein GPU‑beschleunigtes Toolkit zur Batch‑Kantenerkennung mit fünf Algorithmen (HED, Structured Forests, Kornia Canny, BDCN‑Fallback, Fixed Edge CNN), Multiprocessing‑Support und intelligenter Speicherverwaltung. Entwickelt für Forschung und Produktions‑Pipelines.

Primäres Ziel  Schnelle, reproduzierbare Kantenmasken hoher Qualität für große Bildmengen.

Haupt­technologien  Python 3.8+, PyTorch 2.x, Kornia, OpenCV.

Supported OS  Win 10/11, Ubuntu 18.04+, macOS 10.14+ (CPU‑Pfad), CUDA 11.8/12.1 (GPU‑Pfad).



---

2  Verzeichnisstruktur

edge_detection_tool/
├── config.yaml              # Zentrale Konfiguration
├── detectors.py            # Edge‑Algorithmen & Modell‑Loader
├── run_edge_detectors.py   # Batch‑Driver‑Script
├── requirements.txt        # Abhängigkeiten
├── scripts/                # Hilfs‑ & CI‑Skripte
│   └── checks.sh           # ‚Alles‑grün‘‑Check (siehe §6)
├── models/                 # Geklonte/geladene Gewichte (nicht versionieren!)
├── images/                 # Eingabebilder (ignored)
└── results/                # Ausgabemasken (ignored)

Gültigkeitsbereich dieser Datei

Alle Dateien und Unterordner im Repo‑Root. Untergeordnete AGENTS.md überschreiben hier definierte Regeln (§9).


---

3  Coding‑Standards

Kategorie	Vorgabe

Sprache	Python ≥ 3.8, UTF‑8
Formatierung	black (Line‑Length 88), isort –‑profile black
Linting	flake8 (Plugins: flake8‑bugbear, flake8‑annotations)
Static Typing	mypy --strict für alle Module außer scripts/
Docstrings	Google‑Style, deutsch oder englisch konsistent
Imports	Absolute Pfade; keine Stern‑Imports
Fehler­behandlung	Exceptions statt Rückgabe‑Codes, spezifische Exception‑Klassen
CUDA‑Pfad	JIT‑frei (kein torch.cuda.amp.autocast() ohne Bedarf)


Zusätzliche Details stehen in pyproject.toml und werden von CI überprüft.


---

4  Abhängigkeiten & Umgebung

1. Pflicht‑Pakete  laut requirements.txt. Version‑Pins nicht entfernen.


2. GPU‐Support  Erkenntnis via torch.cuda.is_available(). Fallback → CPU.


3. Plattformkompatibilität  Windows‑Pfadtrennzeichen mit pathlib abstrahieren.




---

5  Tests

# Schnelllauf
pytest -q

# Voller Lauf mit Coverage≥90 %
coverage run -m pytest
coverage report --fail-under=90

Tests liegen in tests/ (Mirror der Modulstruktur).

Jedes neue Feature benötigt ≥ 2 Testfälle (Positiv + Negativ/OOM/Pfad ohne GPU).



---

6  Programmatic Checks (CI ‚Alles‑grün‘)

./scripts/checks.sh muss erfolgreich sein, bevor ein PR gemergt wird:

#!/usr/bin/env bash
set -euo pipefail
black --check edge_detection_tool scripts tests
isort --check-only edge_detection_tool scripts tests
flake8 edge_detection_tool scripts tests
mypy edge_detection_tool
pytest -q


---

7  Pull‑Request‑Workflow

1. Branch‑Pattern  feat/<ticket>, fix/<ticket>, chore/<scope>


2. Commit‑Prefix   feat:, fix:, docs:, refactor: usw.


3. PR‑Template   docs/.github/PULL_REQUEST_TEMPLATE.md nutzen (inkl. Check‑Matrix).


4. Required Checks  GitHub Actions‑Job ci.yml → green.


5. Review‑Label  needs‑review, changes‑requested, approved.


6. Squash‑Merge  ist Default; Commit‑Titel = PR‑Titel.




---

8  Verbote & Sicherheitsregeln

Kein Direkt‑Push auf main

Keine Binärdateien > 20 MB im Repo (Modelle werden heruntergeladen)

Keine absoluten Pfade oder Benutzernamen in Code

Keine Debug‑Reste (print, pdb, breakpoint())

Keine Änderung an /models bis auf scripts/model_fetcher.py


Verstöße brechen CI.


---

9  Nested‑Override‑Logik

Eine edge_detection_tool/detectors/AGENTS.md kann Stil‑ und Testvorgaben nur für den Ordner detectors/ überschreiben.

Bei Konflikt gilt: nested > parent > root.



---

10  Edge‑Detection‑Spezifische Leitlinien

1. Inference  Immer torch.no_grad() verwenden.


2. Determinismus  torch.backends.cudnn.deterministic = True, Seed in utils/seed.py.


3. Speichermanagement  clear_cuda_cache() nach jedem Batch, falls gpu_memory_fraction < 0.8.


4. Fallback‑Reihenfolge  HED → BDCN → Structured Forests → Kornia → Fixed CNN.


5. Konfig‑Validierung  Jede CLI setzt config.validate() auf.




---

11  Dok‑Updates & Versionierung

Changelog  Im Footer dieses Files (### Changelog) pflegen.

SemVer  Versions‑Tag = Release‑Version in __init__.py.



---

12  Kontakt & Ownership

Bereich	Owner	Slack‑Tag

Core‑Library	@jdoe	#edge‑detection
CI / Ops	@ops-guru	#devops



---

Changelog

Datum	Version	Änderung

2025‑06‑14	1.0	Erstversion aus README generiert


