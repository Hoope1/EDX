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

REM Submodul pruefen
IF NOT EXIST bdcn_repo\ (
    echo [WARNING] BDCN-Submodul nicht gefunden!
    echo Bitte ausfuehren: git submodule update --init --recursive
)

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
