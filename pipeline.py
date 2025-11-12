# run_pipeline.py
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Percorsi aggiornati
pipeline = [
    "dicts.py",
    "data_processing.py",
    "pk_functions.py",
    "features_ext.py",
    "set_up.py",
    "models/tuning_function.py",
    "models/tuned_models_generation.py",
    "models/stacking_model_generation.py"



]

print("\n=== Avvio pipeline Pokémon Battles ===\n")

for script in pipeline:
    path = BASE_DIR / script
    if not path.exists():
        print(f"File non trovato: {path}")
        sys.exit(1)
    print(f" Esecuzione: {script}")
    result = subprocess.run([sys.executable, str(path)])
    if result.returncode != 0:
        print(f" Errore durante l'esecuzione di {script}")
        sys.exit(result.returncode)

print("\nPipeline completata con successo.\n")