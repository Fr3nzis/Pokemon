# run_pipeline.py
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Percorsi aggiornati
pipeline = [
    "set_up_scripts/dicts.py",
    "set_up_scripts/data_processing.py",
    "set_up_scripts/pk_functions.py",
    "set_up_scripts/features_ext.py",
    "set_up_scripts/set_up.py",
    "models/tuning_function.py",
    "models/tuned_models_generation.py",   
    "models/stacking_functions.py",
    "models/staking_model_generation.py"
    




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