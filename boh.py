# run_pipeline.py
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Percorsi aggiornati
pipeline = [
    
    "pk_functions.py",
    "features_ext.py",
    "set_up.py",
    "models/prove.py"
]

print("\n=== Avvio pipeline Pokémon Battles ===\n")

for script in pipeline:
    path = BASE_DIR / script
    if not path.exists():
        print(f"❌ File non trovato: {path}")
        sys.exit(1)
    print(f"▶ Esecuzione: {script}")
    result = subprocess.run([sys.executable, str(path)])
    if result.returncode != 0:
        print(f"⚠️ Errore durante l'esecuzione di {script}")
        sys.exit(result.returncode)

print("\n✅ Pipeline completata con successo.\n")