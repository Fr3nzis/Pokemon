import json
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_data = []
        self.test_data = []
        self.train_file_path = os.path.join(self.data_path, 'train.jsonl')
        self.test_file_path = os.path.join(self.data_path, 'test.jsonl')

    def load_train_data(self):
        """Carica il file train.jsonl riga per riga."""
        print(f"Loading data from '{self.train_file_path}'...")
        try:
            with open(self.train_file_path, 'r') as f:
                for line in f:
                    self.train_data.append(json.loads(line))
            print(f"Successfully loaded {len(self.train_data)} train battles.")
        except FileNotFoundError:
            print(f"ERROR: Could not find the training file at '{self.train_file_path}'.")
            print("Please make sure you have added the competition data to this notebook.")

    def load_test_data(self):
        """Carica il file test.jsonl riga per riga."""
        print(f"Loading data from '{self.test_file_path}'...")
        try:
            with open(self.test_file_path, 'r') as f:
                for line in f:
                    self.test_data.append(json.loads(line))
            print(f"Successfully loaded {len(self.test_data)} test battles.")
        except FileNotFoundError:
            print(f"ERROR: Could not find the test file at '{self.test_file_path}'.")
            print("Please make sure you have added the competition data to this notebook.")           

    def inspect_first_battle(self):
        """Mostra la struttura della prima battaglia del train set."""
        if not self.train_data:
            print("No training data loaded yet.")
            return

        print("\n--- Structure of the first train battle: ---")
        first_battle = self.train_data[0]

        # Copia e troncamento della timeline per la visualizzazione
        battle_for_display = first_battle.copy()
        battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:2]

        print(json.dumps(battle_for_display, indent=4))
        if len(first_battle.get('battle_timeline', [])) > 3:
            print("    ...")
            print("    (battle_timeline has been truncated for display)")


class DataProcessor:
    def __init__(self, train_df, test_df, save_dir="data"):
        self.train_df = train_df
        self.test_df = test_df
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def prepare_data(self):
        """Prepara X_train, Y_train e X_test per i modelli ML"""
        # Colonne da escludere
        drop_cols = ['battle_id', 'player_won']
        features = [col for col in self.train_df.columns if col not in drop_cols]

        # Split
        X_train = self.train_df[features]
        Y_train = self.train_df['player_won']
        X_test = self.test_df[features]

        # Salvataggio
        X_train.to_pickle(f"{self.save_dir}/X_train.pkl")
        X_train.to_csv(f"{self.save_dir}/X_train.csv", index=False)

        X_test.to_pickle(f"{self.save_dir}/X_test.pkl")
        X_test.to_csv(f"{self.save_dir}/X_test.csv", index=False)

        Y_train.to_pickle(f"{self.save_dir}/Y_train.pkl")
        Y_train.to_csv(f"{self.save_dir}/Y_train.csv", index=False)

        print("Salvataggio completato in pickle e CSV nella cartella 'data'.")
        return Y_train, X_train, X_test