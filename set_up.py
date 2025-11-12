from data_processing import DataHandler
from features_ext import FeatureHandler
from data_processing import DataProcessor, DataProcessor


# Percorso base
DATA_PATH = 'fds-pokemon-battles-prediction-2025'
# Inizializza il gestore dati
handler = DataHandler(DATA_PATH)

# Carica i dataset
handler.load_train_data()
handler.load_test_data()

# Mostra struttura prima battaglia
handler.inspect_first_battle()

# Mostra anteprima numerica
print(f"\nTrain battles loaded: {len(handler.train_data)}")
print(f"Test battles loaded:  {len(handler.test_data)}")


train_data = handler.train_data
test_data = handler.test_data


#Crea le feature
feature_handler = FeatureHandler(train_data)
df_train = feature_handler.create_advanced_features(train_data)
df_test = feature_handler.create_advanced_features(test_data)


#Mostra le prime righe del DataFrame   
print("Prime righe del dataset con feature estratte:")
print(df_train.head())

# salva in formato pickle e CSV
df_train.to_pickle("data/df_train.pkl")
df_train.to_csv("data/df_train.csv", index=False)
df_test.to_pickle("data/df_test.pkl")
df_test.to_csv("data/df_test.csv", index=False)

#Data Frame Processing 
processor = DataProcessor(df_train, df_test)
Y_train, X_train, X_test = processor.prepare_data()


 

