from data_processing import DataHandler
from features_ext_vot import FeatureHandler
from data_processing import DataProcessor, DataProcessor


DATA_PATH = 'fds-pokemon-battles-prediction-2025'
handler = DataHandler(DATA_PATH)

handler.load_train_data()
handler.load_test_data()

handler.inspect_first_battle()

print(f"\nTrain battles loaded: {len(handler.train_data)}")
print(f"Test battles loaded:  {len(handler.test_data)}")


train_data = handler.train_data
test_data = handler.test_data


feature_handler = FeatureHandler(train_data)
df_train = feature_handler.create_advanced_features(train_data)
df_test = feature_handler.create_advanced_features(test_data)


print("Prime righe del dataset con feature estratte:")
print(df_train.head())

df_train.to_pickle("data/df_train.pkl")
df_train.to_csv("data/df_train.csv", index=False)
df_test.to_pickle("data/df_test.pkl")
df_test.to_csv("data/df_test.csv", index=False)

processor = DataProcessor(df_train, df_test)
Y_train, X_train, X_test = processor.prepare_data()


 

