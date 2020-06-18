import pandas as pd


#Load and organize data
df_train = pd.read_csv('train.csv')
df_train.head()


df_target = df_train['Survived']
df_features = df_train[2:]
#df_features.drop('Name')
print(df_features.head())
