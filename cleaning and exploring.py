import pandas as pd


df = pd.read_csv('liked_songs_features.csv')


print(df.head())

print(df.info())


print(df.describe())

print(df.isnull().sum())



print(df.dtypes)

df.fillna(0, inplace=True)  


df.drop_duplicates(inplace=True)



from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
numerical_features = ['danceability', 'energy', 'valence', 'tempo']  
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print(df.head())
print(df.info())

