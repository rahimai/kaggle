import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
#pd.set_option('display.max_columns', None)


#Load and organize data
df_train = pd.read_csv('train.csv')
df_train.head()

#Split target variable from features and drop Name feature
df_target = df_train['Survived']
df_features = df_train.loc[:, 'Pclass':]
df_features.drop('Name', axis=1, inplace=True)
df_features.drop('Ticket', axis=1, inplace=True)

#Feature engineering
#print(df_features['Embarked'].value_counts())


#Get Dummies
df_features = pd.get_dummies(df_features, columns=['Pclass', 'Sex', 'Cabin', 'Embarked'])


#Split training and validation data
df_xtrain, df_xtest, df_ytrain, df_ytest = train_test_split(df_features, df_target, test_size=0.2, random_state=42)


#print(df_features['SibSp'].value_counts())
#print(df_features.head())

xgb_train = xgb.DMatrix(df_xtrain, label=df_ytrain)
xgb_test = xgb.DMatrix(df_ytrain, label=df_ytest)

clf = xgb.XGBClassifier()
parameters = {"eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
     "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight": [1, 3, 5, 7],
     "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
     "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}

grid = GridSearchCV(clf,
                    parameters, n_jobs=4,
                    scoring="accuracy",
                    cv=3, verbose=5)

grid.fit(df_xtrain, df_xtest)