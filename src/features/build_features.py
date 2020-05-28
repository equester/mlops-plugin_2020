import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

def prep_process(data):
    mlflow.set_tracking_uri("http://kubernetes.docker.internal:5000/")
    with mlflow.start_run() as mlrun:
        data_tmp_dir = "C://mlops_plugin//src//data_temp"
        # Drop Ticket & Cabin
        data = data.drop(['Ticket', 'Cabin'], axis=1)

        # Get the title from name
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
         	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data['Title'] = data['Title'].replace('Mlle', 'Miss')
        data['Title'] = data['Title'].replace('Ms', 'Miss')
        data['Title'] = data['Title'].replace('Mme', 'Mrs')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        data['Title'] = data['Title'].map(title_mapping)
        data = data.drop(['Name', 'PassengerId'], axis=1)

        #Change Sex to Numeric
        data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

        #Add Family Size
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1  # gives the number of people per ticket
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
        data = data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

        #Imputing Missing Value
        data['Age'].fillna(data['Age'].dropna().median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].dropna().mode()[0], inplace=True)

        # Categorizing Numerical Value
        data['FareBand'] = pd.qcut(data['Fare'], 4).astype(str)
        data['AgeBand'] = pd.qcut(data['Age'], 4).astype(str)
        # data = data.drop(['Fare', 'Age'], axis=1)

        # Converting Embark to Number
        data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        data['FareBand'] = data['FareBand'].map( {'(-0.001, 7.91]': 0, '(31.0, 512.329]': 3, '(7.91, 14.454]': 1, '(14.454, 31.0]':2 } ).astype(int)
        data['AgeBand'] = data['AgeBand'].map( {'(0.419, 22.0]': 0, '(35.0, 80.0]': 3, '(22.0, 28.0]': 1, '(28.0, 35.0]':2 } ).astype(int)
        data = data.drop(['Fare','Age'],axis=1)
        data.to_csv(data_tmp_dir+"//temp.csv")
        mlflow.log_artifacts(data_tmp_dir, "data writing")
        X = data.loc[:, data.columns != 'Survived']
        y = data.Survived
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    return x_train,x_test,y_train,y_test, X, y
