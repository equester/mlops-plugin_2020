# Data Handling
import pickle
import numpy as np
import pandas as pd
from pydantic import BaseModel
# Server
import uvicorn
from fastapi import FastAPI

# Modeling


app = FastAPI()

# Initialize files
clf = pickle.load(open('C://mlops_plugin//_deploy//model//model.pkl', 'rb'))



class Data(BaseModel):
    Pclass: int
    Sex: int
    Embarked: int
    Title: int
    IsAlone: int
    FareBand: int
    AgeBand: int


@app.post("/predict")
def predict(data: Data):
    data_dict = data.dict()
    # Create and return prediction
    to_predict = pd.DataFrame.from_dict(data_dict,orient='index').T.iloc[0].values.reshape(1, -1)
    prediction = clf.predict(to_predict)

    return {"prediction": int(prediction)}
