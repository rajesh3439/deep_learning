# app.py
from flask import Flask, request
import json
import pickle
import pandas as pd
from io import StringIO

app = Flask(__name__)
model = pickle.load(open('anomaly_detection.pkl', 'rb'))

def get_df(data):
    return pd.read_csv(StringIO(data), sep=',')
    # _data = json.loads(data)
    # return pd.DataFrame(_data, index=range(len(_data)))

@app.route('/predict', methods=['POST'])
def predict():
    df = get_df(request.json['data'])
    # print(df) 
    preds = model.predict(df)
    # reset the predicted labels to match the original labels (0: normal, 1: anomalous)
    preds[preds==1]=0
    preds[preds==-1]=1

    return json.dumps(preds.tolist())

if __name__ == '__main__':
    app.run(debug=True)