import os
import pickle

import mlflow
from flask import Flask, request, jsonify


''' with mlflow server
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000' 

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
RUN_ID = ''
logged_model = f'runs:/{RUN_ID}/models_mlflow'
model = mlflow.pyfunc.load_model(logged_model)
'''

# without mlflow server

logged_model= f's3:/bucket-name//2/{run_ID}/artifacts/models_mlflow'
#logged_model = f's3:///2/{RUN_ID}/artifacts/models_flow'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    print(features)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)