import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)

modelpath = "./xgboost.dill"
load_model(modelpath)

"smoke", "alco", "active"

@app.route("/", methods=["GET"])
def general():
    return """<form enctype='application/json' action="http://127.0.0.1:8080/predict" method="POST" >
  <div>
    <label for="age">What is your age?</label>
    <input name="age" id="age" value="18">
  </div>
  <div>
    <label for="gender">What is your gender?</label>
    <select name="gender">
      <option selected value="0">Male</option>
      <option value="1">Female</option>
    </select>
  </div>
  <div>
    <label for="height">What is your height? (cm)</label>
    <input name="height" id="height" value="180">
  </div>
  <div>
    <label for="weight">What is your weight? (kg)</label>
    <input name="weight" id="weight" value="80">
  </div>
  <div>
    <label for="ap_hi">What is your blood pressure (Hi)?</label>
    <input name="ap_hi" id="ap_hi" value="150">
  </div>
  <div>
    <label for="ap_lo">What is your blood pressure (Low)?</label>
    <input name="ap_lo" id="ap_lo" value="100">
  </div>
  <div>
    <label for="cholesterol">What is your cholesterol level?</label>
    <select name="cholesterol">
      <option selected value="1">Normal</option>
      <option value="2">Above normal</option>
      <option value="3">Well above normal</option>
    </select>
  </div>
  <div>
    <label for="glucose">What is your glucose level?</label>
    <select name="glucose">
      <option selected value="1">Normal</option>
      <option value="2">Above normal</option>
      <option value="3">Well above normal</option>
    </select>
  </div>
  <div>
    <label for="smoke">Do you smoke?</label>
    <select name="smoke">
      <option selected value="1">Yes</option>
      <option value="0">No</option>
    </select>
  </div>
  <div>
    <label for="alco">Do you drink alcohol?</label>
    <select name="alco">
      <option selected value="1">Yes</option>
      <option value="0">No</option>
    </select>
  </div>
  <div>
    <label for="active">Are you an active person?</label>
    <select name="active">
      <option selected value="1">Yes</option>
      <option value="0">No</option>
    </select>
  </div>
  <div>
    <button>Send my data</button>
  </div>
</form>"""

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    if flask.request.method == "POST":
        request_features = {}
        features_user = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "glucose", "smoke", "alco", "active"]
        features_true = ["age", "gender__1", "gender__2", "height", "weight", "weight_height_index", "ap_hi", "ap_lo", "cholesterol__1", "cholesterol__2", "cholesterol__3", "gluc__1", "gluc__2", "gluc__3", "smoke", "alco", "active"]
        
        for feature in features_user:
            
            if feature == "gender":
                if flask.request.form.get(feature) == "0":
                    request_features["gender__1"] = 1
                    request_features["gender__2"] = 0
                else:
                    request_features["gender__1"] = 0
                    request_features["gender__2"] = 1
            elif feature == "cholesterol":
                if flask.request.form.get(feature) == "1":
                    request_features["cholesterol__1"] = 1
                    request_features["cholesterol__2"] = 0
                    request_features["cholesterol__3"] = 0
                elif flask.request.form.get(feature) == "2":
                    request_features["cholesterol__1"] = 0
                    request_features["cholesterol__2"] = 1
                    request_features["cholesterol__3"] = 0
                else:
                    request_features["cholesterol__1"] = 0
                    request_features["cholesterol__2"] = 0
                    request_features["cholesterol__3"] = 1
            elif feature == "glucose":
                if flask.request.form.get(feature) == "1":
                    request_features["gluc__1"] = 1
                    request_features["gluc__2"] = 0
                    request_features["gluc__3"] = 0
                elif flask.request.form.get(feature) == "2":
                    request_features["gluc__1"] = 0
                    request_features["gluc__2"] = 1
                    request_features["gluc__3"] = 0
                else:
                    request_features["gluc__1"] = 0
                    request_features["gluc__2"] = 0
                    request_features["gluc__3"] = 1
            else:
                request_features[feature] = int(flask.request.form.get(feature))
                
        request_features["weight_height_index"] = request_features["weight"]/((request_features["height"]/100)**2)
        
        try:
            df = pd.DataFrame(data=request_features,index=[0])
            pred = model.predict(df)
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            return flask.jsonify(data)

        data["prediction"] = int(pred[0])
        data["success"] = True

    print(data)
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
        "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', debug=True, port=port)