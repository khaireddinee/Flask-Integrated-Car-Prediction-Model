import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model_car.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # extracting features
    make = request.form['make']
    num_of_doors = request.form['num-of-doors']
    body_style = request.form['body-style']
    drive_wheels = request.form['drive-wheels']
    engine_type = request.form['engine-type']
    num_of_cylinders = request.form['num-of-cylinders']
    fuel_system = request.form['fuel-system']
    horse_power = int(request.form['horse-power'])
    peak_rpm = int(request.form['peak-rpm'])
    city_mpg = int(request.form['city-mpg'])
    highway_mpg = int(request.form['highway-mpg'])

    # initializing the scaler
    scaler = MinMaxScaler()
    horse_power_scaled = scaler.fit_transform(np.array(horse_power).reshape(-1, 1)).flatten()[0]
    peak_rpm_scaled = scaler.fit_transform(np.array(peak_rpm).reshape(-1, 1)).flatten()[0]
    city_mpg_scaled = scaler.fit_transform(np.array(city_mpg).reshape(-1, 1)).flatten()[0]
    highway_mpg_scaled = scaler.fit_transform(np.array(highway_mpg).reshape(-1, 1)).flatten()[0]

    # label encoding
    door_mapping = {'two': 0, 'four': 1}
    num_of_doors_int = door_mapping.get(num_of_doors.lower(), None)

    engine_mapping = {'dohc':0, 'ohcv':1, 'ohc':2, 'l':3, 'rotor':4, 'ohcf':5, 'dohcv':6}
    engine_type_int = engine_mapping.get(engine_type.lower(), None)

    make_mapping = {'alfa-romero':0, 'audi':1, 'bmw':2, 'chevrolet':3, 'dodge':4, 'honda':5,'isuzu':6, 'jaguar':7, 'mazda':8, 'mercedes-benz':9, 'mercury':10,
       'mitsubishi':11, 'nissan':12, 'peugot':13, 'plymouth':14, 'porsche':15, 'renault':16,
       'saab':17, 'subaru':18, 'toyota':19, 'volkswagen':20, 'volvo':21}
    make_int = make_mapping.get(make.lower(),None)

    cylinders_mapping = {'four':4, 'six':6, 'five':5, 'three':3, 'twelve':12, 'two':2, 'eight':8}
    num_of_cylinders_int = cylinders_mapping.get(num_of_cylinders.lower(),None)

    body_mapping = {'convertible':0, 'hatchback':1, 'sedan':2, 'wagon':3, 'hardtop':4}
    body_style_int = body_mapping.get(body_style.lower(),None)

    fuel_mapping = {'mpfi':0, '2bbl':1, 'mfi':2, '1bbl':3, 'spfi':4, '4bbl':5, 'idi':6, 'spdi':7}
    fuel_system_int = fuel_mapping.get(fuel_system.lower(),None)

    drive_mapping = {'rwd':0, 'fwd':1, '4wd':2}
    drive_wheels_int = drive_mapping.get(drive_wheels.lower(),None)

    # filling features that don't exist in the form

    normalized_losses = 0.2912431988502207
    aspiration = 0
    wheel_base = 0.35587091979649016
    length = 0.4921641791044777
    width = 0.4680555555555559
    height = 53.749019607843145
    curb_weight = 0.4141206133345503
    engine_size = 0.24864964853866076
    bore = 0.5633403361344538
    stroke = 0.5661531279178339

    all_features = []
    all_features.extend([normalized_losses,make_int,aspiration,num_of_doors_int, body_style_int, drive_wheels_int,wheel_base,length,width,height,curb_weight,engine_type_int,num_of_cylinders_int,engine_size,fuel_system_int,bore,stroke,horse_power_scaled, peak_rpm_scaled, city_mpg_scaled, highway_mpg_scaled])

    features_array = [np.array(all_features)]
    #return render_template("index.html", prediction_text = "The price of the car is {}".format(all_features))
    prediction = model.predict(features_array)
    return render_template("index.html", prediction_text = "The price of the car is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)