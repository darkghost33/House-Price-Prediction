from sys import stderr

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
col=['Bommanahalli	','Whitefield','BHK','Furnishing','Sq_ft','Old','Floor']

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final=[np.array(int_features, dtype=float)]
    prediction=model.predict(final)
    output=round(prediction[0],2)

    return render_template('index.html', pred='The predicted price of your house is  ${} only.'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
