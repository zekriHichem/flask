from flask import Flask

from flask_cors import CORS

from numpy import genfromtxt

app = Flask(__name__)
CORS(app)

@app.route('/data1')
def data1():
    dict = {3: 8, 2: 16, 0.3: 4}

    return dict

@app.route('/data2')
def data2():
    dict = {6: 8, 4: 1, 1: 7}

    return dict

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
