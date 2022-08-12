from flask import *
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import Test
import os

app = Flask(__name__)

result = ""


@app.route('/')
def index():
    return render_template('index.html', output=result)


@app.route('/documentation')
def documentation():
    return render_template('documentation.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = Test.test_model(file_path)
        os.remove(file_path)
        # return result
    return render_template('index.html', output=result)


# Disable during deployment
if __name__ == "__main__":
    app.run(debug=True)
