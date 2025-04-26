#!/usr/bin/env python3
#!/usr/bin/env python3
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import logging

app = Flask(__name__)
model = tf.keras.models.load_model('model/test_model_00.keras', compile=False)
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_array = data.get('input')
    
    if input_array is None:
        return jsonify({'error': 'Missing input array'}), 400

    try:
        input_np = np.array(input_array, dtype=np.float32)
        prediction = model.predict(input_np)
        logging.debug(prediction)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
