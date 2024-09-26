from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from utils.image_processing import load_and_resize_images, calculate_total_similarity

app = Flask(__name__)
CORS(app)

@app.route('/api/compare', methods=['POST'])
def compare_images():
    try:
        image1 = request.files['image1'].read()
        image2 = request.files['image2'].read()


        img1, img2 = load_and_resize_images(image1, image2)

        result = calculate_total_similarity(img1, img2)

        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': '서버 오류가 발생했습니다: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)