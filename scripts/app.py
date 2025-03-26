import json
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load dictionary
with open('data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)

def process_page(img, scale, margin, use_dictionary, min_words_per_line, text_scale):
    """Processes the image and returns recognized text & visualization."""
    read_lines = read_page(img,
                           detector_config=DetectorConfig(scale=scale, margin=margin),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=min_words_per_line),
                           reader_config=ReaderConfig(decoder='word_beam_search' if use_dictionary else 'best_path',
                                                      prefix_tree=prefix_tree))

    res = ''
    for read_line in read_lines:
        res += ' '.join(read_word.text for read_word in read_line) + '\n'

    for read_line in read_lines:
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(img, (aabb.xmin, aabb.ymin), 
                          (aabb.xmin + aabb.width, aabb.ymin + aabb.height), 
                          (255, 0, 0), 2)
            cv2.putText(img, read_word.text, 
                        (aabb.xmin, aabb.ymin + aabb.height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0))

    return res, img

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to process the image and return recognized text."""
    try:
        print("Received request:", request)  # Debug log
        print("Request files:", request.files)  # Debug log

        # Check if image exists
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        print("Received file:", file.filename)  # Debug log

        scale = float(request.form.get('scale', 1))
        margin = int(request.form.get('margin', 1))
        use_dictionary = request.form.get('use_dictionary', 'false').lower() == 'true'
        min_words_per_line = int(request.form.get('min_words_per_line', 2))
        text_scale = float(request.form.get('text_scale', 1))

        # Convert to OpenCV image
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Process the image
        text, img = process_page(img, scale, margin, use_dictionary, min_words_per_line, text_scale)

        # Encode the image
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({
            'text': text,
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
