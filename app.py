from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
HISTOGRAM_FOLDER = 'static/histograms'

# Buat folder jika belum ada
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, HISTOGRAM_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def apply_enhancement(image_path, method):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, "Failed to load image."

    # Histogram Equalization
    if method == "HE":
        processed = cv2.equalizeHist(image)

    # Adaptive Histogram Equalization
    elif method == "AHE":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed = clahe.apply(image)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    elif method == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        processed = clahe.apply(image)
    
    else:
        return None, "Invalid enhancement method."

    # Simpan gambar hasil
    processed_filename = f"processed_{os.path.basename(image_path)}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_path, processed)

    # Simpan histogram
    hist_original_path = os.path.join(HISTOGRAM_FOLDER, f"hist_original_{os.path.basename(image_path)}.png")
    hist_processed_path = os.path.join(HISTOGRAM_FOLDER, f"hist_processed_{os.path.basename(image_path)}.png")

    save_histogram(image, hist_original_path)
    save_histogram(processed, hist_processed_path)

    return processed_filename, hist_original_path, hist_processed_path, None

def save_histogram(image, hist_path):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=[0,256], color='black', alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.grid(True)
    plt.savefig(hist_path)
    plt.close()

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files['image']
    method = request.form.get('method', 'HE')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    output_filename, hist_original_path, hist_processed_path, error = apply_enhancement(file_path, method)

    if error:
        return jsonify({"error": error}), 400

    return render_template('result.html', 
                           original=file.filename, 
                           processed=output_filename,
                           hist_original=os.path.basename(hist_original_path),
                           hist_processed=os.path.basename(hist_processed_path),
                           method=method)

@app.route('/static/uploads/<filename>')
def get_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/processed/<filename>')
def get_processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/static/histograms/<filename>')
def get_histogram(filename):
    return send_from_directory(HISTOGRAM_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
