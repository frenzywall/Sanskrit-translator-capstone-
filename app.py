import os
from flask import Flask, render_template, request, redirect, url_for
import pytesseract
import cv2
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for OCR
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

# Function to extract text using OCR
def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed_image, config='--oem 3 --psm 6')
    return text

# Home route to upload an image
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and text extraction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from image
        extracted_text = extract_text_from_image(filepath)
        
        return render_template('result.html', text=extracted_text, filename=filename)
    else:
        return 'Invalid file type, please upload an image.'

if __name__ == '__main__':
    app.run(debug=True)
