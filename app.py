from flask import Flask, render_template, request, flash
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
import numpy as np
import logging

# Critical environment variables for Render compatibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configure production logging
if os.environ.get('ENV') == 'production':
    logging.basicConfig(level=logging.WARNING)
else:
    logging.basicConfig(level=logging.INFO)

# Load model BEFORE app starts (critical for Render)
try:
    model = YOLO('best.pt', verbose=False)
    model.fuse()  # Optimize model
    app.logger.info("Model loaded and optimized successfully")
except Exception as e:
    app.logger.critical(f"Model loading failed: {str(e)}")
    raise RuntimeError("Model initialization failed") from e

def process_image(image):
    """Optimized image processing pipeline"""
    try:
        return np.array(image.convert("RGB").resize((640, 640)))
    except Exception as e:
        app.logger.error(f"Image processing error: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    input_image_data = None
    output_image_data = None
    error = None

    if request.method == 'POST':
        try:
            # Validate file upload
            if 'file' not in request.files:
                raise ValueError("No file uploaded")
            
            file = request.files['file']
            if not file or file.filename == '':
                raise ValueError("No file selected")

            # Read and validate image
            img_bytes = file.read()
            if len(img_bytes) == 0:
                raise ValueError("Empty file uploaded")
            if len(img_bytes) > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("File size exceeds 10MB limit")

            # Process image
            image = Image.open(io.BytesIO(img_bytes))
            input_array = process_image(image)
            
            # Create preview
            input_image_data = base64.b64encode(img_bytes).decode('utf-8')

            # Run inference with memory limits
            results = model.predict(
                source=input_array,
                imgsz=640,
                conf=0.5,
                save=False,
                max_det=10  # Limit detections
            )

            if not results or len(results[0].boxes) == 0:
                raise RuntimeError("No watermarks detected")

            # Process results
            result_img = Image.fromarray(results[0].plot())
            buf = io.BytesIO()
            result_img.save(buf, format='JPEG', quality=85, optimize=True)
            output_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        except Exception as e:
            error = str(e)
            app.logger.error(f"Error: {error}")
            flash(f"Error: {error}", 'error')

    return render_template(
        'index.html',
        input_image_data=input_image_data,
        output_image_data=output_image_data,
        error=error
    )

@app.route('/health')
def health_check():
    """Render health check endpoint"""
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
