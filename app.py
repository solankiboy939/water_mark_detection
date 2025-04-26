from flask import Flask, render_template, request, flash
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model once at startup
try:
    model = YOLO('best.pt')
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Could not initialize model") from e

def process_image(image):
    """Process image for model inference"""
    try:
        # Convert to RGB and resize
        img = image.convert("RGB").resize((640, 640))
        return np.array(img)
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
            if 'file' not in request.files:
                raise ValueError("No file uploaded")
            
            file = request.files['file']
            if file.filename == '':
                raise ValueError("No file selected")

            # Read and validate image
            img_bytes = file.read()
            if len(img_bytes) == 0:
                raise ValueError("Empty file uploaded")
            
            # Process image
            image = Image.open(io.BytesIO(img_bytes))
            input_array = process_image(image)
            
            # Create input image preview
            input_image_data = base64.b64encode(img_bytes).decode('utf-8')

            # Run inference
            results = model.predict(
                source=input_array,
                imgsz=640,
                conf=0.5,
                save=False
            )

            if not results:
                raise RuntimeError("No results from model")

            # Process results
            result_img_array = results[0].plot()
            result_img = Image.fromarray(result_img_array)

            # Create output image
            buf = io.BytesIO()
            result_img.save(buf, format='JPEG', quality=90)
            output_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        except Exception as e:
            error = str(e)
            app.logger.error(f"Processing error: {error}")
            flash(error, 'error')

    return render_template(
        'index.html',
        input_image_data=input_image_data,
        output_image_data=output_image_data,
        error=error
    )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
