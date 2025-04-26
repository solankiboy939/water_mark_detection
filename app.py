from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load YOLO model and force CPU
model = YOLO('best.pt')
model.to('cpu')  # FORCE CPU (important)

@app.route('/', methods=['GET', 'POST'])
def index():
    input_image_data = None
    output_image_data = None

    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        # Convert input image to base64 for preview
        input_image_data = base64.b64encode(img_bytes).decode('utf-8')

        # Save uploaded image temporarily
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        temp_input_path = 'temp.jpg'
        image.save(temp_input_path)

        # Run YOLO detection on CPU
        results = model(temp_input_path, device='cpu', save=False, imgsz=640)

        # Get image with bounding boxes
        result_img = results[0].plot()
        im_pil = Image.fromarray(result_img)

        # Convert output image to base64
        buf = io.BytesIO()
        im_pil.save(buf, format='JPEG')
        output_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Clean up temporary file
        os.remove(temp_input_path)

    return render_template('index.html',
                           input_image_data=input_image_data,
                           output_image_data=output_image_data)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)