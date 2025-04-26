from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

model = None  # Lazy load

@app.route('/', methods=['GET', 'POST'])
def index():
    global model
    input_image_data = None
    output_image_data = None

    if request.method == 'POST':
        # Load model if not already loaded
        if model is None:
            model = YOLO('best.pt')
            model.to('cpu')  # FORCE CPU
        
        file = request.files['file']
        img_bytes = file.read()

        # Input image base64
        input_image_data = base64.b64encode(img_bytes).decode('utf-8')

        # Open image directly
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Predict on CPU
        results = model.predict(image, save=False, imgsz=640, device='cpu')

        # Plot results
        result_img = results[0].plot()
        im_pil = Image.fromarray(result_img)

        # Output image base64
        buf = io.BytesIO()
        im_pil.save(buf, format='JPEG')
        output_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template('index.html',
                           input_image_data=input_image_data,
                           output_image_data=output_image_data)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
