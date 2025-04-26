from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
import numpy as np

app = Flask(__name__)

# 🛠️ Load model ONCE when server starts
model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def index():
    input_image_data = None
    output_image_data = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_bytes = file.read()

            # Convert input image to base64 for preview
            input_image_data = base64.b64encode(img_bytes).decode('utf-8')

            # Open the uploaded image
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # 🚀 Resize (optional) to avoid huge RAM usage
            image.thumbnail((640, 640))

            # Convert PIL Image to numpy array
            img_array = np.array(image)

            # 🔥 Run detection (NO model loading here)
            results = model.predict(source=img_array, save=False, imgsz=640, device='cpu')

            # Get the resulting image with boxes
            result_img = results[0].plot()
            im_pil = Image.fromarray(result_img)

            # Convert output image to base64
            buf = io.BytesIO()
            im_pil.save(buf, format='JPEG')
            output_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template('index.html',
                           input_image_data=input_image_data,
                           output_image_data=output_image_data)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
