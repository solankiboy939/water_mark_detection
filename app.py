from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
model = YOLO('best.pt')  # Your trained watermark logo detection model

@app.route('/', methods=['GET', 'POST'])
def index():
    input_image_data = None
    output_image_data = None

    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        # Convert input image to base64 for preview
        input_image_data = base64.b64encode(img_bytes).decode('utf-8')

        # Save image temporarily in memory
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        temp_input_path = 'temp.jpg'
        image.save(temp_input_path)

        # Run YOLO detection
        results = model(temp_input_path, save=False, imgsz=640)

        # Get image with boxes
        result_img = results[0].plot()
        im_pil = Image.fromarray(result_img)

        # Convert output image to base64
        buf = io.BytesIO()
        im_pil.save(buf, format='JPEG')
        output_image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Clean up temp image if needed
        os.remove(temp_input_path)

    return render_template('index.html',
                           input_image_data=input_image_data,
                           output_image_data=output_image_data)

if __name__ == "__main__":
    app.run(debug=True)
