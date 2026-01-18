from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from bm3d import bm3d, BM3DStages
import threading

app = Flask(__name__)

# Folders inside static/
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join('static', 'outputs')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load ESRGAN model once
esrgan_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

# Globals to track progress
progress = 0
processing_complete = False
input_file_name = ""
output_file_name = ""

def denoise_image_task(input_path, output_path):
    global progress, processing_complete
    progress = 10

    # --- ESRGAN super-resolution ---
    img = Image.open(input_path).convert("RGB")
    img = img.resize((256, 256))
    x = np.array(img)/255.0
    x = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    
    progress = 30
    y = esrgan_model(x)
    y = tf.clip_by_value(y[0], 0.0, 1.0).numpy()
    y = (y*255).astype(np.uint8)
    output_img = Image.fromarray(y).convert("L")
    
    temp_path = output_path  # reuse same output path
    output_img.save(temp_path)
    progress = 60

    # --- BM3D Denoising ---
    img_gray = np.array(output_img, dtype=np.float32) / 255.0
    sigma_est = np.mean(np.std(img_gray))
    denoised = bm3d(img_gray, sigma_psd=sigma_est, stage_arg=BM3DStages.ALL_STAGES)
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(denoised).save(output_path)

    progress = 100
    processing_complete = True


@app.route("/", methods=["GET", "POST"])
def index():
    global progress, processing_complete, input_file_name, output_file_name

    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        # Save uploaded file
        input_file_name = file.filename
        output_file_name = "denoised_" + file.filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_file_name)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file_name)
        file.save(input_path)

        # Reset progress
        progress = 0
        processing_complete = False

        # Start background processing
        threading.Thread(target=denoise_image_task, args=(input_path, output_path)).start()

        return redirect(url_for("progress_page"))

    return render_template("index.html")


@app.route("/progress")
def progress_page():
    return render_template("progress.html",
                           progress=progress,
                           complete=processing_complete,
                           input_file=input_file_name,
                           output_file=output_file_name)


@app.route("/progress_status")
def progress_status():
    global progress, processing_complete
    return {"progress": progress, "complete": processing_complete}


if __name__ == "__main__":
    app.run(debug=True)
