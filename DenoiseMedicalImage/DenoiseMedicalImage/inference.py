import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from bm3d import bm3d, BM3DStages

# Load ESRGAN from TF Hub
model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

# Load and preprocess image
img = Image.open("Image1.png").convert("RGB")
img = img.resize((256,256))
x = np.array(img)/255.0
x = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)  # add batch dim

# Inference
y = model(x)
y = tf.clip_by_value(y[0], 0.0, 1.0).numpy()
y = (y*255).astype(np.uint8)
output_img = Image.fromarray(y).convert("L")  # convert to grayscale
output_img.save("output_image_gray.png")


def denoise_medical_image(image_path):
    # Load image in grayscale using PIL
    img = Image.open(image_path).convert("L")
    img = np.array(img, dtype=np.float32) / 255.0

    # Estimate noise sigma (simple heuristic)
    sigma_est = np.mean(np.std(img))

    # BM3D denoising
    denoised = bm3d(
        img,
        sigma_psd=sigma_est,
        stage_arg=BM3DStages.ALL_STAGES
    )

    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    return denoised


# ---- Example usage in Colab ----
input_image = "output_image_gray.png"
output_image = "denoised_image.png"

result = denoise_medical_image(input_image)

# Save output
Image.fromarray(result).save(output_image)

# Display result
plt.figure(figsize=(6, 6))
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.title("Denoised Image")
plt.show()

print(f"Denoised image saved as: {output_image}")
