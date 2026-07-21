import cv2
import matplotlib.pyplot as plt
import os

from denoise import denoise_image
from clahe import apply_clahe
from sharpen import sharpen_image
from metrics import calculate_mse, calculate_psnr, calculate_ssim



# Load image

image = cv2.imread(
    "../dataset/image1.png",
    cv2.IMREAD_GRAYSCALE
)


if image is None:
    print("Image not found")
    exit()



print("Image Loaded Successfully")


# Enhancement pipeline

denoised = denoise_image(image)

clahe_image = apply_clahe(denoised)

final_image = sharpen_image(clahe_image)



# Calculate metrics

mse = calculate_mse(
    image,
    final_image
)

psnr = calculate_psnr(
    image,
    final_image
)

ssim_value = calculate_ssim(
    image,
    final_image
)



print("\n========== IMAGE QUALITY METRICS ==========")

print("MSE  :", mse)

print("PSNR :", psnr, "dB")

print("SSIM :", ssim_value)



# Save final image

if not os.path.exists("../output"):
    os.makedirs("../output")


cv2.imwrite(
    "../output/enhanced_xray.png",
    final_image
)


print("\nEnhanced image saved!")



# Display

plt.figure(figsize=(16,5))


plt.subplot(1,4,1)
plt.imshow(image,cmap="gray")
plt.title("Original X-ray")
plt.axis("off")


plt.subplot(1,4,2)
plt.imshow(denoised,cmap="gray")
plt.title("Denoised")
plt.axis("off")


plt.subplot(1,4,3)
plt.imshow(clahe_image,cmap="gray")
plt.title("CLAHE")
plt.axis("off")


plt.subplot(1,4,4)
plt.imshow(final_image,cmap="gray")
plt.title("Final Enhanced")
plt.axis("off")


plt.show()