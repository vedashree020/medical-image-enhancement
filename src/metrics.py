import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_mse(original, enhanced):

    mse_value = np.mean(
        (original.astype(float) - enhanced.astype(float)) ** 2
    )

    return mse_value



def calculate_psnr(original, enhanced):

    mse_value = calculate_mse(original, enhanced)

    if mse_value == 0:
        return 100

    psnr_value = 10 * np.log10(
        (255 ** 2) / mse_value
    )

    return psnr_value



def calculate_ssim(original, enhanced):

    score, _ = ssim(
        original,
        enhanced,
        full=True
    )

    return score