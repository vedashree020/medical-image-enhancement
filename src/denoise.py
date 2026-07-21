import cv2


def denoise_image(image):

    # Convert to grayscale if required
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Mild denoising (preserves edges)
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=5,
        templateWindowSize=7,
        searchWindowSize=21
    )

    return denoised