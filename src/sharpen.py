import cv2


def sharpen_image(image):

    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (0,0), 3)

    # Unsharp masking
    sharpened = cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

    return sharpened