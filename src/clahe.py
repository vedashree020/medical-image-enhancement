import cv2


def apply_clahe(image):

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    # Apply CLAHE
    enhanced = clahe.apply(image)

    return enhanced