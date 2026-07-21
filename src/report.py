import numpy as np



def generate_report(original, enhanced):


    # brightness calculation

    original_brightness = np.mean(
        original
    )


    enhanced_brightness = np.mean(
        enhanced
    )



    # contrast calculation

    original_contrast = np.std(
        original
    )


    enhanced_contrast = np.std(
        enhanced
    )



    brightness_change = (
        enhanced_brightness -
        original_brightness
    )



    contrast_change = (
        (
            enhanced_contrast -
            original_contrast
        )
        /
        original_contrast
    ) * 100



    report = {

        "Original Brightness":
        round(
            original_brightness,
            2
        ),


        "Enhanced Brightness":
        round(
            enhanced_brightness,
            2
        ),


        "Brightness Change":
        round(
            brightness_change,
            2
        ),


        "Contrast Improvement":
        round(
            contrast_change,
            2
        )

    }



    return report