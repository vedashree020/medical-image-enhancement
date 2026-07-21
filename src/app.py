import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os


from denoise import denoise_image
from clahe import apply_clahe
from sharpen import sharpen_image

from metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim
)

from report import generate_report



# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="MedEnhance AI",
    page_icon="🩻",
    layout="wide"
)



# =========================
# CSS DESIGN
# =========================


st.markdown(
"""
<style>


.stApp{

background:
linear-gradient(
135deg,
#E3F2FD,
#FCE4EC,
#E8F5E9
);

}



.title{

font-size:55px;

font-weight:900;

text-align:center;


background:
linear-gradient(
90deg,
#1565C0,
#8E24AA,
#00C853
);


-webkit-background-clip:text;

-webkit-text-fill-color:transparent;

}



.subtitle{

text-align:center;

font-size:22px;

color:#37474F;

font-weight:bold;

}



.section{

font-size:30px;

font-weight:800;

color:#1565C0;

}



.card{

background:white;

padding:25px;

border-radius:20px;

box-shadow:
0px 8px 25px rgba(0,0,0,0.15);

border-top:
6px solid #2196F3;

text-align:center;

}



.metric-title{

font-size:18px;

color:#607D8B;

}



.metric-value{

font-size:35px;

font-weight:900;

color:#8E24AA;

}



img{

border-radius:20px;

box-shadow:
0px 5px 20px rgba(0,0,0,0.2);

}



[data-testid="stFileUploader"]{

background:white;

padding:20px;

border-radius:20px;

border:
3px dashed #2196F3;

}


section[data-testid="stSidebar"]{


background:
linear-gradient(
180deg,
#1565C0,
#7B1FA2
);

}


section[data-testid="stSidebar"] *{

color:white !important;

}


.stDownloadButton button{


background:
linear-gradient(
90deg,
#00C853,
#009688
);


color:white;

font-weight:bold;

border-radius:15px;

height:50px;

width:100%;

}


</style>

""",
unsafe_allow_html=True
)




# =========================
# HEADER
# =========================


st.markdown(
"""
<div class="title">
🩻 MedEnhance AI
</div>

<div class="subtitle">
Medical X-Ray Image Enhancement System
</div>

<br>

""",
unsafe_allow_html=True
)




# =========================
# SIDEBAR
# =========================


with st.sidebar:


    st.header("🩺 Project Overview")


    st.write(
"""
### Problem

Low quality X-ray images may hide important details.


### Solution

Image enhancement pipeline:

✔ Noise Removal

✔ Contrast Improvement

✔ Edge Enhancement


### Technologies

🐍 Python

🖼 OpenCV

🌐 Streamlit

"""
)




# =========================
# IMAGE UPLOAD
# =========================


st.markdown(
"""
<div class="section">
📤 Upload Chest X-Ray
</div>
""",
unsafe_allow_html=True
)



uploaded_file = st.file_uploader(

"Upload image",

type=[
"png",
"jpg",
"jpeg"
]

)




if uploaded_file:


    image = Image.open(
        uploaded_file
    )


    image=np.array(image)



    if len(image.shape)==3:


        gray=cv2.cvtColor(

            image,

            cv2.COLOR_RGB2GRAY

        )


    else:

        gray=image




    st.success(
        "Image loaded successfully"
    )



    # =====================
    # ENHANCEMENT PIPELINE
    # =====================


    with st.spinner(
        "Enhancing X-Ray..."
    ):


        denoised = denoise_image(
            gray
        )


        clahe = apply_clahe(
            denoised
        )


        enhanced = sharpen_image(
            clahe
        )



    st.success(
        "Enhancement completed"
    )




    # =====================
    # IMAGE DISPLAY
    # =====================


    st.markdown(
"""
<div class="section">
🖼 Before and After
</div>
""",
unsafe_allow_html=True
)



    c1,c2=st.columns(2)



    with c1:


        st.subheader(
            "Original"
        )


        st.image(
            gray,
            channels="GRAY",
            use_container_width=True
        )



    with c2:


        st.subheader(
            "Enhanced"
        )


        st.image(
            enhanced,
            channels="GRAY",
            use_container_width=True
        )




    # =====================
    # AI REPORT
    # =====================


    report = generate_report(
        gray,
        enhanced
    )



    st.markdown(
"""
<div class="section">
🧠 AI Enhancement Report
</div>
""",
unsafe_allow_html=True
)



    r1,r2=st.columns(2)



    with r1:


        st.info(
f"""
### Processing Applied

✅ Noise Reduction

✅ CLAHE Contrast Enhancement

✅ Edge Sharpening


The image visibility has been improved.
"""
)



    with r2:


        st.success(
f"""
### Image Analysis


Original Brightness:
{report["Original Brightness"]}


Enhanced Brightness:
{report["Enhanced Brightness"]}


Contrast Improvement:
{report["Contrast Improvement"]} %

"""
)



    # =====================
    # METRICS
    # =====================


    mse=calculate_mse(
        gray,
        enhanced
    )


    psnr=calculate_psnr(
        gray,
        enhanced
    )


    ssim=calculate_ssim(
        gray,
        enhanced
    )




    st.markdown(
"""
<div class="section">
📊 Quality Metrics
</div>
""",
unsafe_allow_html=True
)



    m1,m2,m3=st.columns(3)



    with m1:

        st.markdown(

f"""
<div class="card">

<div class="metric-title">
MSE
</div>

<div class="metric-value">

{mse:.2f}

</div>

</div>
""",

unsafe_allow_html=True

)




    with m2:


        st.markdown(

f"""
<div class="card">

<div class="metric-title">
PSNR
</div>

<div class="metric-value">

{psnr:.2f} dB

</div>

</div>
""",

unsafe_allow_html=True

)




    with m3:


        st.markdown(

f"""
<div class="card">

<div class="metric-title">
SSIM
</div>

<div class="metric-value">

{ssim:.4f}

</div>

</div>
""",

unsafe_allow_html=True

)




    # =====================
    # DOWNLOAD
    # =====================


    os.makedirs(
        "output",
        exist_ok=True
    )


    output_path="output/enhanced_xray.png"



    cv2.imwrite(
        output_path,
        enhanced
    )



    with open(
        output_path,
        "rb"
    ) as file:


        st.download_button(

            "⬇ Download Enhanced X-Ray",

            file,

            file_name="enhanced_xray.png",

            mime="image/png"

        )



else:


    st.info(
        "Upload an X-ray image to start"
    )