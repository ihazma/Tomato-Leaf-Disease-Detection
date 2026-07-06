import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("G:/Project/Tomato_Disease_Prediction/model/best_model.keras")


model = load_model()

# =====================================================
# CLASS NAMES
# =====================================================

class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy Leaf"
]

# =====================================================
# DISEASE INFORMATION
# =====================================================

disease_info = {
    "Bacterial Spot":
        "A bacterial disease causing dark spots on leaves and fruits. Management includes using disease-free seeds and copper-based bactericides.",

    "Early Blight":
        "A fungal disease characterized by brown concentric rings on older leaves. Remove infected foliage and apply appropriate fungicides.",

    "Late Blight":
        "A severe disease caused by Phytophthora infestans. It spreads rapidly under cool and humid conditions.",

    "Leaf Mold":
        "Common in humid environments and greenhouses. Improve ventilation and avoid excessive humidity.",

    "Septoria Leaf Spot":
        "Produces numerous small circular spots on leaves. Crop rotation and sanitation help reduce infection.",

    "Spider Mites":
        "Tiny pests that feed on leaf tissue, causing yellow speckling. Biological control or miticides can help.",

    "Target Spot":
        "A fungal disease producing circular lesions with concentric rings on leaves.",

    "Tomato Yellow Leaf Curl Virus":
        "A viral disease transmitted by whiteflies. Control whiteflies and use resistant varieties whenever possible.",

    "Tomato Mosaic Virus":
        "A viral infection causing mottled leaves and reduced fruit production. Prevent spread through clean tools and healthy seedlings.",

    "Healthy Leaf":
        "No disease symptoms detected. The tomato leaf appears healthy."
}

# =====================================================
# TITLE
# =====================================================

st.title("🍅 Tomato Leaf Disease Detection")

st.markdown(
"""
Upload a **tomato leaf image** and the model will identify the disease.

"""
)

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # ==========================================
    # PREPROCESS
    # ==========================================

    img = image.resize((224,224))

    img_array = np.array(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    # ==========================================
    # PREDICTION
    # ==========================================

    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)

    confidence = float(predictions[predicted_index] * 100)

    predicted_class = class_names[predicted_index]

    # ==========================================
    # RESULT
    # ==========================================

    with col2:

        st.success(f"### Prediction")

        st.markdown(f"# {predicted_class}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.write("### Disease Information")

        st.info(disease_info[predicted_class])

    # ==========================================
    # TOP 3
    # ==========================================

    st.divider()

    st.subheader("Top 3 Predictions")

    top3 = np.argsort(predictions)[::-1][:3]

    df = pd.DataFrame({
        "Disease": [class_names[i] for i in top3],
        "Confidence (%)": [round(predictions[i]*100,2) for i in top3]
    })

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

    st.write("### Confidence Distribution")

    for idx in top3:

        st.write(f"**{class_names[idx]}**")

        st.progress(float(predictions[idx]))

        st.write(f"{predictions[idx]*100:.2f}%")

    # ==========================================
    # WARNING
    # ==========================================

    st.divider()

    if confidence < 70:

        st.warning(
            """
### Low Confidence Prediction

The uploaded image may differ from the images used during training.

Possible reasons:

- Complex background
- Poor lighting
- Different camera angle
- Leaf occupies only a small portion of the image
- Disease not present in the training dataset
"""
        )

    elif confidence < 90:

        st.info(
            """
### Moderate Confidence Prediction

The model is reasonably confident, but verifying the diagnosis with additional images or agricultural resources is recommended.
"""
        )

    else:

        st.success(
            """
### High Confidence Prediction

The model is highly confident in this prediction.
"""
        )
