import streamlit as st
from model_helper import predict
import base64




# Centered car GIF at the top
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://www.gifcen.com/wp-content/uploads/2021/05/car-gif-7.gif' width='400'>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h1 style='text-align: center; color: #FF5733;'>
        🚗 Car Damage Detection <span style='color:#33C3FF;'>AI</span> 🔍
    </h1>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    #st.image(uploaded_file, caption="Uploaded File", use_container_width=True)
    # Read uploaded file as bytes
    file_bytes = uploaded_file.read()

    # Encode to base64
    img_str = base64.b64encode(file_bytes).decode()
    st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: center;
        margin: 20px 0;
    ">
        <div style="
            border: 4px solid #0078ff;
            border-radius: 15px;
            padding: 10px;
            max-width: 1000px;
            background-color: rgba(255,255,255,0.85);
            box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        ">
            <img src="data:image/png;base64,{img_str}" style="width:100%; border-radius:10px;">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    prediction = predict(image_path)
    st.success(f"Prediction: {prediction}")
