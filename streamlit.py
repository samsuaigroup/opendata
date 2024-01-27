import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
from predict_func import image_predict
import os

st.title("Ekinlarning kasalliklarni va ularga tushgan zararkunanda hashoratlarni aniqlaydi!")

st.markdown('''
    Bu sizning ekinlaringizni kasalliklarini aniqlaydi 
    Ishlatish uchun rasm yuklang natijalarini ko'ring.
''')

file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'wpeg','jpg'])

if file:
    local_image_path = f"uploaded_image.{file.name.split('.')[-1]}"
    with open(local_image_path, "wb") as f:
        f.write(file.getbuffer())

    # Read the uploaded image as a PIL image
    image = Image.open(local_image_path)

    # Display the image
    st.image(image)

    # Run the prediction and get results
    natija = image_predict(local_image_path)

    st.success(f"Kasallik: {natija['kasallik']}")
    st.info(f"ehtimolligi: {natija['ehtimolligi']}")
    st.write(f"maslahat : {natija['tashxis']}\n{natija['qushimcha']}")

    delete = st.button("rasmni o'chirish")
    if delete:
        os.remove(local_image_path)
        st.info("Rasm va natijalar muvaffaqiyatli o'chirildi.")
