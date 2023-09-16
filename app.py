import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

# 실습. 모델 불러오기
model = load_model('model.h5')

st.write('# 🚩숫자 인식기')
CANVAS_SIZE = 192

# col1, col2,col3 = st.beta_columns(3)
col1, col2 = st.columns(2)
col1.write('마우스로 숫자입력')
col2.write('예측한 숫자')
with col1:
    canvas = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas'
    )

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))

    
if st.button('Predict') : #st.button('Predict'):
    # ai 숫자 출력하기
    preview_img = cv2.resize(img, dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
    col2.image(preview_img)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = x.reshape((1, 28*28))
    preds = model.predict(x)
   

    # 웹앱에 출력하기
    st.subheader(f'result: {np.argmax(preds[0])}')
    #st.write(f'result: {np.argmax(preds[0])}')
    st.bar_chart(preds[0], color='#ffaa00')#use_container_width=True,
    
