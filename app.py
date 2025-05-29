import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os
import datetime
from fpdf import FPDF
import base64

# モデル設定
MODEL_PATH = "arch_classifier_model.h5"
MODEL_VERSION = "final_streamlit_export"
GDRIVE_URL = "https://drive.google.com/uc?id=1-0jLv-ahm5Vs06Q7aXE3N4SS22R4HOAh"

# モデルダウンロード
if not os.path.exists(MODEL_PATH):
    st.warning("📦 モデルファイルが見つからないため、ダウンロードを開始します…")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# モデル読み込み
try:
    model = load_model(MODEL_PATH, compile=False)
    modified_date = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y/%m/%d %H:%M')
    st.caption(f"🧠 使用モデル：{MODEL_VERSION}（更新日時：{modified_date}）")
except Exception as e:
    st.error(f"モデル読み込みエラー：{e}")
    st.stop()

# アプリUI
st.title("足型インソール診断アプリ（PDF出力対応）")
leg_shape = st.radio("脚の形状を選んでください", ["O脚", "X脚", "正常"])
has_bunion = st.radio("外反母趾の有無", ["あり", "なし"])
uploaded_file = st.file_uploader("足裏画像をアップロードしてください（.jpg/.png）", type=["jpg", "jpeg", "png"])

# 説明辞書
arch_descriptions = {
    "Flat": "偏平足は土踏まずが低下または消失し、足裏全体が地面に接している状態です。本来、土踏まずは歩行時の衝撃を吸収する役割を持っていますが、それが機能しにくくなるため、足の疲れやすさ、足裏の痛み、膝や腰への負担増加といったトラブルが起こりやすくなります。",
    "High": "ハイアーチは土踏まずが通常より高く、足裏の接地面が少ない状態です。このため、衝撃が集中しやすく、足裏や膝に痛みが出やすい傾向があります。",
    "Normal": "正常足は土踏まずが適度に形成され、衝撃吸収と安定性のバランスが取れている理想的な形です。"
}

leg_descriptions = {
    "O脚": "O脚は、両膝がつかず脚が外側に湾曲している状態で、膝や股関節に負担がかかりやすい傾向があります。",
    "X脚": "X脚は、膝がくっつく一方で足首が離れてしまう状態で、歩行時に膝や足首に負担がかかることがあります。",
    "正常": "正常脚は、太もも・膝・ふくらはぎ・くるぶしが自然に接する理想的な脚の形です。"
}

bunion_description = "外反母趾は、母趾が外側に曲がり付け根が内側に突出する症状で、早期対策が重要です。"

# ラベル辞書
label_map = {0: "High", 1: "Normal", 2: "Flat"}

# 分析
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    resized = image.resize((224, 224))
    array = img_to_array(resized) / 255.0
    input_data = np.expand_dims(array, axis=0)

    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)
    arch_label = label_map[predicted_index]

    st.markdown(f"### 🧠 AI診断結果：**{arch_label}**")
    st.success(f"🦶 パターンID：**{arch_label}-{leg_shape}-{has_bunion}**")

    # 解説表示
    st.subheader("📝 解説")
    st.markdown(f"**アーチタイプ**：{arch_label}  \n{arch_descriptions.get(arch_label, '')}")
    st.markdown(f"**脚の形状**：{leg_shape}  \n{leg_descriptions.get(leg_shape, '')}")
    if has_bunion == "あり":
        st.markdown(f"**外反母趾**：あり  \n{bunion_description}")

    # PDF生成
    if st.button("📄 PDFで診断結果を出力"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="足型インソール診断結果", ln=1, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"アーチタイプ：{arch_label}", ln=1)
        pdf.multi_cell(0, 10, txt=arch_descriptions.get(arch_label, ""))
        pdf.ln(5)

        pdf.cell(200, 10, txt=f"脚の形状：{leg_shape}", ln=1)
        pdf.multi_cell(0, 10, txt=leg_descriptions.get(leg_shape, ""))
        pdf.ln(5)

        if has_bunion == "あり":
            pdf.cell(200, 10, txt="外反母趾：あり", ln=1)
            pdf.multi_cell(0, 10, txt=bunion_description)
        else:
            pdf.cell(200, 10, txt="外反母趾：なし", ln=1)

        # 保存
        pdf_path = "/tmp/diagnosis_result.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
            pdf_link = f'<a href="data:application/pdf;base64,{b64_pdf}" download="足型診断結果.pdf">📥 PDFをダウンロード</a>'
            st.markdown(pdf_link, unsafe_allow_html=True)
