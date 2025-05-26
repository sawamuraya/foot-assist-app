import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import datetime

# アプリのタイトル（← これを先に書く）
st.title("足型インソール診断アプリ（AI画像分類つき）")

# モデルファイル情報
MODEL_PATH = "arch_classifier_model.h5"
MODEL_VERSION = "arch_classifier_model_v3"
MODEL_UPDATE_DATE = "2025/05/26"

# モデル更新日時（自動取得）
if os.path.exists(MODEL_PATH):
    timestamp = os.path.getmtime(MODEL_PATH)
    modified_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d %H:%M')
    st.caption(f"🧠 使用モデル：{MODEL_VERSION}（更新日時：{modified_date}）")
else:
    st.caption("⚠️ モデルファイルが見つかりません")

# ユーザー入力
leg_shape = st.radio("脚の形状を選んでください", ["O脚", "X脚", "正常"])
has_bunion = st.radio("外反母趾の有無", ["あり", "なし"])
uploaded_file = st.file_uploader("足裏画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# モデル読み込み
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"モデル読み込みエラー：{e}")
    st.stop()

# 推論処理
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"画像読み込みエラー：{e}")
        st.stop()

    st.image(image, caption="アップロードした足裏画像", use_column_width=True)

    resized = image.resize((224, 224))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label_map = {0: "High", 1: "Normal", 2: "Flat"}
    arch_label = label_map[np.argmax(prediction)]

    def get_pattern_id(arch, leg, bunion):
        arch_map = {"Flat": 0, "High": 1, "Bunion": 2, "Normal": 3}
        leg_map = {"O脚": 0, "X脚": 1, "正常": 2}
        if bunion == "あり":
            arch = "Bunion"
        return arch_map.get(arch, 3) * 3 + leg_map.get(leg, 2) + 1

    pattern_id = get_pattern_id(arch_label, leg_shape, has_bunion)

    st.markdown(f"### 🧠 AI診断結果：**{arch_label}**")
    st.success(f"🦶 あなたの足型分類パターンID：**{pattern_id} / 12**")
    st.info(f"このタイプにおすすめのインソール：**インソール{pattern_id}番** をお試しください！")

    st.write(leg_text)

    if bunion_text:
        st.subheader("👣 外反母趾について")
        st.write(bunion_text)

    # PDF出力（日本語フォント）
    if st.button("📄 診断結果をPDFでダウンロード"):
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('IPAexG', '', 'ipaexg.ttf', uni=True)
        pdf.set_font("IPAexG", size=12)

        pdf.cell(200, 10, txt="足型AI診断結果", ln=True, align="C")
        pdf.cell(200, 10, txt=f"診断日: {datetime.date.today()}", ln=True)
        pdf.ln(5)
        pdf.cell(200, 10, txt=f"AI診断アーチ分類：{arch_label}", ln=True)
        pdf.cell(200, 10, txt=f"脚の形状：{leg_shape}", ln=True)
        pdf.cell(200, 10, txt=f"外反母趾：{has_bunion}", ln=True)
        pdf.cell(200, 10, txt=f"分類パターンID：{pattern_id} / 12", ln=True)
        pdf.ln(8)
        pdf.multi_cell(0, 8, f"[アーチ説明]\n{arch_text}")
        pdf.ln(4)
        pdf.multi_cell(0, 8, f"[脚型説明]\n{leg_text}")
        if bunion_text:
            pdf.ln(4)
            pdf.multi_cell(0, 8, f"[外反母趾説明]\n{bunion_text}")

        pdf_output = "diagnosis_result.pdf"
        pdf.output(pdf_output)

        with open(pdf_output, "rb") as f:
            st.download_button(
                label="📥 PDFをダウンロードする",
                data=f,
                file_name=pdf_output,
                mime="application/pdf"
            )
