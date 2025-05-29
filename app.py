import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import os
import datetime

# モデルファイルのパスとバージョン
MODEL_PATH = "arch_classifier_model.h5"
MODEL_VERSION = "final_streamlit_export"
GDRIVE_URL = "https://drive.google.com/uc?id=1-0jLv-ahm5Vs06Q7aXE3N4SS22R4HOAh"
MODEL_UPDATE_DATE = "2025/05/29"

# モデルファイルがなければ Google Drive からダウンロード
if not os.path.exists(MODEL_PATH):
    st.warning("📦 モデルファイルが見つからないため、ダウンロードを開始します…")
    try:
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    except Exception as e:
        st.error(f"モデル読み込みエラー：{e}")
        st.stop()

# モデル読み込み
try:
    model = load_model(MODEL_PATH, compile=False)
    timestamp = os.path.getmtime(MODEL_PATH)
    modified_date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d %H:%M')
    st.caption(f"🧠 使用モデル：{MODEL_VERSION}（更新日時：{modified_date}）")
except Exception as e:
    st.error(f"モデル読み込みエラー：{e}")
    st.stop()

# アプリタイトル
st.title("足型インソール診断アプリ（ResNetベースAI画像分類）")

# ユーザー入力
leg_shape = st.radio("脚の形状を選んでください", ["O脚", "X脚", "正常"])
has_bunion = st.radio("外反母趾の有無", ["あり", "なし"])

# 画像アップロード
uploaded_file = st.file_uploader("足裏画像をアップロードしてください（.jpg/.png）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"画像読み込みエラー：{e}")
        st.stop()

    st.image(image, caption="アップロードされた足裏画像", use_column_width=True)

    # 前処理
    image_resized = image.resize((224, 224))
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 予測
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)

    # ラベル定義
    label_map = {0: "High", 1: "Normal", 2: "Flat"}
    arch_label = label_map[predicted_index]

    st.markdown(f"### 🧠 AI診断結果：**{arch_label}**")

    # パターンID（12分類）
    def get_pattern_id(arch, leg, bunion):
        arch_map = {"Flat": 0, "High": 1, "Bunion": 2, "Normal": 3}
        leg_map = {"O脚": 0, "X脚": 1, "正常": 2}
        if bunion == "あり":
            arch = "Bunion"
        return arch_map[arch] * 3 + leg_map[leg] + 1

    pattern_id = get_pattern_id(arch_label, leg_shape, has_bunion)
    st.success(f"🦶 あなたの足型分類パターンID：**{pattern_id} / 12**")

    # 推奨インソール表示
    st.info(f"このタイプにおすすめのインソール：**インソール{pattern_id}番** をお試しください！")


