import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fpdf import FPDF
import datetime

# 説明文データ（省略せず掲載）
arch_descriptions = {
    "Flat": "偏平足は土踏まずが低下または消失し、足裏全体が地面に接している状態です。本来、土踏まずは歩行時の衝撃を吸収する役割を持っていますが、それが機能しにくくなるため、足の疲れやすさ、足裏の痛み、膝や腰への負担増加といったトラブルが起こりやすくなります。また、外反母趾や内反小趾のリスクも高まります。長時間の立ち仕事や歩行で不調を感じることが多いため、土踏まずを支えるインソールや、足にフィットした靴選びが重要です。早めの対策が、将来的な関節トラブルの予防につながります。",
    "High": "ハイアーチは土踏まずが通常より高く、足裏の接地面が少ない状態です。このため、歩行や走行時の衝撃が一点に集中しやすく、足裏、かかと、膝、腰などに痛みを引き起こしやすい傾向があります。また、足の柔軟性が低下しがちで、バランスが不安定になりやすく、捻挫のリスクも増加します。クッション性のある靴や衝撃吸収性に優れたインソールを活用することで、負担を軽減し、快適な歩行が可能になります。日常的なストレッチや足のケアも予防につながります。",
    "Normal": "正常足は土踏まずが適度に形成され、足裏全体にバランスよく荷重がかかる理想的な形です。衝撃をしっかりと吸収し、膝や腰への負担も少なく、安定した歩行が可能です。トラブルが少ない一方で、加齢や体重増加、合わない靴の使用などにより形状が崩れることがあります。定期的な足のチェックと、自分の足に合った靴選びを続けることで、健康な足を維持できます。正常だからこそ油断せず、予防の意識を持つことが大切です。"
}

leg_descriptions = {
    "O脚": "O脚は、両足を揃えて立った際に膝がくっつかず、脚全体がアルファベットの「O」のように外側に湾曲している状態です。主に骨格のゆがみや筋力バランスの崩れ、座り方・歩き方の癖が原因とされます。見た目の問題だけでなく、膝や股関節、足首に過剰な負担がかかりやすく、変形性膝関節症や膝痛のリスクが高まります。",
    "X脚": "X脚は、膝が内側に寄って接触し、足首が離れてしまう状態で、脚の形がアルファベットの「X」に見えるのが特徴です。歩行時に膝の内側や足首に負担がかかりやすく、痛みや疲れ、将来的な関節障害の原因となることもあります。",
    "正常": "正常脚は、まっすぐに立ったときに太もも・膝・ふくらはぎ・くるぶしが自然に接する、バランスのとれた脚の状態です。体重が均等に分散され、膝や足首、腰などに無理な負荷がかかりにくいです。"
}

bunion_description = "外反母趾とは、足の親指（母趾）が外側に曲がり、付け根の関節が内側に突出して変形する症状です。適切な靴選びや、足指を広げる体操・インソールによるサポートで進行を防ぐことが可能です。"

# アプリUI
st.title("足型インソール診断アプリ（AI分類＋PDF出力）")
leg_shape = st.radio("脚の形状を選んでください", ["O脚", "X脚", "正常"])
has_bunion = st.radio("外反母趾の有無", ["あり", "なし"])
uploaded_file = st.file_uploader("足裏画像をアップロードしてください", type=["jpg", "jpeg", "png"])

try:
    model = load_model("arch_classifier_model.h5")
except Exception as e:
    st.error(f"モデル読み込みエラー：{e}")
    st.stop()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"画像読み込み失敗：{e}")
        st.stop()

    st.image(image, caption="アップロード画像", use_column_width=True)

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

    # 説明文の表示
    arch_text = arch_descriptions.get(arch_label, "")
    leg_text = leg_descriptions.get(leg_shape, "")
    bunion_text = bunion_description if has_bunion == "あり" else ""

    st.subheader("🦶 足アーチの診断説明")
    st.write(arch_text)

    st.subheader("🦵 脚型の診断説明")
    st.write(leg_text)

    if bunion_text:
        st.subheader("👣 外反母趾について")
        st.write(bunion_text)

  # PDF出力（日本語対応）
if st.button("📄 診断結果をPDFでダウンロード"):
    pdf = FPDF()
    pdf.add_page()

    # 日本語フォント（IPAexゴシックなど）を指定
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
