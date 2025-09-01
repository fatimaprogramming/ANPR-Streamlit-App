import cv2
import pytesseract
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Configure Tesseract path (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🚗 Load Lottie Animations (Car + Bike)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_car = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")   # Car driving
lottie_bike = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_puciaact.json")  # Bike rider

# 🎉 Confetti Effect with JS
def show_confetti():
    st.markdown(
        """
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <script>
        function throwConfetti() {
            confetti({
                particleCount: 200,
                spread: 140,
                origin: { y: 0.8 }
            });
        }
        throwConfetti();
        </script>
        """,
        unsafe_allow_html=True
    )

# 🌸 Custom CSS for Black Theme + Floating Cute Emojis
def local_css():
    st.markdown(
        """
        <style>
        /* Black background everywhere */
        .stApp {
            background-color: #000000 !important;
        }
        .st-emotion-cache-18ni7ap, .st-emotion-cache-1dp5vir, .st-emotion-cache-1wrcr25 {
            background-color: #000000 !important;
        }
        .main {
            background-color: #000000 !important;
            color: #00ffcc;
        }
        h1, h2, h3, h4, h5 {
            color: #00ffcc !important;
            text-align: center;
            font-family: "Courier New", monospace;
        }
        .stTextInput, .stFileUploader, .stButton button {
            background-color: #111111 !important;
            color: #00ffcc !important;
            border: 1px solid #00ffcc !important;
            border-radius: 10px;
        }

        /* Floating Emoji Animation */
        @keyframes floatUp {
            0% { transform: translateY(0) scale(1); opacity: 1; }
            100% { transform: translateY(-800px) scale(1.3); opacity: 0; }
        }
        .floating-container {
            position: fixed;
            bottom: -50px;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
            z-index: 9999;
        }
        .emoji {
            position: absolute;
            bottom: 0;
            font-size: 28px;
            animation: floatUp 14s linear infinite;
            opacity: 0.8;
        }
        .emoji:nth-child(1) { left: 10%; animation-duration: 12s; }
        .emoji:nth-child(2) { left: 25%; animation-duration: 16s; }
        .emoji:nth-child(3) { left: 45%; animation-duration: 14s; }
        .emoji:nth-child(4) { left: 65%; animation-duration: 18s; }
        .emoji:nth-child(5) { left: 80%; animation-duration: 13s; }
        .emoji:nth-child(6) { left: 95%; animation-duration: 15s; }
        </style>
        """,
        unsafe_allow_html=True
    )

# 🚦 Main App
def main():
    st.set_page_config(page_title="ANPR Black Edition 🚗🏍️", page_icon="🌸", layout="centered")
    local_css()

    # Floating Emojis (Flowers, Hearts, Sparkles)
    st.markdown(
        """
        <div class="floating-container">
            <div class="emoji">🌸</div>
            <div class="emoji">💖</div>
            <div class="emoji">✨</div>
            <div class="emoji">🌼</div>
            <div class="emoji">🌺</div>
            <div class="emoji">🌟</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("🚦 Automatic Number Plate Recognition (ANPR) 🌸💖")

    # Show both animations
    st_lottie(lottie_car, height=200, key="car")
    st_lottie(lottie_bike, height=200, key="bike")

    st.write("Upload a **car/bike image** and I’ll detect the number plate 🚗✨")

    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption="Uploaded Vehicle", use_container_width=True)

        if st.button("🔍 Detect Number Plate"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)

            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            NumberPlateCnt = None

            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                if len(approx) == 4:
                    NumberPlateCnt = approx
                    break

            if NumberPlateCnt is not None:
                cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(NumberPlateCnt)
                plate = gray[y:y+h, x:x+w]

                text = pytesseract.image_to_string(plate, config='--psm 8')
                st.success(f"🎉 License Plate Detected: **{text.strip()}**")

                st.image(img, channels="BGR", caption="Detected Plate", use_container_width=True)

                # 🎊 Show Confetti
                show_confetti()
            else:
                st.error("⚠️ No license plate detected. Try another image!")

if __name__ == "__main__":
    main()
