import ollama
import streamlit as st

# Sayfa yapılandırması
st.set_page_config(
    page_title="Hızlı ve Kesin RAG Asistanı", page_icon="🎯", layout="centered"
)

st.title("🎯 Kesin ve Doğru RAG Asistanı")
st.write(
    "Yüklediğiniz metindeki bilgileri doğrudan alıntılayarak yanıt veren"
    " güncellenmiş sürüm."
)

# --- 1. KENAR ÇUBUĞU: BELGE YÜKLEME ---
with st.sidebar:
  st.header("📂 Belge Yönetimi")
  uploaded_file = st.file_uploader("Metin dosyası yükle (.txt)", type=["txt"])

  document_text = ""
  if uploaded_file is not None:
    try:
      document_text = uploaded_file.read().decode("utf-8")
      st.success("Belge başarıyla yüklendi!")
    except Exception as e:
      st.error(f"Dosya okunamadı: {e}")

# --- 2. SOHBET GEÇMİŞİ BAŞLATMA ---
if "messages" not in st.session_state:
  st.session_state.messages = []

# Eski mesajları ekrana yazdırma
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# --- 3. KULLANICI GİRDİSİ VE BİREBİR ALINTI MANTIĞI ---
if prompt := st.chat_input("Sorunuzu yazın..."):
  # Kullanıcı mesajını geçmişe ekle ve ekranda göster
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Modelin yorum yapmasını engelleyen ve birebir alıntı yapmasını zorunlu kılan komut
  if document_text:
    final_prompt = (
        "Sen bir metin arama robotusun. Aşağıdaki metni dikkatle oku.\n"
        "Soruya cevap verirken KESİNLİKLE kendi cümlelerini kurma ve yorum"
        " yapma.\n"
        "Sadece metinde geçen ilgili cümleyi veya bölümü birebir kopyala ve"
        " yanıt olarak ver.\n\n"
        f"METİN:\n{document_text}\n\n"
        f"SORU: {prompt}\n"
        "METİNDEKİ ORİJİNAL CEVAP:"
    )
  else:
    final_prompt = prompt

  with st.chat_message("assistant"):
    with st.spinner("Metinden alıntı yapılıyor..."):
      try:
        response = ollama.chat(
            model="gemma:2b",
            messages=[{"role": "user", "content": final_prompt}],
            options={
                "num_predict": 120,
                "temperature": 0.0,
            },  # Yaratıcılık tamamen kapalı (0.0)
        )
        full_response = response["message"]["content"]
      except Exception as e:
        full_response = (
            f"Hata oluştu (Lütfen Ollama'nın çalıştığından emin olun): {e}"
        )

    st.markdown(full_response)

  # Asistan yanıtını geçmişe ekle
  st.session_state.messages.append(
      {"role": "assistant", "content": full_response}
  )