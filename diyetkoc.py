import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import streamlit as st

# Sayfa yapılandırması
st.set_page_config(
    page_title="Sağlıklı Yaşam & Diyet RAG Asistanı",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 Evrensel Sağlıklı Yaşam ve Diyet Koçu")

# --- CHROMA DB BAŞLATMA ---
if "chroma_client" not in st.session_state:
  st.session_state.chroma_client = chromadb.Client()
  st.session_state.collection = (
      st.session_state.chroma_client.get_or_create_collection(
          name="saglik_rag_koleksiyonu"
      )
  )

# --- ÇOKLU SOHBET (GEÇMİŞ) YÖNETİMİ ---
if "sessions" not in st.session_state:
  st.session_state.sessions = {
      "Oturum 1": [{
          "role": "assistant",
          "content": (
              "Merhaba! Ben diyet ve sağlıklı yaşam koçunuzum. Size nasıl"
              " yardımcı olabilirim?"
          ),
      }]
  }
  st.session_state.current_session = "Oturum 1"

# --- 40+ RESMİ BESİN VERİ TABANI (100 gramdaki net protein değerleri) ---
BESIN_VERITABANI = {
    "yumurta": {"kal": 78, "pro": 12.0, "karb": 1.2, "lif": 0.0},  # 100g için
    "tavuk": {"kal": 165, "pro": 31.0, "karb": 0.0, "lif": 0.0},
    "kıyma": {"kal": 215, "pro": 26.0, "karb": 0.0, "lif": 0.0},
    "hindi": {"kal": 150, "pro": 30.0, "karb": 0.0, "lif": 0.0},
    "ton balığı": {"kal": 190, "pro": 29.0, "karb": 0.0, "lif": 0.0},
    "somon": {"kal": 206, "pro": 22.0, "karb": 0.0, "lif": 0.0},
    "et": {"kal": 250, "pro": 28.0, "karb": 0.0, "lif": 0.0},
    "köfte": {"kal": 240, "pro": 20.0, "karb": 4.0, "lif": 0.5},
    "süzme yoğurt": {"kal": 130, "pro": 6.0, "karb": 4.0, "lif": 0.0},
    "yoğurt": {"kal": 120, "pro": 4.0, "karb": 5.0, "lif": 0.0},
    "süt": {"kal": 100, "pro": 3.25, "karb": 4.75, "lif": 0.0},
    "lor": {"kal": 98, "pro": 18.0, "karb": 3.5, "lif": 0.0},
    "peynir": {"kal": 90, "pro": 16.6, "karb": 1.6, "lif": 0.0},
    "kaşar": {"kal": 115, "pro": 23.3, "karb": 1.6, "lif": 0.0},
    "yulaf": {"kal": 150, "pro": 12.5, "karb": 67.5, "lif": 10.0},
    "tam buğday ekmek": {"kal": 70, "pro": 12.0, "karb": 48.0, "lif": 8.0},
    "ekmek": {"kal": 65, "pro": 8.0, "karb": 52.0, "lif": 2.4},
    "pirinç": {"kal": 130, "pro": 2.5, "karb": 28.0, "lif": 0.4},
    "bulgur": {"kal": 140, "pro": 4.5, "karb": 25.0, "lif": 4.5},
    "makarna": {"kal": 130, "pro": 4.0, "karb": 25.0, "lif": 1.5},
    "mercimek": {"kal": 116, "pro": 9.0, "karb": 20.0, "lif": 8.0},
    "nohut": {"kal": 164, "pro": 9.0, "karb": 27.0, "lif": 7.0},
    "mung": {"kal": 105, "pro": 7.0, "karb": 19.0, "lif": 7.5},
    "kuru fasulye": {"kal": 140, "pro": 9.0, "karb": 25.0, "lif": 6.0},
    "domates": {"kal": 18, "pro": 0.9, "karb": 3.9, "lif": 1.2},
    "salatalık": {"kal": 15, "pro": 0.7, "karb": 3.6, "lif": 0.5},
    "salata": {"kal": 20, "pro": 1.5, "karb": 3.0, "lif": 2.0},
    "brokoli": {"kal": 35, "pro": 2.4, "karb": 7.0, "lif": 3.3},
    "kabak": {"kal": 45, "pro": 1.5, "karb": 6.0, "lif": 2.0},
    "patates": {"kal": 87, "pro": 2.0, "karb": 20.0, "lif": 1.8},
    "chia": {"kal": 48, "pro": 17.0, "karb": 42.0, "lif": 34.0},
    "keten tohumu": {"kal": 53, "pro": 18.0, "karb": 29.0, "lif": 28.0},
    "zeytinyağı": {"kal": 90, "pro": 0.0, "karb": 0.0, "lif": 0.0},
    "badem": {"kal": 180, "pro": 20.0, "karb": 20.0, "lif": 11.6},
    "fındık": {"kal": 190, "pro": 15.0, "karb": 16.6, "lif": 10.0},
    "ceviz": {"kal": 200, "pro": 15.0, "karb": 13.3, "lif": 6.6},
    "elma": {"kal": 80, "pro": 0.26, "karb": 14.0, "lif": 2.4},
    "muz": {"kal": 105, "pro": 1.08, "karb": 22.5, "lif": 2.5},
    "çilek": {"kal": 40, "pro": 0.7, "karb": 9.0, "lif": 2.4},
    "fıstık ezmesi": {"kal": 95, "pro": 26.6, "karb": 20.0, "lif": 8.0},
}

# --- ANA SEKMELER (TABS) ---
sekme_asistan, sekme_kalori_takip = st.tabs(
    ["💬 Yapay Zeka Koç & RAG", "🍎 Akıllı Günlük Kalori ve Makro Çubuk Takibi"]
)

# ==========================================
# 1. SEKME: YAPAY ZEKA KOÇ & RAG
# ==========================================
with sekme_asistan:
  with st.sidebar:
    st.header("📂 Veri & Oturum Yönetimi")

    st.subheader("💬 Sohbet Geçmişi")
    session_names = list(st.session_state.sessions.keys())
    selected_session = st.selectbox(
        "Aktif Sohbeti Seç",
        session_names,
        index=session_names.index(st.session_state.current_session),
    )
    st.session_state.current_session = selected_session

    if st.button("➕ Yeni Sohbet Başlat", use_container_width=True):
      new_name = f"Oturum {len(st.session_state.sessions) + 1}"
      st.session_state.sessions[new_name] = [{
          "role": "assistant",
          "content": (
              "Yeni oturum açıldı! Sağlık, diyet veya kalori hakkında ne"
              " konuşmak istersiniz?"
          ),
      }]
      st.session_state.current_session = new_name
      st.rerun()

    if st.button("🗑️ Mevcut Sohbeti Temizle", use_container_width=True):
      st.session_state.sessions[st.session_state.current_session] = [{
          "role": "assistant",
          "content": "Sohbet temizlendi. Nasıl yardımcı olabilirim?",
      }]
      st.rerun()

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Sağlık/Diyet Belgesi Yükle (.txt)", type=["txt"]
    )

    if uploaded_file is not None:
      try:
        bytes_data = uploaded_file.getvalue()
        document_text = bytes_data.decode("utf-8")

        if document_text.strip():
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=250, chunk_overlap=30
          )
          chunks = text_splitter.split_text(document_text)

          for i, chunk in enumerate(chunks):
            st.session_state.collection.add(
                documents=[chunk], ids=[f"chunk_{i}_{uploaded_file.name}"]
            )

          st.success(
              f"✅ Başarılı! Toplam {len(chunks)} parça veritabanına işlendi."
          )
        else:
          st.warning("Yüklediğiniz dosya boş görünüyor.")
      except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")

  st.subheader("💡 Hızlı Soru Önerileri")
  col1, col2, col3, col4 = st.columns(4)

  selected_quick_prompt = None
  with col1:
    if st.button("🥩 20g/30g Protein Tarif"):
      selected_quick_prompt = (
          "20 gram veya 30 gram protein içeren, besin isimleri ve net"
          " gramajları açıkça yazılmış bir akşam yemeği veya atıştırmalık"
          " tarifi ver."
      )
  with col2:
    if st.button("💧 Su ve Lifin Önemi"):
      selected_quick_prompt = (
          "Günlük su tüketimi ve lifli beslenmenin sindirime etkisi nedir?"
      )
  with col3:
    if st.button("🌙 Akşam Öğünü Kuralları"):
      selected_quick_prompt = (
          "Akşam öğünlerinde karbonhidrat tüketimi nasıl olmalıdır?"
      )
  with col4:
    if st.button("⚖️ Kalori Açığı Nedir?"):
      selected_quick_prompt = (
          "Kalori açığı nedir ve yağ yakımı için neden gereklidir?"
      )

  st.markdown("---")

  current_messages = st.session_state.sessions[
      st.session_state.current_session
  ]

  for message in current_messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  prompt = st.chat_input(
      "Sağlık, diyet veya tarifler hakkında bir şeyler sorun..."
  )

  if selected_quick_prompt:
    prompt = selected_quick_prompt

  if prompt:
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    # Otomatik net porsiyon hesaplaması ekleme (20g veya 30g hedefi algılama)
    hedef_prot = 30  # Varsayılan
    if "20 gram" in prompt or "20g" in prompt:
      hedef_prot = 20
    elif "30 gram" in prompt or "30g" in prompt:
      hedef_prot = 30

    # Python ile veritabanından hedef gramajları hesapla
    hesaplanan_tarifler = []
    for besin_adi, veriler in BESIN_VERITABANI.items():
      if veriler["pro"] > 0:
        gereken_gram = (hedef_prot / veriler["pro"]) * 100
        hesaplanan_tarifler.append(
            f"- {besin_adi.capitalize()}: Tam {hedef_prot} gram protein için"
            f" **{int(gereken_gram)} gram** tüketmelisin."
        )

    # Örnek ilk 8 popüler seçeneği prompta ekleyelim ki koç bunları kullansın
    ornek_hesaplamalar = "\n".join(hesaplanan_tarifler[:10])

    final_prompt = (
        "Sen profesyonel bir diyet ve sağlıklı yaşam koçusun.\n"
        f"Kullanıcı net olarak {hedef_prot} gram protein içeren besinleri ve"
        " gramajlarını istiyor.\n"
        "Lütfen aşağıdaki kesin matematiksel hesaplamaları kullanarak net"
        " porsiyon gramajlarıyla birlikte güzel bir akşam yemeği veya"
        " atıştırmalık tarifi ver. Asla yanlış ondalıklı orantılar kurma, gerçek"
        " gramajları yaz:\n\n"
        f"HESAPLANMIŞ GERÇEK GRAMAJLAR:\n{ornek_hesaplamalar}\n\n"
        f"SORU: {prompt}\n"
        "KOÇUN YANITI:"
    )

    with st.chat_message("assistant"):
      with st.spinner("Koç yanıt hazırlıyor..."):
        try:
          response = ollama.chat(
              model="gemma:2b",
              messages=[{"role": "user", "content": final_prompt}],
              options={
                  "num_predict": 250,
                  "temperature": 0.1,
              },
          )
          full_response = response["message"]["content"]
        except Exception as e:
          full_response = f"Bağlantı Hatası: {e}"

      st.markdown(full_response)

    current_messages.append({"role": "assistant", "content": full_response})

# ==========================================
# 2. SEKME: AKILLI GÜNLÜK KALORİ VE MAKRO ÇUBUK TAKİBİ
# ==========================================
with sekme_kalori_takip:
  st.header("🍎 Günlük Kalori ve Makro İlerleme Paneli")
  st.write(
      "Önce kişisel bilgilerine göre hedeflerini hesapla, ardından gün içinde"
      " yediklerini girerek ilerleme çubuklarını görüntüle."
  )

  col_profil, col_analiz = st.columns([1, 1])

  with col_profil:
    st.subheader("1️⃣ Kişisel Hedef Belirleme")
    with st.form("Akilli_hedef_form"):
      k_kilo = st.number_input(
          "Kilo (kg)", min_value=30.0, max_value=200.0, value=60.0
      )
      k_boy = st.number_input(
          "Boy (cm)", min_value=120.0, max_value=220.0, value=165.0
      )
      k_yas = st.number_input("Yaş", min_value=10, max_value=100, value=20)
      k_hedef = st.selectbox(
          "Hedefin", ["Yağ Yakımı / Kilo Ver", "Korumak", "Kas Kazanımı / Kilo Al"]
      )
      hedef_hesapla_btn = st.form_submit_button("Hedefimi Hesapla")

      if hedef_hesapla_btn:
        bmh = 10 * k_kilo + 6.25 * k_boy - 5 * k_yas - 161
        if "Yağ Yakımı" in k_hedef:
          hedef_k = int(bmh * 1.2 - 400)
          hedef_p = int(k_kilo * 2.0)
          hedef_c = int((hedef_k * 0.4) / 4)
          hedef_l = int((hedef_k * 0.25) / 9)
        elif "Kas Kazanımı" in k_hedef:
          hedef_k = int(bmh * 1.3 + 300)
          hedef_p = int(k_kilo * 2.2)
          hedef_c = int((hedef_k * 0.5) / 4)
          hedef_l = int((hedef_k * 0.2) / 9)
        else:
          hedef_k = int(bmh * 1.2)
          hedef_p = int(k_kilo * 1.6)
          hedef_c = int((hedef_k * 0.45) / 4)
          hedef_l = int((hedef_k * 0.25) / 9)

        st.session_state.hedef_kalori = hedef_k
        st.session_state.hedef_protein = hedef_p
        st.session_state.hedef_karb = hedef_c
        st.session_state.hedef_lif = max(25, int(hedef_k / 100))

        st.success(
            f"🎯 **Hedef Kalori:** {hedef_k} kcal | **Protein:** {hedef_p}g"
            f" | **Karb:** {hedef_c}g | **Lif:** {st.session_state.hedef_lif}g"
        )

  with col_analiz:
    st.subheader("2️⃣ Günlük Öğün Girişi")

    if "hedef_kalori" in st.session_state:
      st.info("Hedefleriniz yüklendi. Aşağıdan öğünlerinizi girebilirsiniz.")
    else:
      st.warning(
          "⚠️ Lütfen önce sol taraftan kişisel hedefinizi hesaplayın."
      )

    kahvalti = st.text_input(
        "Kahvaltı", placeholder="Örn: 2 yumurta, 1 dilim ekmek"
    )
    ogle = st.text_input(
        "Öğle Yemeği", placeholder="Örn: 1 porsiyon tavuk göğsü, salata"
    )
    aksam = st.text_input(
        "Akşam Yemeği", placeholder="Örn: Sebze yemeği, yoğurt"
    )
    atistirmalik = st.text_input(
        "Atıştırmalık", placeholder="Örn: 1 avuç badem, yeşil çay"
    )

    analiz_et_btn = st.button("İlerleme Çubuklarını Hesapla")

    if analiz_et_btn:
      if "hedef_kalori" not in st.session_state:
        st.error("Önce soldan hedeflerinizi hesaplamalısınız!")
      else:
        tum_metin = (
            f"{kahvalti} {ogle} {aksam} {atistirmalik}".lower()
        )

        t_kal, t_pro, t_kar, t_lif = 0.0, 0.0, 0.0, 0.0

        for anahtar, besin in BESIN_VERITABANI.items():
          if anahtar in tum_metin:
            adet = tum_metin.count(anahtar)
            t_kal += besin["kal"] * adet
            t_pro += (besin["pro"] / 100) * 100 * adet  # Basit oranlama
            t_kar += (besin["karb"] / 100) * 100 * adet
            t_lif += (besin["lif"] / 100) * 100 * adet

        st.markdown("### 📊 Günlük İlerleme Durumu")

        h_kal = st.session_state.hedef_kalori
        oran_kal = min(float(t_kal / h_kal), 1.0) if h_kal > 0 else 0.0
        st.write(
            f"🔥 **Kalori**: {int(t_kal)} kcal / {int(h_kal)} kcal"
        )
        st.progress(oran_kal)

        h_pro = st.session_state.hedef_protein
        oran_pro = min(float(t_pro / h_pro), 1.0) if h_pro > 0 else 0.0
        st.write(
            f"🥩 **Protein**: {int(t_pro)}g / {int(h_pro)}g"
        )
        st.progress(oran_pro)

        h_kar = st.session_state.hedef_karb
        oran_kar = min(float(t_kar / h_kar), 1.0) if h_kar > 0 else 0.0
        st.write(
            f"🍞 **Karbonhidrat**: {int(t_kar)}g / {int(h_kar)}g"
        )
        st.progress(oran_kar)

        h_lif = st.session_state.hedef_lif
        oran_lif = min(float(t_lif / h_lif), 1.0) if h_lif > 0 else 0.0
        st.write(f"🌾 **Lif**: {int(t_lif)}g / {int(h_lif)}g")
        st.progress(oran_lif)