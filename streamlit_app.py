# streamlit_app.py
# Web Interface untuk E-commerce Chatbot menggunakan Streamlit

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time
import re
import random
from typing import Dict, List, Optional
import numpy as np

# Untuk ML models (jika ingin menggunakan ML-based chatbot)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Scikit-learn tidak tersedia. Hanya Rule-Based Chatbot yang dapat digunakan.")

class TextPreprocessor:
    """Kelas untuk preprocessing teks bahasa Indonesia"""
    
    def __init__(self):
        # Stopwords bahasa Indonesia (sample)
        self.stop_words = set([
            'yang', 'untuk', 'pada', 'ke', 'para', 'namun', 'menurut', 'antara', 
            'dia', 'dua', 'ia', 'seperti', 'jika', 'jadi', 'bahwa', 'atau',
            'dan', 'di', 'dari', 'ini', 'itu', 'dengan', 'adalah', 'ada',
            'akan', 'oleh', 'saya', 'kamu', 'kami', 'mereka', 'sudah', 'telah'
        ])
    
    def clean_text(self, text: str) -> str:
        """Membersihkan teks dari karakter tidak perlu"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenisasi dan filtering stopwords"""
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def preprocess(self, text: str) -> str:
        """Pipeline preprocessing lengkap"""
        text = self.clean_text(text)
        tokens = self.tokenize_and_filter(text)
        return ' '.join(tokens)

class SimpleRuleBasedChatbot:
    """Versi rule-based chatbot untuk Streamlit"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.rules = {
            'status_pesanan': [
                r'.*status.*pesanan.*', r'.*cek.*order.*', r'.*tracking.*',
                r'.*dimana.*pesanan.*', r'.*lacak.*', r'.*kapan.*sampai.*',
                r'.*order.*belum.*sampai.*'
            ],
            'retur_barang': [
                r'.*retur.*', r'.*kembalikan.*', r'.*refund.*',
                r'.*tukar.*barang.*', r'.*komplain.*', r'.*barang.*rusak.*',
                r'.*return.*', r'.*garansi.*'
            ],
            'promo_diskon': [
                r'.*promo.*', r'.*diskon.*', r'.*voucher.*',
                r'.*kupon.*', r'.*sale.*', r'.*cashback.*',
                r'.*potongan.*harga.*'
            ],
            'pembayaran': [
                r'.*cara.*bayar.*', r'.*metode.*pembayaran.*',
                r'.*payment.*', r'.*transfer.*', r'.*cod.*',
                r'.*ovo.*', r'.*gopay.*', r'.*dana.*'
            ],
            'pengiriman': [
                r'.*ongkir.*', r'.*pengiriman.*', r'.*kirim.*',
                r'.*ekspedisi.*', r'.*delivery.*', r'.*jne.*',
                r'.*jnt.*', r'.*sicepat.*'
            ],
            'informasi_umum': [
                r'.*jam.*buka.*', r'.*customer.*service.*', r'.*cs.*',
                r'.*kontak.*', r'.*alamat.*', r'.*cabang.*'
            ]
        }
        
        self.responses = {
            'status_pesanan': [
                "Untuk mengecek status pesanan, silakan masuk ke akun Anda dan pilih menu 'Pesanan Saya'. Atau berikan nomor pesanan untuk pengecekan lebih lanjut.",
                "Anda bisa melacak pesanan melalui halaman 'Riwayat Pesanan' di aplikasi kami. Pastikan Anda sudah login.",
                "Status pesanan dapat dicek dengan cara: Login → Menu Pesanan → Pilih pesanan yang ingin dicek."
            ],
            'retur_barang': [
                "Untuk retur barang, ajukan pengembalian melalui menu 'Retur' di halaman pesanan Anda. Pastikan barang dalam kondisi baik dan masih dalam periode retur (7-14 hari).",
                "Proses retur: Masuk ke akun → Pilih pesanan → Klik 'Ajukan Retur' → Isi alasan retur → Tunggu konfirmasi.",
                "Syarat retur: Barang belum digunakan, kemasan lengkap, dalam periode 14 hari sejak diterima."
            ],
            'promo_diskon': [
                "Promo terbaru dapat dilihat di halaman utama website atau aplikasi kami. Kami juga mengirim info promo via email dan notifikasi.",
                "Dapatkan voucher diskon melalui flash sale, member exclusive, atau program loyalitas kami.",
                "Ikuti sosial media kami untuk update promo terbaru: Flash Sale setiap Jumat, Member Sale setiap bulan!"
            ],
            'pembayaran': [
                "Kami menerima: Transfer Bank (BCA, Mandiri, BNI, BRI), E-wallet (OVO, GoPay, DANA), Kartu Kredit, dan COD untuk area tertentu.",
                "Pembayaran dapat dilakukan melalui virtual account, QRIS, atau COD. Pilih yang paling nyaman saat checkout.",
                "Metode pembayaran: Bank Transfer, E-wallet, Credit/Debit Card, COD (khusus Jakarta & sekitarnya)."
            ],
            'pengiriman': [
                "Kami bekerja sama dengan JNE, J&T, SiCepat, Pos Indonesia, dan GoSend untuk same-day delivery area tertentu.",
                "Ongkir dihitung berdasarkan berat barang dan jarak. Cek estimasi ongkir di halaman checkout.",
                "Estimasi pengiriman: 1-3 hari (Jabodetabek), 2-5 hari (Jawa), 3-7 hari (luar Jawa)."
            ],
            'informasi_umum': [
                "Customer Service kami tersedia 24/7 melalui live chat, email: cs@tokoonline.com, atau WhatsApp: 0811-1234-5678.",
                "Jam operasional: Senin-Jumat 08:00-17:00 WIB. Live chat tersedia 24 jam untuk bantuan cepat.",
                "Hubungi kami: Email cs@tokoonline.com, Phone 021-12345678, atau gunakan fitur live chat di website."
            ]
        }
        self.interaction_count = 0
    
    def get_response(self, user_input: str) -> tuple[str, str]:
        """Menghasilkan respons dan mendeteksi intent"""
        self.interaction_count += 1
        user_input_clean = self.preprocessor.preprocess(user_input)
        
        for intent, patterns in self.rules.items():
            for pattern in patterns:
                if re.search(pattern, user_input_clean):
                    response = random.choice(self.responses[intent])
                    return response, intent
        
        # Fallback response
        fallback_responses = [
            "Maaf, saya belum mengerti pertanyaan Anda. Apakah Anda menanyakan tentang status pesanan, retur barang, promo, pembayaran, atau pengiriman?",
            "Bisa tolong diperjelas pertanyaannya? Saya bisa membantu dengan: Status Pesanan, Retur, Promo, Pembayaran, atau Pengiriman.",
            "Mohon maaf, saya tidak memahami maksud Anda. Silakan pilih topik: Status Pesanan | Retur | Promo | Pembayaran | Pengiriman | Info Umum"
        ]
        return random.choice(fallback_responses), "unknown"

class SimpleMLChatbot:
    """Versi sederhana ML-based chatbot untuk Streamlit"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.is_trained = False
        self.intent_labels = ['status_pesanan', 'retur_barang', 'promo_diskon', 'pembayaran', 'pengiriman', 'informasi_umum']
        self.fallback_chatbot = SimpleRuleBasedChatbot()
        self.confidence_threshold = 0.6
        
    def generate_training_data(self):
        """Generate sample training data"""
        training_data = {
            'status_pesanan': [
                "dimana pesanan saya", "cek status order", "kapan barang sampai",
                "tracking pesanan", "pesanan belum sampai", "order sudah berapa lama",
                "status pengiriman", "update pesanan", "konfirmasi pesanan",
                "barang sudah dikirim belum", "nomor resi tracking"
            ],
            'retur_barang': [
                "mau retur barang", "gimana cara return", "barang rusak mau tukar",
                "refund pesanan", "komplain produk", "barang tidak sesuai",
                "mau kembalikan barang", "garansi produk", "tukar barang baru",
                "barang cacat", "return policy"
            ],
            'promo_diskon': [
                "ada promo apa", "voucher diskon", "sale kapan", "kupon gratis ongkir",
                "flash sale hari ini", "cashback berapa", "member discount",
                "promo weekend", "diskon spesial", "kode promo", "potongan harga"
            ],
            'pembayaran': [
                "cara bayar gimana", "payment method", "bisa cod tidak", "transfer kemana",
                "bayar pakai ovo", "cicilan kartu kredit", "virtual account",
                "qris payment", "gopay bisa tidak", "dana wallet", "bca mobile"
            ],
            'pengiriman': [
                "ongkir berapa", "ekspedisi apa saja", "same day delivery", "gratis ongkir",
                "lama pengiriman", "jne atau jnt", "kirim ke luar kota",
                "area pengiriman", "estimasi sampai", "sicepat tersedia"
            ],
            'informasi_umum': [
                "jam operasional", "customer service", "alamat toko", "kontak cs",
                "cara daftar member", "syarat ketentuan", "kebijakan privasi",
                "tentang toko", "cabang dimana saja", "nomor telepon"
            ]
        }
        
        texts = []
        labels = []
        for intent, samples in training_data.items():
            for sample in samples:
                texts.append(self.preprocessor.preprocess(sample))
                labels.append(intent)
        
        return texts, labels
    
    def train(self):
        """Melatih model ML"""
        if not ML_AVAILABLE:
            st.error("Scikit-learn tidak tersedia untuk ML model")
            return False
            
        try:
            texts, labels = self.generate_training_data()
            
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
            
            self.model.fit(texts, labels)
            self.is_trained = True
            return True
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False
    
    def predict_intent(self, text: str) -> tuple[str, float]:
        """Prediksi intent dengan confidence score"""
        if not self.is_trained:
            return "unknown", 0.0
        
        processed_text = self.preprocessor.preprocess(text)
        intent = self.model.predict([processed_text])[0]
        confidence = max(self.model.predict_proba([processed_text])[0])
        
        return intent, confidence
    
    def get_response(self, user_input: str) -> tuple[str, str, float]:
        """Generate response dengan ML prediction"""
        if not self.is_trained:
            response, intent = self.fallback_chatbot.get_response(user_input)
            return response, intent, 0.0
        
        try:
            intent, confidence = self.predict_intent(user_input)
            
            if confidence >= self.confidence_threshold:
                # Gunakan responses dari fallback chatbot berdasarkan predicted intent
                if intent in self.fallback_chatbot.responses:
                    response = random.choice(self.fallback_chatbot.responses[intent])
                    return response, intent, confidence
            
            # Fallback jika confidence rendah
            response, fallback_intent = self.fallback_chatbot.get_response(user_input)
            return response, fallback_intent, confidence
            
        except Exception as e:
            response, intent = self.fallback_chatbot.get_response(user_input)
            return response, intent, 0.0

def init_session_state():
    """Inisialisasi session state untuk Streamlit"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot_rule' not in st.session_state:
        st.session_state.chatbot_rule = SimpleRuleBasedChatbot()
    if 'chatbot_ml' not in st.session_state:
        st.session_state.chatbot_ml = SimpleMLChatbot()
        if ML_AVAILABLE:
            with st.spinner("Melatih ML Model..."):
                st.session_state.chatbot_ml.train()
    if 'total_interactions' not in st.session_state:
        st.session_state.total_interactions = 0
    if 'intent_stats' not in st.session_state:
        st.session_state.intent_stats = {}
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []

def update_stats(intent: str, response_time: float):
    """Update statistik chatbot"""
    st.session_state.total_interactions += 1
    
    if intent in st.session_state.intent_stats:
        st.session_state.intent_stats[intent] += 1
    else:
        st.session_state.intent_stats[intent] = 1
    
    st.session_state.response_times.append(response_time)

def display_analytics():
    """Menampilkan analytics dashboard"""
    st.subheader("📊 Analytics Dashboard")
    
    if st.session_state.total_interactions == 0:
        st.info("Belum ada interaksi untuk ditampilkan.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Interaksi", st.session_state.total_interactions)
    
    with col2:
        avg_response_time = np.mean(st.session_state.response_times) if st.session_state.response_times else 0
        st.metric("Avg Response Time", f"{avg_response_time:.3f}s")
    
    with col3:
        most_common_intent = max(st.session_state.intent_stats, key=st.session_state.intent_stats.get) if st.session_state.intent_stats else "N/A"
        st.metric("Top Intent", most_common_intent)
    
    # Intent Distribution Chart
    if st.session_state.intent_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Intent")
            intent_df = pd.DataFrame(list(st.session_state.intent_stats.items()), 
                                   columns=['Intent', 'Count'])
            fig_pie = px.pie(intent_df, values='Count', names='Intent', 
                           title="Distribusi Pertanyaan berdasarkan Intent")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Trend Response Time")
            if len(st.session_state.response_times) > 1:
                time_df = pd.DataFrame({
                    'Interaction': range(1, len(st.session_state.response_times) + 1),
                    'Response_Time': st.session_state.response_times
                })
                fig_line = px.line(time_df, x='Interaction', y='Response_Time',
                                 title="Response Time Trend")
                st.plotly_chart(fig_line, use_container_width=True)

def display_chat_interface(chatbot_type: str):
    """Menampilkan interface chat"""
    st.subheader(f"💬 Chat dengan {chatbot_type}")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user", avatar="👤"):
                st.write(chat["user_input"])
            
            with st.chat_message("assistant", avatar="🤖"):
                st.write(chat["bot_response"])
                if "intent" in chat:
                    st.caption(f"Intent: {chat['intent']}")
                if "confidence" in chat:
                    st.caption(f"Confidence: {chat['confidence']:.2f}")
    
    # Input area
    user_input = st.chat_input("Ketik pertanyaan Anda di sini...")
    
    if user_input:
        start_time = time.time()
        
        # Get response based on chatbot type
        if chatbot_type == "Rule-Based Chatbot":
            response, intent = st.session_state.chatbot_rule.get_response(user_input)
            confidence = None
        else:  # ML-Based Chatbot
            response, intent, confidence = st.session_state.chatbot_ml.get_response(user_input)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Add to chat history
        chat_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": response,
            "intent": intent,
            "response_time": response_time
        }
        
        if confidence is not None:
            chat_entry["confidence"] = confidence
        
        st.session_state.chat_history.append(chat_entry)
        
        # Update statistics
        update_stats(intent, response_time)
        
        # Rerun to show new message
        st.rerun()

def export_chat_history():
    """Export chat history sebagai JSON"""
    if st.session_state.chat_history:
        chat_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_interactions': st.session_state.total_interactions,
            'intent_statistics': st.session_state.intent_stats,
            'chat_history': st.session_state.chat_history
        }
        
        json_string = json.dumps(chat_data, indent=2, ensure_ascii=False)
        return json_string
    return None

def main():
    """Aplikasi utama Streamlit"""
    
    # Konfigurasi halaman
    st.set_page_config(
        page_title="E-commerce Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inisialisasi session state
    init_session_state()
    
    # Header
    st.title("🛒 E-commerce Chatbot Assistant")
    st.markdown("*Asisten virtual untuk membantu kebutuhan e-commerce Anda*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Pengaturan Chatbot")
        
        # Pilihan jenis chatbot
        chatbot_options = ["Rule-Based Chatbot"]
        if ML_AVAILABLE:
            chatbot_options.append("ML-Based Chatbot")
        
        chatbot_type = st.selectbox(
            "Pilih Jenis Chatbot:",
            chatbot_options,
            index=0
        )
        
        st.markdown("---")
        
        # Statistik real-time
        st.markdown("### 📊 Statistik Real-time")
        st.metric("Total Interaksi", st.session_state.total_interactions)
        
        if st.session_state.response_times:
            avg_time = np.mean(st.session_state.response_times)
            st.metric("Avg Response Time", f"{avg_time:.3f}s")
        
        if st.session_state.intent_stats:
            top_intent = max(st.session_state.intent_stats, key=st.session_state.intent_stats.get)
            st.metric("Top Intent", top_intent)
        
        st.markdown("---")
        
        # Informasi chatbot
        st.markdown("### ℹ️ Topik yang Didukung")
        topics = [
            "✅ Status Pesanan",
            "✅ Retur Barang",
            "✅ Promo & Diskon", 
            "✅ Cara Pembayaran",
            "✅ Pengiriman",
            "✅ Informasi Umum"
        ]
        for topic in topics:
            st.markdown(topic)
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Hapus Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.total_interactions = 0
                st.session_state.intent_stats = {}
                st.session_state.response_times = []
                st.rerun()
        
        with col2:
            chat_export = export_chat_history()
            if chat_export:
                st.download_button(
                    label="💾 Export",
                    data=chat_export,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "ℹ️ Info"])
    
    with tab1:
        display_chat_interface(chatbot_type)
    
    with tab2:
        display_analytics()
    
    with tab3:
        st.markdown("""
        ## 🤖 Tentang E-commerce Chatbot
        
        Chatbot ini dirancang untuk membantu customer service e-commerce dengan berbagai pertanyaan umum seperti:
        
        ### 🎯 Fitur Utama:
        - **Status Pesanan**: Cek tracking dan status pengiriman
        - **Retur Barang**: Panduan proses pengembalian barang
        - **Promo & Diskon**: Informasi penawaran dan voucher terbaru
        - **Pembayaran**: Metode dan cara pembayaran yang tersedia
        - **Pengiriman**: Info ongkir dan ekspedisi
        - **Informasi Umum**: Kontak CS dan informasi toko
        
        ### 🔧 Jenis Chatbot:
        1. **Rule-Based**: Menggunakan pattern matching dengan regular expressions
        2. **ML-Based**: Menggunakan machine learning untuk intent classification (jika tersedia)
        
        ### 📊 Analytics:
        - Real-time statistics tentang interaksi pengguna
        - Distribusi intent yang paling sering ditanyakan  
        - Response time monitoring
        - Export chat history untuk analisis lebih lanjut
        
        ### 💡 Tips Penggunaan:
        - Gunakan bahasa natural dalam bertanya
        - Sertakan kata kunci yang relevan (contoh: "pesanan", "retur", "promo")
        - Chatbot akan memberikan respons terbaik berdasarkan pattern yang dikenali
        
        ---
        *Dikembangkan dengan Streamlit dan Python* 🐍
        """)

if __name__ == "__main__":
    main()