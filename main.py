# E-commerce Chatbot Implementation
# Proyek Pembangunan Asisten Virtual E-commerce

import re
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pickle

# For ML/DL models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

import nltk

# Tambah path manual
nltk.data.path.append(r'C:\Users\ACER\AppData\Roaming\nltk_data')


# For text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    """Kelas untuk preprocessing teks bahasa Indonesia"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
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
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def preprocess(self, text: str) -> str:
        """Pipeline preprocessing lengkap"""
        text = self.clean_text(text)
        tokens = self.tokenize_and_filter(text)
        # Stemming (optional, karena Indonesian stemming kompleks)
        # tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(tokens)

class RuleBasedChatbot:
    """Tahap 1: Chatbot berbasis aturan dengan regex pattern matching"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.rules = self._load_rules()
        self.faq_responses = self._load_faq_responses()
    
    def _load_rules(self) -> Dict[str, List[str]]:
        """Definisi aturan pattern matching"""
        return {
            'status_pesanan': [
                r'.*status.*pesanan.*',
                r'.*cek.*order.*',
                r'.*tracking.*pesanan.*',
                r'.*dimana.*pesanan.*',
                r'.*lacak.*pesanan.*'
            ],
            'retur_barang': [
                r'.*retur.*barang.*',
                r'.*kembalikan.*barang.*',
                r'.*refund.*',
                r'.*tukar.*barang.*',
                r'.*komplain.*barang.*'
            ],
            'promo_diskon': [
                r'.*promo.*',
                r'.*diskon.*',
                r'.*voucher.*',
                r'.*kupon.*',
                r'.*sale.*'
            ],
            'pembayaran': [
                r'.*cara.*bayar.*',
                r'.*metode.*pembayaran.*',
                r'.*payment.*',
                r'.*transfer.*',
                r'.*cod.*'
            ],
            'pengiriman': [
                r'.*ongkir.*',
                r'.*pengiriman.*',
                r'.*kirim.*',
                r'.*ekspedisi.*',
                r'.*delivery.*'
            ]
        }
    
    def _load_faq_responses(self) -> Dict[str, List[str]]:
        """Template respons untuk setiap kategori"""
        return {
            'status_pesanan': [
                "Untuk mengecek status pesanan Anda, silakan masuk ke akun Anda dan pilih menu 'Pesanan Saya'. Atau berikan nomor pesanan Anda untuk pengecekan lebih lanjut.",
                "Anda bisa melacak pesanan melalui halaman 'Riwayat Pesanan' di aplikasi atau website kami. Pastikan Anda sudah login ke akun Anda."
            ],
            'retur_barang': [
                "Untuk retur barang, Anda dapat mengajukan pengembalian melalui menu 'Retur' di halaman pesanan. Pastikan barang masih dalam kondisi baik dan dalam periode retur (7-14 hari).",
                "Proses retur dapat dilakukan dengan cara: 1) Masuk ke akun Anda, 2) Pilih pesanan yang ingin diretur, 3) Klik 'Ajukan Retur', 4) Isi alasan retur."
            ],
            'promo_diskon': [
                "Promo terbaru dapat Anda lihat di halaman utama website atau aplikasi kami. Kami juga mengirimkan info promo melalui email dan notifikasi aplikasi.",
                "Untuk mendapatkan voucher diskon, Anda bisa mengikuti flash sale, member exclusive, atau program loyalitas kami."
            ],
            'pembayaran': [
                "Kami menerima berbagai metode pembayaran: Transfer Bank (BCA, Mandiri, BNI, BRI), E-wallet (OVO, GoPay, DANA), Kartu Kredit/Debit, dan COD untuk area tertentu.",
                "Pembayaran dapat dilakukan melalui virtual account, QRIS, atau COD. Pilih metode yang paling nyaman untuk Anda saat checkout."
            ],
            'pengiriman': [
                "Kami bekerja sama dengan berbagai ekspedisi: JNE, J&T, SiCepat, Pos Indonesia, dan GoSend untuk same-day delivery di area tertentu.",
                "Ongkos kirim dihitung berdasarkan berat barang dan jarak pengiriman. Anda bisa cek estimasi ongkir di halaman checkout."
            ]
        }
    
    def match_intent(self, user_input: str) -> Optional[str]:
        """Mencocokkan input dengan pattern rules"""
        cleaned_input = self.preprocessor.preprocess(user_input)
        
        for intent, patterns in self.rules.items():
            for pattern in patterns:
                if re.match(pattern, cleaned_input):
                    return intent
        return None
    
    def get_response(self, user_input: str) -> str:
        """Menghasilkan respons berdasarkan rule matching"""
        intent = self.match_intent(user_input)
        
        if intent and intent in self.faq_responses:
            responses = self.faq_responses[intent]
            return random.choice(responses)
        else:
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Respons fallback ketika tidak ada pattern yang cocok"""
        fallback_options = [
            "Maaf, saya belum mengerti pertanyaan Anda. Apakah Anda menanyakan tentang status pesanan, retur barang, atau promo?",
            "Bisa tolong diperjelas pertanyaannya? Saya bisa membantu dengan: cek status pesanan, prosedur retur, info promo, cara pembayaran, atau pengiriman.",
            "Mohon maaf, saya tidak memahami maksud Anda. Silakan pilih topik: Status Pesanan | Retur Barang | Promo | Pembayaran | Pengiriman"
        ]
        return random.choice(fallback_options)

class IntentClassifier:
    """Tahap 2A: Intent Detection menggunakan Machine Learning"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.model = None
        self.vectorizer = None
        self.intent_labels = [
            'status_pesanan', 'retur_barang', 'promo_diskon', 
            'pembayaran', 'pengiriman', 'informasi_umum'
        ]
    
    def generate_sample_data(self) -> Tuple[List[str], List[str]]:
        """Generate sample training data untuk demo"""
        sample_data = {
            'status_pesanan': [
                "dimana pesanan saya", "cek status order", "kapan barang sampai",
                "tracking pesanan 12345", "pesanan belum sampai", "order sudah berapa lama",
                "status pengiriman", "update pesanan", "konfirmasi pesanan"
            ],
            'retur_barang': [
                "mau retur barang", "gimana cara return", "barang rusak mau tukar",
                "refund pesanan", "komplain produk", "barang tidak sesuai",
                "mau kembalikan barang", "garansi produk", "tukar barang baru"
            ],
            'promo_diskon': [
                "ada promo apa", "voucher diskon", "sale kapan", "kupon gratis ongkir",
                "flash sale hari ini", "cashback berapa", "member discount",
                "promo weekend", "diskon spesial"
            ],
            'pembayaran': [
                "cara bayar gimana", "payment method", "bisa cod tidak", "transfer kemana",
                "bayar pakai ovo", "cicilan kartu kredit", "virtual account",
                "qris payment", "gopay bisa tidak"
            ],
            'pengiriman': [
                "ongkir berapa", "ekspedisi apa saja", "same day delivery", "gratis ongkir",
                "lama pengiriman", "jne atau jnt", "kirim ke luar kota",
                "area pengiriman", "estimasi sampai"
            ],
            'informasi_umum': [
                "jam operasional", "customer service", "alamat toko", "kontak cs",
                "cara daftar member", "syarat dan ketentuan", "kebijakan privasi",
                "tentang toko", "cabang dimana saja"
            ]
        }
        
        texts = []
        labels = []
        for intent, samples in sample_data.items():
            texts.extend(samples)
            labels.extend([intent] * len(samples))
        
        return texts, labels
    
    def train(self):
        """Melatih model intent classifier"""
        texts, labels = self.generate_sample_data()
        
        # Preprocessing
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Intent Classifier Accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Prediksi intent dari input teks"""
        if self.model is None:
            raise ValueError("Model belum dilatih. Jalankan train() terlebih dahulu.")
        
        processed_text = self.preprocessor.preprocess(text)
        predicted_intent = self.model.predict([processed_text])[0]
        confidence = max(self.model.predict_proba([processed_text])[0])
        
        return predicted_intent, confidence

class MLChatbot:
    """Chatbot berbasis Machine Learning dengan Intent Classification"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.rule_chatbot = RuleBasedChatbot()  # Fallback ke rule-based
        self.confidence_threshold = 0.6
    
    def train(self):
        """Melatih model chatbot"""
        print("Melatih Intent Classifier...")
        accuracy = self.intent_classifier.train()
        print(f"Model berhasil dilatih dengan akurasi: {accuracy:.3f}")
    
    def get_response(self, user_input: str) -> str:
        """Menghasilkan respons berdasarkan ML prediction"""
        try:
            intent, confidence = self.intent_classifier.predict_intent(user_input)
            
            if confidence >= self.confidence_threshold:
                # Gunakan respons berdasarkan predicted intent
                if intent in self.rule_chatbot.faq_responses:
                    responses = self.rule_chatbot.faq_responses[intent]
                    response = random.choice(responses)
                    return f"{response}\n\n(Confidence: {confidence:.2f})"
                else:
                    return self.rule_chatbot._fallback_response()
            else:
                # Fallback ke rule-based jika confidence rendah
                return self.rule_chatbot.get_response(user_input)
                
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self.rule_chatbot.get_response(user_input)

class ChatbotEvaluator:
    """Kelas untuk evaluasi performa chatbot"""
    
    def __init__(self):
        self.test_cases = self._generate_test_cases()
    
    def _generate_test_cases(self) -> List[Tuple[str, str]]:
        """Generate test cases untuk evaluasi"""
        return [
            ("dimana pesanan saya nomor 12345", "status_pesanan"),
            ("mau retur barang yang rusak", "retur_barang"),
            ("ada promo diskon gak", "promo_diskon"),
            ("gimana cara bayarnya", "pembayaran"),
            ("ongkir ke jakarta berapa", "pengiriman"),
            ("jam buka toko berapa", "informasi_umum"),
            ("barang belum sampai sudah seminggu", "status_pesanan"),
            ("voucher cashback kapan", "promo_diskon"),
            ("bisa cod tidak", "pembayaran"),
            ("ekspedisi jne tersedia", "pengiriman")
        ]
    
    def evaluate_intent_accuracy(self, chatbot) -> float:
        """Evaluasi akurasi intent detection"""
        if not hasattr(chatbot, 'intent_classifier'):
            print("Chatbot tidak memiliki intent classifier")
            return 0.0
        
        correct_predictions = 0
        total_predictions = len(self.test_cases)
        
        for text, expected_intent in self.test_cases:
            try:
                predicted_intent, confidence = chatbot.intent_classifier.predict_intent(text)
                if predicted_intent == expected_intent:
                    correct_predictions += 1
            except:
                pass
        
        accuracy = correct_predictions / total_predictions
        print(f"Intent Detection Accuracy: {accuracy:.3f}")
        return accuracy
    
    def evaluate_response_quality(self, chatbot, num_samples: int = 10) -> Dict:
        """Evaluasi kualitas respons (simulasi human evaluation)"""
        sample_questions = [
            "pesanan saya belum sampai, gimana ceknya?",
            "barang yang saya terima rusak, bisa ditukar?",
            "kapan ada flash sale lagi?",
            "bisa bayar pakai dana?",
            "ongkir ke bandung berapa ya?"
        ]
        
        results = {
            'fluency_scores': [],
            'relevance_scores': [],
            'coherence_scores': []
        }
        
        print("\n=== Response Quality Evaluation ===")
        for i, question in enumerate(sample_questions[:num_samples]):
            response = chatbot.get_response(question)
            print(f"\nQ{i+1}: {question}")
            print(f"A{i+1}: {response}")
            
            # Simulasi scoring (dalam implementasi nyata, ini akan dinilai oleh manusia)
            fluency = random.uniform(3.5, 5.0)  # Simulated fluency score
            relevance = random.uniform(3.0, 5.0)  # Simulated relevance score
            coherence = random.uniform(3.5, 5.0)  # Simulated coherence score
            
            results['fluency_scores'].append(fluency)
            results['relevance_scores'].append(relevance)
            results['coherence_scores'].append(coherence)
        
        # Calculate averages
        avg_fluency = np.mean(results['fluency_scores'])
        avg_relevance = np.mean(results['relevance_scores'])
        avg_coherence = np.mean(results['coherence_scores'])
        
        print(f"\n=== Evaluation Results ===")
        print(f"Average Fluency: {avg_fluency:.2f}/5.0")
        print(f"Average Relevance: {avg_relevance:.2f}/5.0")
        print(f"Average Coherence: {avg_coherence:.2f}/5.0")
        print(f"Overall Score: {(avg_fluency + avg_relevance + avg_coherence)/3:.2f}/5.0")
        
        return results

class ChatbotInterface:
    """Interface untuk interaksi dengan chatbot"""
    
    def __init__(self, chatbot_type: str = "ml"):
        if chatbot_type == "rule":
            self.chatbot = RuleBasedChatbot()
            self.bot_name = "Rule-Based Chatbot"
        elif chatbot_type == "ml":
            self.chatbot = MLChatbot()
            self.chatbot.train()
            self.bot_name = "ML-Based Chatbot"
        else:
            raise ValueError("chatbot_type harus 'rule' atau 'ml'")
        
        self.conversation_history = []
    
    def start_conversation(self):
        """Memulai percakapan dengan chatbot"""
        print(f"\n🤖 Selamat datang di {self.bot_name} E-commerce!")
        print("Saya siap membantu Anda dengan pertanyaan seputar:")
        print("✓ Status Pesanan  ✓ Retur Barang  ✓ Promo & Diskon")
        print("✓ Cara Pembayaran  ✓ Pengiriman  ✓ Informasi Umum")
        print("\nKetik 'quit' atau 'exit' untuk mengakhiri percakapan.\n")
        
        while True:
            try:
                user_input = input("Anda: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye', 'selesai']:
                    print("🤖 Terima kasih telah menggunakan layanan kami! Sampai jumpa! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Get response from chatbot
                response = self.chatbot.get_response(user_input)
                print(f"🤖 {self.bot_name}: {response}\n")
                
                # Save to conversation history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user_input': user_input,
                    'bot_response': response
                })
                
            except KeyboardInterrupt:
                print("\n\n🤖 Percakapan dihentikan. Sampai jumpa!")
                break
            except Exception as e:
                print(f"🤖 Maaf, terjadi kesalahan: {e}")
                print("Silakan coba lagi atau ketik 'quit' untuk keluar.")
    
    def save_conversation(self, filename: str = None):
        """Menyimpan riwayat percakapan"""
        if not filename:
            filename = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"Riwayat percakapan disimpan ke: {filename}")

def main():
    """Fungsi utama untuk menjalankan demo chatbot"""
    print("🚀 E-commerce Chatbot Demo")
    print("=" * 50)
    
    # Pilihan jenis chatbot
    print("\nPilih jenis chatbot yang ingin dicoba:")
    print("1. Rule-Based Chatbot (Tahap 1)")
    print("2. ML-Based Chatbot (Tahap 2)")
    print("3. Evaluasi Perbandingan")
    
    try:
        choice = input("\nMasukkan pilihan (1/2/3): ").strip()
        
        if choice == "1":
            print("\n🔧 Memuat Rule-Based Chatbot...")
            interface = ChatbotInterface("rule")
            interface.start_conversation()
            
        elif choice == "2":
            print("\n🧠 Memuat ML-Based Chatbot...")
            print("Sedang melatih model...")
            interface = ChatbotInterface("ml")
            interface.start_conversation()
            
        elif choice == "3":
            print("\n📊 Menjalankan Evaluasi Perbandingan...")
            
            # Inisialisasi chatbots
            print("Menyiapkan Rule-Based Chatbot...")
            rule_bot = RuleBasedChatbot()
            
            print("Menyiapkan ML-Based Chatbot...")
            ml_bot = MLChatbot()
            ml_bot.train()
            
            # Evaluasi
            evaluator = ChatbotEvaluator()
            
            print("\n" + "="*50)
            print("EVALUASI RULE-BASED CHATBOT")
            print("="*50)
            evaluator.evaluate_response_quality(rule_bot)
            
            print("\n" + "="*50)
            print("EVALUASI ML-BASED CHATBOT")
            print("="*50)
            evaluator.evaluate_intent_accuracy(ml_bot)
            evaluator.evaluate_response_quality(ml_bot)
            
        else:
            print("Pilihan tidak valid!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()