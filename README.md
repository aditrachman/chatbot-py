# Chatbot E-Commerce Berbahasa Indonesia

Chatbot ini adalah asisten virtual untuk platform e-commerce, dirancang untuk menangani pertanyaan umum pelanggan dalam bahasa Indonesia secara otomatis, seperti cek status pesanan, pengembalian barang, promo, dan info pembayaran. Chatbot mampu melakukan fallback jika tidak mengerti, serta belajar dari log percakapan.

## Latar Belakang

Banyak pelanggan e-commerce membutuhkan respons cepat untuk pertanyaan berulang. Chatbot ini membantu:
- Menjawab pertanyaan FAQ secara otomatis.
- Memberikan pengalaman pelanggan yang konsisten.
- Mengurangi beban customer support.

## Dataset

- **Chat Historis Customer Support:**  
  > 10.000–15.000 pasang dialog customer-agent (Bahasa Indonesia).
- **Dokumen FAQ & Knowledge Base:**  
  > List pertanyaan umum, jawaban, dan deskripsi singkat produk.

## Fitur Utama

1. **Chatbot Rule-Based (Baseline)**
   - Pattern matching dengan regex (NLTK/spaCy).
   - Contoh rules:  
     - “status” atau “tracking” + “pesanan” → otomatis jawab tentang cek status pesanan.
     - “retur” atau “kembali” + “barang” → otomatis jawab tentang retur.
   - Fallback jika tidak ada pola yang cocok.

2. **Chatbot Machine Learning / Deep Learning**
   - **Intent Detection & Response Retrieval:**
     - Intent classifier untuk deteksi maksud.
     - Response retrieval dari FAQ.
   - **Neural Seq2Seq / Transformer-based Model:**
     - Encoder-Decoder LSTM dengan attention.
     - Atau fine-tuning model generatif (DialoGPT multilingual, dst.).
     - Training dengan chat historis.

3. **Evaluasi & Analisis**
   - **Evaluasi Otomatis:**  
     - Akurasi intent classifier, metrik BLEU/ROUGE untuk model generatif.
   - **Evaluasi Manual:**  
     - Skala 1–5 untuk fluency, relevance, coherence.

4. **Deployment Minimal (Opsional)**
   - Prototipe antarmuka chat (CLI/web sederhana).
   - Mempertahankan konteks percakapan sederhana.

## Cara Instalasi

```bash
# 1. Clone repository
git clone https://github.com/aditrachman/chatbot-py.git
cd chatbot-py

# 2. Install dependencies (misal: pip)
pip install -r requirements.txt

# 3. (Opsional) Download/preprocess dataset
# 4. Jalankan prototipe chatbot
python main.py
```

> **Catatan:**  
> Update langkah instalasi sesuai dengan struktur dan dependensi aktual di repo.

## Struktur Proyek (Contoh)

```
chatbot-py/
├── data/                  # Dataset chat & FAQ
├── rule_based/            # Modul chatbot rule-based
├── ml/                    # Intent detection & response retrieval
├── seq2seq/               # Model LSTM/Transformer
├── evaluation/            # Evaluasi otomatis & manual
├── prototype/             # CLI/web chat demo
├── requirements.txt
└── main.py
```

## Dokumentasi & Laporan

- **Deskripsi Dataset**
- **Metode Preprocessing**
- **Evaluasi (Automatic & Manual)**
- **Analisis Perbandingan**
- **Kesimpulan**

## Kontribusi

Kontributor dipersilakan untuk mengembangkan, menambah fitur, atau memperbaiki dokumentasi.

## Lisensi

Project ini dilisensikan di bawah [MIT License](LICENSE).

---

> Untuk informasi lebih lanjut, silakan baca dokumentasi pada folder terkait atau hubungi maintainer repo.
