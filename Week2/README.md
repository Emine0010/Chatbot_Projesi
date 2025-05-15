# Türkçe Embedding Tabanlı Soru-Cevap Chatbot

Bu proje, Türkçe teknik sorulara yanıt vermek amacıyla geliştirilmiş bir **embedding tabanlı etkileşimli chatbot** sistemidir. **SentenceTransformer**, **ChromaDB** ve **OpenAI GPT** teknolojilerini birleştirerek, kullanıcıdan gelen sorulara akıllı ve verimli yanıtlar sunar.

---

## 🚀 Temel Özellikler

- 🧠 **Türkçe metinlerden vektör çıkarımı** (SentenceTransformer)
- 🗃️ **ChromaDB ile vektör tabanlı veri sorgulama ve saklama**
- ✂️ **Türkçe stopword temizleme ve ön işleme işlemleri (NLTK ile)**
- 🤖 **OpenAI GPT-3.5-Turbo ile kısa ve net yanıt üretimi**
- 🔁 **Tekrarlanan verilerin kontrolü ve önlenmesi**
- 🧼 **Embedding güncellemelerini destekleyen altyapı**
- 📊 **Veritabanı durumu görüntüleme ve yeniden başlatma seçenekleri**

---

## 🛠️ Kurulum Adımları

1. **Python 3.8+** kurulu olmalıdır.
2. Gerekli bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

3. OpenAI API anahtarınızı `.env` dosyasına aşağıdaki formatta ekleyin:

```
OPENAI_API_KEY=your_api_key_here
```

4. İlk çalıştırma öncesi NLTK veri setlerini indirin:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

5. ChromaDB sunucusunun `localhost:8000` adresinde çalıştığından emin olun.

---

## 🧪 Kullanım

Terminal üzerinden aşağıdaki komutu çalıştırın:

```bash
python main.py
```

### Ana Menü Seçenekleri

```
1: Yeni bir soru sor
2: Veritabanı durumunu görüntüle
3: Veritabanını temizle ve örnek verileri yeniden yükle
q: Çıkış yap
```

---

## 📁 Veri Yapısı

- Tüm sorular `soru_cevaplar` adında bir ChromaDB koleksiyonunda saklanır.
- Her kayıt aşağıdaki bileşenlerden oluşur:
  - Soru (metin)
  - Cevap (metin)
  - Embedding vektörü (SentenceTransformer)
  - Tarih/saat damgası

---

## 🔧 Geliştirilebilir Alanlar

- 🌐 **Web tabanlı kullanıcı arayüzü** (Streamlit, Flask entegrasyonu)
- ⚙️ **Esnek benzerlik eşiği ayarları**
- 🧬 **Sadece değişen embedding’lerin güncellenmesi**
- 🧠 **Yinelenen veriler için hash tabanlı kontrol mekanizması**
- 📚 **Kullanıcı sorularının kategori ve konuya göre sınıflandırılması**

