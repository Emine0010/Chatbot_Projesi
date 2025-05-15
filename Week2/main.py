from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import chromadb 
from chromadb.config import Settings 
import string
import re 
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv(dotenv_path="openai.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# İlk kullanımda bu iki satırı çalıştır (bir kereye mahsus)
# nltk.download("punkt")
# nltk.download("stopwords")

# Türkçe stopwords listesi
stop_words = set(stopwords.words("turkish"))

# Cümledeki stopwords'leri temizleyen fonksiyon
def temizle_stopwords(cumle):
    cumle = cumle.lower()  #Hepsini kçük harfe çeviriyor
    cumle = re.sub(r"[^\w\s]", "", cumle)  # Noktalama işaretlerini ve özel karakterleri kaldır
    kelimeler = word_tokenize(cumle, language='turkish')
    temiz = [kelime for kelime in kelimeler if kelime not in stop_words and kelime.isalpha()]
    return " ".join(temiz)

def API_ILE_CEVAP_AL(soru):
    """OpenAI API kullanarak teknik sorulara kısa ve net cevaplar döner"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen bir teknik yapay zekasın. "
                        "Sadece yazılım, programlama, yapay zeka, siber güvenlik gibi teknik konularda gelen sorulara cevap ver. "
                        "Cevapların kısa, net, doğrudan ve teknik dille yazılmalı. Gereksiz açıklama, öneri veya selamlaşma ekleme. "
                        "Eğer gelen soru teknik değilse, 'Bu sistem yalnızca teknik sorulara cevap verir.' yanıtını ver."
                    )
                },
                {"role": "user", "content": soru}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def veritabani_kontrol(collection):
    # Veritabanındaki kayıtları kontrol eder ve istatistikleri gösterir.
    try:
        count = collection.count()
        print(f"\n🔍 Veritabanı Durumu: {count} kayıt bulunuyor")
        
        if count > 0:
            results = collection.peek(limit=count)
            print("\nÖrnek Kayıtlar:")
            for doc, meta in zip(results['documents'], results['metadatas']):
                print(f"- Soru: {doc} | Cevap: {meta['cevap']}")
        return True
    except Exception as e:
        print(f"Veritabanı kontrol hatası: {e}")
        return False

def veritabani_temizle(client, collection_name="soru_cevaplar"):
    """Belirtilen koleksiyonu tamamen siler"""
    try:
        client.delete_collection(name=collection_name)
        print(f"\n🗑 '{collection_name}' koleksiyonu tamamen silindi")
        return True
    except Exception as e:
        print(f"Silme hatası: {e}")
        return False

# Model yükleme
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Chroma istemcisi
try:
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    # Koleksiyon oluştur (eğer yoksa)
    collection = chroma_client.get_or_create_collection(
        name="soru_cevaplar",
        metadata={"hnsw:space": "cosine"}  # Cosine similarity kullan
    )
    
    # Veritabanı durumunu kontrol et
    # veritabani_kontrol(collection)
    
except Exception as e:
    print(f"ChromaDB bağlantı hatası: {e}")
    exit()

# Soru-cevap veri seti (ilk defa açılış için)
data = {
    "soru": [
        "Makine öğrenmesi nedir?",
        "Denetimli öğrenme ne demektir?",
        "Denetimsiz öğrenme nedir?",
        "Veri kümesi (dataset) nedir?",
        "Makine öğrenmesinde model ne demektir?",
        "Overfitting ne anlama gelir?",
        "Doğruluk (accuracy) nasıl hesaplanır?",
        "Makine öğrenmesinde eğitim ve test verisi neden ayrılır?",
        "Lineer regresyon ne işe yarar?",
        "Scikit-learn hangi amaçla kullanılır?",
        "Karmaşık bir modelin öğrenmesi neden daha uzun sürer?",
        "Hiperparametre nedir?",
        "Modelin doğruluğu nasıl iyileştirilir?",
        "Veri temizleme neden önemlidir?",
        "Özellik mühendisliği (feature engineering) nedir?",
        "Veri madenciliği (data mining) nedir?",
        "Yapay sinir ağları nasıl çalışır?",
        "Derin öğrenme nedir?",
        "Aktivasyon fonksiyonları nelerdir?",
        "Sınıflandırma ve regresyon arasındaki fark nedir?",
    ],
    "cevap": [
        "Makine öğrenmesi, veriden öğrenen algoritmalardır.",
        "Denetimli öğrenme, etiketli veriyle modeli eğitmektir.",
        "Denetimsiz öğrenme, verideki yapıları etiket olmadan keşfetmektir.",
        "Veri kümesi, modelin öğrenmesi için kullanılan örnekler topluluğudur.",
        "Model, öğrenme süreçlerini kontrol eden yapıdır.",
        "Overfitting, modelin veriye aşırı uyum sağlamasıdır.",
        "Doğruluk, doğru tahminlerin toplam tahminlere oranıdır.",
        "Eğitim verisi, modelin öğrenmesini sağlarken test verisi modelin doğruluğunu test eder.",
        "Lineer regresyon, sürekli veriyle ilişkiyi modellemeyi amaçlar.",
        "Scikit-learn, Python'da makine öğrenmesi için kullanılan bir kütüphanedir.",
        "Karmaşık modeller daha fazla parametreye sahip olduğu için daha fazla hesaplama gerektirir.",
        "Hiperparametre, modelin dışsal parametreleridir ve doğru ayarlanması gereklidir.",
        "Model doğruluğunu artırmak için parametrelerin optimizasyonu ve veri iyileştirmesi yapılabilir.",
        "Veri temizleme, hatalı ve eksik verilerin düzeltilmesidir ve modelin başarısı için önemlidir.",
        "Özellik mühendisliği, ham verilerden anlamlı özellikler çıkarmaktır.",
        "Veri madenciliği, büyük veri kümelerinden desenlerin ve ilişkilerin çıkarılması sürecidir.",
        "Yapay sinir ağları, insan beynine benzer şekilde çalışan öğrenme modelleridir.",
        "Derin öğrenme, çok katmanlı yapılarla veri analizini içeren bir makine öğrenmesi türüdür.",
        "Aktivasyon fonksiyonları, modelin öğrenme sürecinde doğrusal olmayan özellikler ekler.",
        "Sınıflandırma, veriyi kategorilere ayırırken regresyon sürekli bir çıktıyı tahmin eder.",
    ]
}

# İlk kullanım için verileri ekleme (bir kere çalıştırılmalı)
'''
if collection.count() == 0:
    print("Veritabanı boş, örnek veriler ekleniyor...")
    embeddings = model.encode(data["soru"]).tolist()
    ids = [f"id_{i}" for i in range(len(data["soru"]))]
    
    collection.add(
        documents=data["soru"],
        metadatas=[{"cevap": cevap} for cevap in data["cevap"]],
        embeddings=embeddings,
        ids=ids
    )
    print(f"{len(data['soru'])} adet soru-cevap çifti veritabanına eklendi.")
'''

# Yeni soru alma
while True:
    print("\n" + "="*50)
    print("1: Yeni soru sor")
    print("2: Veritabanı durumunu göster")
    print("3: Veritabanını temizle")
    print("q: Çıkış")
    secim = input("🔹 Seçiminiz: ").strip().lower()
    
    if secim == 'q':
        break
    elif secim == '1':
        yeni_soru = input("\n🔹 Yeni soruyu girin: ").strip()
        if not yeni_soru:
            continue
            
        yeni_soru_temiz = temizle_stopwords(yeni_soru)
        yeni_embedding = model.encode(yeni_soru_temiz).tolist()

        try:
            # Benzer sorgu - Daha fazla sonuç alıp en iyisini seçelim
            sonuc = collection.query(
                query_embeddings=[yeni_embedding],
                n_results=3  # İlk 3 sonucu al
            )

            if sonuc["documents"] and len(sonuc["documents"][0]) > 0:
                # En iyi eşleşmeyi bul
                best_match_idx = 0
                best_similarity = 1 - sonuc["distances"][0][0]
                
                # Diğer sonuçları da kontrol et
                for i in range(1, len(sonuc["distances"][0])):
                    current_similarity = 1 - sonuc["distances"][0][i]
                    if current_similarity > best_similarity:
                        best_similarity = current_similarity
                        best_match_idx = i
                
                en_benzer_soru = sonuc["documents"][0][best_match_idx]
                en_benzer_cevap = sonuc["metadatas"][0][best_match_idx]["cevap"]
                
                
                if best_similarity > 0.7:  # Eşik değerini yükselttim                   
                    print(f"\nEn benzer soru: {en_benzer_soru}")
                    print(f"\nCevap: {en_benzer_cevap}")
                    print(f"Benzerlik skoru: {best_similarity:.2f}")
                
                    # Eğer soru tam olarak aynı değilse veritabanına ekle
                    if yeni_soru.lower() != en_benzer_soru.lower():
                        print("Soru benzer ama tam aynı değil. Veri setine yeni varyasyon olarak eklendi.")
                        new_id = f"id_{collection.count()}"
                        collection.add(
                            documents=[yeni_soru],
                            metadatas=[{"cevap": en_benzer_cevap}],
                            embeddings=[yeni_embedding],
                            ids=[new_id]
                        )
                else:
                    print("\n🧠 Bu konuda yeterli bilgim yok. API'den yanıt alınıyor...")
                    yeni_cevap = API_ILE_CEVAP_AL(yeni_soru)
                    print(f"\n✨ Yeni cevap: {yeni_cevap}")
                    
                    # Yeni soru-cevap çiftini veritabanına ekle
                    new_id = f"id_{collection.count()}"
                    collection.add(
                        documents=[yeni_soru],
                        metadatas=[{"cevap": yeni_cevap}],
                        embeddings=[yeni_embedding],
                        ids=[new_id]
                    )
                    
        except Exception as e:
            print(f"\nHata oluştu: {e}")
            
    elif secim == '2':
        veritabani_kontrol(collection)
        
    elif secim == '3':
        onay = input("\n⚠ Tüm veritabanı silinecek! Emin misiniz? (e/h): ").lower()
        if onay == 'e':
            if veritabani_temizle(chroma_client):
                # Koleksiyonu yeniden oluştur
                collection = chroma_client.get_or_create_collection(
                    name="soru_cevaplar",
                    metadata={"hnsw:space": "cosine"}
                )
                # Örnek verileri yeniden yükle
                if collection.count() == 0:
                    embeddings = model.encode(data["soru"]).tolist()
                    ids = [f"id_{i}" for i in range(len(data["soru"]))]
                    collection.add(
                        documents=data["soru"],
                        metadatas=[{"cevap": cevap} for cevap in data["cevap"]],
                        embeddings=embeddings,
                        ids=ids
                    )
                    print("\n✅ Örnek veriler yeniden yüklendi")
    else:
        print("Geçersiz seçim, tekrar deneyin")

# İstemciyi kapat (opsiyonel)
chroma_client.heartbeat()  # Son bir bağlantı testi
print("\nUygulama sonlandırıldı")