from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Model yükleme
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

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

# Önceki veri setini yükleme
try:
    df = pd.read_pickle("guncel_veri_seti.pkl")
except FileNotFoundError:
    df = pd.DataFrame(data)

# Soru embedding'lerini oluşturma
df["embedding"] = df["soru"].apply(lambda x: model.encode(x))

# Yeni soru alma
yeni_soru = input("🔹 Yeni soruyu girin: ")
yeni_embedding = model.encode(yeni_soru)

# Benzerlikleri hesaplama
df["benzerlik"] = df["embedding"].apply(lambda x: cosine_similarity([x], [yeni_embedding])[0][0])

# En benzer soruyu bul
en_benzer_satir = df.sort_values("benzerlik", ascending=False).iloc[0]
en_benzerlik = en_benzer_satir["benzerlik"]
en_benzer_soru = en_benzer_satir["soru"]
en_benzer_cevap = en_benzer_satir["cevap"]

# Eşik
esik = 0.75

if en_benzerlik > esik:
    if yeni_soru.strip() != en_benzer_soru.strip():
        print(" Soru benzer ama tam aynı değil. Veri setine yeni varyasyon olarak eklendi.")
        # Yeni soruyu aynı cevapla ekliyoruz
        yeni_kayit = pd.DataFrame({
            "soru": [yeni_soru],
            "cevap": [en_benzer_cevap],
            "embedding": [yeni_embedding]
        })
        df = pd.concat([df, yeni_kayit], ignore_index=True)

    print(f" Cevap: {en_benzer_cevap}")
else:
    print(" Bu konuda elimde bilgi yok. Lütfen ilgili bir teknik soru girin.")

# Yeni veriyi kaydet
df.to_pickle("guncel_veri_seti.pkl")
