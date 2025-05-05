# Soru-Cevap Asistanı (Makine Öğrenmesi Konuları)

Bu proje, makine öğrenmesi alanındaki teknik sorulara cevap veren bir Türkçe Soru-Cevap sistemidir. 
Gelen soruyu anlamak için `SentenceTransformer` modeli kullanılır ve daha önce tanımlı sorularla karşılaştırılarak en benzer cevabı verir.

## Özellikler

- Türkçe makine öğrenmesi soru-cevap sistemi
- Benzerlik tespiti için cümle gömme (sentence embedding) ve cosine similarity kullanımı
- Eşik değerine göre benzer sorulara aynı cevabı verme veya bilgi eksikliğini bildirme
- Yeni benzer soruların veri setine otomatik olarak eklenmesi
- Kalıcı veri kaydı (`pkl` dosyası) sayesinde öğrenme devamlılığı
  
## Kullanım

-  main.py (veya dosyanın adı neyse) dosyasını çalıştırın:

 python main.py

-  Komut satırına bir soru yazın. Örneğin:

 Yeni soruyu girin: Denetimli öğrenme nasıl çalışır?

-  Sistem, en benzer soruyu bulup cevabını verir. Eğer yeterli benzerlik varsa bu yeni soruyu varyasyon olarak veri setine ekler.

## Notlar
- İlk çalıştırmada sistem örnek veri setini yükler. Daha sonra oluşturulan guncel_veri_seti.pkl dosyası sayesinde sistem öğrendiği soruları unutmaz.
- Benzerlik eşiği (threshold) 0.75 olarak ayarlanmıştır. Daha yüksek hassasiyet istenirse bu değer değiştirilebilir.

