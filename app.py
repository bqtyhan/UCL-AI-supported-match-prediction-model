import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------------------------------------------------------
# 1. AYARLAR
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CL AI Tahmin", layout="wide")

# -----------------------------------------------------------------------------
# 2. VERİLERİ VE MODELİ YÜKLEME
# -----------------------------------------------------------------------------
@st.cache_resource
def verileri_yukle():
    try:
        model = joblib.load('sampiyonlar_ligi_modeli.pkl')
        le = joblib.load('takim_isimleri_encoder.pkl')
        home_stats = joblib.load('ev_sahibi_istatistikleri.pkl')
        away_stats = joblib.load('deplasman_istatistikleri.pkl')
    except:
        st.error("DOSYA HATASI: .pkl dosyaları bulunamadı.")
        st.stop()
        
    df = pd.read_csv("champions_league_matches.csv")
    df_gelecek = df[df['score'].isnull()].copy()
    df_gelecek = df_gelecek.dropna(subset=['home_team', 'away_team']).reset_index(drop=True)
    
    tum_takimlar = le.classes_
    sayisal_sutunlar = home_stats.columns.tolist()
    
    return model, le, home_stats, away_stats, df_gelecek, tum_takimlar, sayisal_sutunlar

model, le, home_stats, away_stats, df_gelecek, tum_takimlar, sayisal_sutunlar = verileri_yukle()

# -----------------------------------------------------------------------------
# 3. ARAYÜZ TASARIMI
# -----------------------------------------------------------------------------
st.title("🏆 Şampiyonlar Ligi: Tam Kontrol Modu (Güvenli V3)")
st.markdown("---")

tab1, tab2 = st.tabs(["🛠️ Detaylı Manuel Tahmin", "📅 Gelecek Fikstür"])

# --- TAB 1: HER ŞEYİ SEN BELİRLE ---
with tab1:
    st.write("Burada maçın kaderini sen yazıyorsun. (Hatalı veri girersen buton kaybolur!)")
    
    col_team1, col_team2 = st.columns(2)
    with col_team1:
        ev_sahibi = st.selectbox("🏠 Ev Sahibi", tum_takimlar, index=0)
    with col_team2:
        deplasman = st.selectbox("✈️ Deplasman", tum_takimlar, index=1)

    st.markdown("### 📊 Maç İstatistikleri")
    
    kullanici_verisi = {}
    col_sol, col_sag = st.columns(2)

    # Hata kontrol bayrağı (Başlangıçta hata yok varsayıyoruz)
    hata_var = False

    def guvenli_input(label, default_val, column_name):
        is_percentage = 'pct' in column_name or 'possession' in column_name
        
        # Eğer yüzde ise maks 100, değilse 1000
        max_limit = 100.0 if is_percentage else 1000.0
        
        val = st.number_input(
            label, 
            value=float(default_val), 
            min_value=0.0, 
            step=1.0,
            format="%.2f",
            key=column_name # Streamlit'in karışmaması için benzersiz anahtar
        )
        
        # ANLIK KONTROL: Eğer yüzde verisi 100'ü geçerse hata var demektir
        if is_percentage and val > 100.0:
            return val, True # Değeri döndür ama "Hata Var" de
        
        return val, False

    # --- SÜTUNLARI DOLDUR ---
    with col_sol:
        st.info(f"{ev_sahibi} İstatistikleri")
        for col in sayisal_sutunlar:
            if "home_" in col: 
                varsayilan = float(home_stats.loc[ev_sahibi][col])
                baslik = col.replace("home_", "").replace("_", " ").title()
                
                # Inputu al ve hata durumunu kontrol et
                deger, hatali_mi = guvenli_input(f"{baslik}", varsayilan, col)
                kullanici_verisi[col] = deger
                
                if hatali_mi:
                    st.error(f"⚠️ {baslik} değeri 100'den büyük olamaz!")
                    hata_var = True

    with col_sag:
        st.success(f"{deplasman} İstatistikleri")
        for col in sayisal_sutunlar:
            if "away_" in col:
                varsayilan = float(away_stats.loc[deplasman][col])
                baslik = col.replace("away_", "").replace("_", " ").title()
                
                deger, hatali_mi = guvenli_input(f"{baslik}", varsayilan, col)
                kullanici_verisi[col] = deger
                
                if hatali_mi:
                    st.error(f"⚠️ {baslik} değeri 100'den büyük olamaz!")
                    hata_var = True

    # --- TOPLA OYNAMA KONTROLÜ (EN KRİTİK NOKTA) ---
    st.markdown("---")
    
    # Possession verilerini bul
    home_pos = kullanici_verisi.get('home_possession', 0)
    away_pos = kullanici_verisi.get('away_possession', 0)
    toplam_pos = home_pos + away_pos

    # HATA KONTROL MERKEZİ
    if abs(toplam_pos - 100.0) > 0.1: # Ufak küsurat hatalarını görmezden gelmek için 0.1 tolerans
        st.error(f"⛔ DUR: Topla oynama oranları toplamı 100 olmalı! Şu an: {toplam_pos:.1f}")
        hata_var = True
    
    # EĞER HATA YOKSA BUTONU GÖSTER, VARSA GÖSTERME
    if not hata_var:
        if st.button("MAÇI OYNAT VE TAHMİN ET 🎲", type="primary", use_container_width=True):
            input_df = pd.DataFrame(columns=['home_team_code', 'away_team_code'] + sayisal_sutunlar)
            input_df.loc[0] = 0
            input_df['home_team_code'] = le.transform([ev_sahibi])[0]
            input_df['away_team_code'] = le.transform([deplasman])[0]
            
            for col, deger in kullanici_verisi.items():
                input_df[col] = deger
                
            sonuc = model.predict(input_df)[0]
            
            st.markdown("## 🏁 MAÇ SONUCU")
            if sonuc == 1:
                st.success(f"🏆 KAZANAN: {ev_sahibi} (EV SAHİBİ)")
            elif sonuc == -1:
                st.error(f"🏆 KAZANAN: {deplasman} (DEPLASMAN)")
            else:
                st.warning("⚖️ MAÇ SONUCU: BERABERE")
    else:
        # Hata varsa buton yerine uyarı kutusu koy
        st.warning("⚠️ Tahmin yapabilmek için yukarıdaki kırmızı hataları düzeltmelisin.")

# --- TAB 2: GERÇEK FİKSTÜR (AYNI) ---
with tab2:
    st.header("2026 Sezonu Gelecek Maçlar")
    if st.button("Geleceği Hesapla 🔮"):
        bar = st.progress(0)
        tahminler_listesi = []
        toplam_mac = len(df_gelecek)
        
        for i, row in df_gelecek.iterrows():
            ev, dep = row['home_team'], row['away_team']
            if ev not in tum_takimlar or dep not in tum_takimlar: continue

            tek_mac = pd.DataFrame(columns=['home_team_code', 'away_team_code'] + sayisal_sutunlar)
            tek_mac.loc[0] = 0
            tek_mac['home_team_code'] = le.transform([ev])[0]
            tek_mac['away_team_code'] = le.transform([dep])[0]
            
            for col in sayisal_sutunlar:
                if 'home_' in col: tek_mac[col] = home_stats.loc[ev][col]
                elif 'away_' in col: tek_mac[col] = away_stats.loc[dep][col]
            
            tek_mac = tek_mac.fillna(0)
            sonuc = model.predict(tek_mac)[0]
            sonuc_metin = {1: '1 (Ev)', 0: 'X (Berabere)', -1: '2 (Dep)'}[sonuc]
            tahminler_listesi.append({'Tarih': row['date'], 'Ev': ev, 'Dep': dep, 'Tahmin': sonuc_metin})
            
            if toplam_mac > 0: bar.progress((i + 1) / toplam_mac)
            
        bar.empty()
        st.dataframe(pd.DataFrame(tahminler_listesi), hide_index=True, use_container_width=True)
