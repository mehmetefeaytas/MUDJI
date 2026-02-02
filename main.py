import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime, timedelta # Gerçek zaman için
from numba import jit # Numba kütüphanesi eklendi

# -----------------------------------------------------------
# 1. Sabitler ve Başlangıç Koşulları
# -----------------------------------------------------------

# Simülasyon Genel Parametreleri
TIME_STEP = 0.05           # saniye (Simülasyonun hassasiyeti)
GRAVITY = 9.81             # m/s^2

# Model Uydu Fiziksel Parametreleri
KUTLE_TASIYICI = 0.2347      # kg (Taşıyıcının kütlesi)
KUTLE_GOREV_YUKU = 0.4897    # kg (Görev Yükünün kütlesi)

# Başlangıçta düşen toplam kütle (Taşıyıcı + Görev Yükü)
TOPLAM_BASLANGIC_KUTLE = KUTLE_TASIYICI + KUTLE_GOREV_YUKU # 0.7244 kg

CD_PARASUT = 1.5             # Yuvarlak paraşüt için sürüklenme katsayısı (PDR Sayfa 2)
HAVA_YOGUNLUGU = 1.225       # kg/m^3 (Deniz seviyesi, 15°C - PDR Sayfa 2)
BASINC_DENIZ_SEVIYESI = 101325 # Pascal (Deniz seviyesi standart atmosfer basıncı)

# Hedef İniş Hızları (Teorik hedefler)
HEDEF_INIS_HIZI_BASLANGIC = 13.23  # m/s (500m'den 400m'ye düşüş için hedef hız)
HEDEF_INIS_HIZI_TASIYICI = 13.00   # m/s (400m sonrası taşıyıcı için hedef hız - KULLANICI İSTEĞİ ÜZERİNE GÜNCELLENDİ)
HEDEF_INIS_HIZI_GOREV_YUKU = 7.00  # m/s (400m sonrası görev yükü için hedef hız - KULLANICI İSTEĞİ ÜZERİNE GÜNCELLENDİ)

# Paraşüt Alanları (Hedef hızlara ve verilen değerlere göre hesaplandı/atandı)
PARASUT_ALANI_BASLANGIC = 0.05 # Yükseliş ve ayrılma öncesi iniş için daha küçük bir alan

# Taşıyıcı için paraşüt alanı, yeni hedef hıza göre hesaplandı (13 m/s)
# A = (2 * m * g) / (rho * Cd * V_t^2)
# A_tas = (2 * 0.2347 * 9.81) / (1.225 * 1.5 * 13.0^2) = 4.604214 / (1.8375 * 169) = 4.604214 / 310.6875 = 0.01482
PARASUT_ALANI_TASIYICI = 0.0148  # Taşıyıcı için paraşüt alanı - GÜNCELLENDİ

# Görev Yükü için paraşüt alanı, yeni hedef hıza göre hesaplandı (7 m/s)
# A_gy = (2 * 0.4897 * 9.81) / (1.225 * 1.5 * 7.0^2) = 9.607434 / (1.8375 * 49) = 9.607434 / 90.0375 = 0.10670
PARASUT_ALANI_GOREV_YUKU = 0.1067 # Görev Yükü için paraşüt alanı - GÜNCELLENDİ

print("\n--- TEORİK HESAPLANAN DEĞERLER (Paraşüt Alanları) ---")
print(f"  Başlangıç Sistemi (Toplam Kütle {TOPLAM_BASLANGIC_KUTLE:.4f} kg) için Hesaplanan Paraşüt Alanı (Hedef {HEDEF_INIS_HIZI_BASLANGIC} m/s): {PARASUT_ALANI_BASLANGIC:.4f} m^2")
print(f"  Taşıyıcı ({KUTLE_TASIYICI:.4f} kg) için Hesaplanan Paraşüt Alanı (Hedef {HEDEF_INIS_HIZI_TASIYICI} m/s): {PARASUT_ALANI_TASIYICI:.4f} m^2")
print(f"  Görev Yükü ({KUTLE_GOREV_YUKU:.4f} kg) için Hesaplanan Paraşüt Alanı (Hedef {HEDEF_INIS_HIZI_GOREV_YUKU} m/s): {PARASUT_ALANI_GOREV_YUKU:.4f} m^2")


BASLANGIC_IRTIFASI = 0       # metre (Yerden başlayacak)
PEAK_ALTITUDE = 800          # metre (Hedeflenen maksimum irtifa, gerçek tepe noktası fiziksel olarak belirlenecek)
ASCENT_ACCELERATION = 25.0   # m/s^2 (Yükseliş ivmesi, roket motoru gibi, daha hızlı yükseliş için ayarlandı)
ASCENT_DURATION = 10.0       # saniye (Yükseliş fazının yaklaşık süresi, yakıt bitişi)

BASLANGIC_HIZI = 0           # m/s (Başlangıçta duruyor varsayımı)

# Paraşüt kapalıyken sistemlerin aerodinamik sürtünmesi (Varsayımlar)
CD_MODEL_UYDU_GOKYUZU = 0.2     # Model uydunun (veya taşıyıcının) kendi aerodinamik sürtünme katsayısı
KESIT_ALANI_MODEL_UYDU = 0.00866  # m^2 (Verilen değer) - bu başlangıçtaki entegre sistem için kullanılacak

# Sensör Hataları (Gerçekçi simülasyon için)
BASINC_SENSOR_NOISE_STD = 25    # Pascal cinsinden gürültü standart sapması (Örnek değer)
SICAKLIK_SENSOR_NOISE_STD = 0.2 # derece cinsinden gürültü standard sapması (Daha hassas)
GPS_NOISE_LAT_LON = 0.0000005   # GPS enlem/boylam için gürültü standard sapması, çok az hareket etmesi için azaltıldı
PIL_GERILIM_NOISE_STD = 0.05    # V cinsinden gürültü standard sapması
IMU_NOISE_STD = 0.1             # Derece cinsinden IMU gürültü standart sapması

# Haberleşme Parametreleri
TELEMETRI_GONDERIM_SIKLIGI = 1 # saniyede 1 kez

# Uçuş Olayları İrtifaları
AYRILMA_IRTIFASI = 400       # metre (Taşıyıcı ve Görev Yükü ayrılacak)
PARASUT_ACILMA_IRTIFASI = 400 # metre (Ayrılma ile aynı anda paraşütler açılacak)

# GPS Mesafe Hesaplamaları İçin Sabitler (Yaklaşık değerler)
METERS_PER_LAT_DEGREE = 111139  # Yaklaşık 111.139 metre/derece enlem
# Simülasyon başlangıç enlemine göre boylam dönüşüm faktörü
initial_gps_latitude_val = 40.90185 # İstanbul Gedik Üniversitesi'nin yakınları
METERS_PER_LON_DEGREE_AT_START = 111320 * np.cos(np.radians(initial_gps_latitude_val))

# --- YENİ EKLENEN PARAMETRELER ---
# Gerçek zamanlı başlangıç
START_DATETIME = datetime.now() # Simülasyonun başladığı an

# Pil Gerilimi Parametreleri
PIL_GERILIM_BASLANGIC = 11.7 # V (Kullanıcının isteği üzerine başlangıç gerilimi)
# Toplam düşüş: 11.7 - 10.2 = 1.5V. Simülasyon süresi 900s. Oran: 1.5V / 900s = 0.001666...
PIL_DEGRASIN_ORANI = 0.00167 # Volt/saniye - her saniyede düşüş oranı

# Eğim Açısı Parametreleri (IMU için)
# Düşüş sırasında rastgele salınımlar için max sapma
IMU_MAX_PITCH_ROLL = 5 # Derece (Max salınım açısı)
IMU_MAX_YAW_RATE = 10  # Derece/saniye (Max yaw dönme hızı)

# IoT İstasyonları Parametreleri
IOT_STATION_1_LAT = 40.90200 # Gedik Üniversitesi çevresi
IOT_STATION_1_LON = 29.21920
IOT_STATION_1_TEMP_BASE = 22.0 # Ortam sıcaklığı (İstanbul Gedik Üniversitesi civarı için örnek)
IOT_STATION_1_TEMP_VARIATION = 0.5 # +/- derece, rastgele varyasyon

IOT_STATION_2_LAT = 40.90150 # Biraz farklı bir konum
IOT_STATION_2_LON = 29.21880
IOT_STATION_2_TEMP_BASE = 21.5 # Hafif farklı ortam sıcaklığı
IOT_STATION_2_TEMP_VARIATION = 0.7 # +/- derece, rastgele varyasyon


# -----------------------------------------------------------
# 2. Durum Değişkenleri ve Veri Toplama
# -----------------------------------------------------------
current_time = 0.0
simulation_step_count = 0 # Simülasyon adım sayacı

# Başlangıçta düşen tek sistem (Taşıyıcı + Görev Yükü)
altitude_initial_system = BASLANGIC_IRTIFASI
velocity_initial_system = BASLANGIC_HIZI

# Ayrıldıktan sonraki sistemler için değişkenler
altitude_tas = None # Taşıyıcı irtifası
velocity_tas = None # Taşıyıcı hızı
parasut_acik_tas = False

altitude_gy = None # Görev Yükü irtifası
velocity_gy = None # Görev Yükü hızı
parasut_acik_gy = False

ayrilma_gerceklesti = False  # Ayrılma mekanizmasının durumu
gorev_yuku_aktif = False     # Görev yükünün aktif olup olmadığı (Ayrıldıktan sonra çalışır)

# GPS koordinatları
initial_gps_latitude = 40.90185 # Fırlatma noktası enlem
initial_gps_longitude = 29.21913 # Fırlatma noktası boylam

current_gps_latitude_gy = initial_gps_latitude # Görev Yükü'nün anlık GPS enlemi
current_gps_longitude_gy = initial_gps_longitude # Görev Yükü'nün anlık GPS boylamı

current_gps_latitude_tas = initial_gps_latitude # Taşıyıcı'nın anlık GPS enlemi
current_gps_longitude_tas = initial_gps_longitude # Taşıyıcı'nın anlık GPS boylamı

# --- YENİ EKLENEN DURUM DEĞİŞKENLERİ ---
current_battery_voltage = PIL_GERILIM_BASLANGIC # Anlık pil gerilimi
current_pitch = 0.0 # Anlık pitch açısı
current_roll = 0.0  # Anlık roll açısı
current_yaw = 0.0   # Anlık yaw açısı

# Uçuş fazı takibi
FLIGHT_PHASE = 'ASCENT' # 'ASCENT', 'COAST', 'DESCENT_PRE_SEPARATION', 'DESCENT_POST_SEPARATION'


# Simülasyon verilerini depolamak için listeler
# Veri depolama sıklığı TELEMETRI_GONDERIM_SIKLIGI ile aynı olacak şekilde optimize edildi.
# Bu, grafiklerin daha az veri noktasına sahip olmasına neden olabilir ancak performansı artırır.
time_data = []
altitude_initial_system_data = []
velocity_initial_system_data = []

altitude_tas_data = []
velocity_tas_data = []

altitude_gy_data = []
velocity_gy_data = []

sim_telemetry_data = [] # Simülasyon tarafından üretilen telemetri

# Ortalam hızları hesaplamak için hız verileri
velocities_initial_system_flight = []
velocities_tas_flight = []
velocities_gy_flight = []

# GPS verilerini toplamak için yeni listeler
gps_latitude_data = []
gps_longitude_data = []
gps_altitude_data = []


# -----------------------------------------------------------
# 3. Alt Sistem Modelleri ve Fonksiyonlar
# -----------------------------------------------------------

# Numba ile derleme için @jit dekoratörü eklendi
@jit(nopython=True) # nopython=True, sadece NumPy ve temel Python tipleriyle çalışmayı zorlar, daha hızlı
def calculate_drag_force(v, current_mass, drag_area_factor, current_alt):
    """
    Sürüklenme kuvvetini hesaplar.
    v: Anlık hız (m/s)
    current_mass: Anlık kütle (kg) - doğrudan kullanılmasa da parametre uyumu için tutuldu
    drag_area_factor: Sürtünme alanı katsayısı (Paraşüt alanı * CD veya Kesit alanı * CD)
    current_alt: Anlık irtifa (m)
    """
    if current_alt <= 0 or v <= 0: # Yere çarptıysa veya hız yoksa sürüklenme olmaz
        return 0.0 # Numba için float döndürmek önemli

    # Hava yoğunluğu irtifaya göre değişir (Basit üstel azalım modeli)
    density_at_altitude = HAVA_YOGUNLUGU * np.exp(-current_alt / 8000.0) # Numba için float değişmezler
    
    drag_force = 0.5 * density_at_altitude * v**2 * drag_area_factor
    return drag_force

# Numba ile derleme için @jit dekoratörü eklendi
@jit(nopython=True)
def update_physics(dt, current_alt, current_vel, current_mass, drag_area_open, drag_area_closed, parachute_open, phase_int, thrust_active):
    """
    Tek bir cismin fiziksel durumunu (irtifa, hız) günceller.
    dt: Zaman adımı (s)
    current_alt: Anlık irtifa (m)
    current_vel: Anlık hız (m/s)
    current_mass: Cismin kütlesi (kg)
    drag_area_open: Paraşüt açıkken sürtünme alanı katsayısı (Alan * CD)
    drag_area_closed: Paraşüt kapalıyken sürtünme alanı katsayısı (Kesit Alanı * CD)
    parachute_open: Paraşütün açık olup olmadığı (boolean)
    phase_int: Mevcut uçuş fazı (int olarak temsil edilecek: 0=ASCENT, 1=COAST, 2=DESCENT_PRE_SEPARATION, 3=DESCENT_POST_SEPARATION)
    thrust_active: İtki kuvvetinin aktif olup olmadığı (boolean)
    """
    if current_alt <= 0.0 and current_vel == 0.0 and phase_int != 0: # Zaten yere inmişse ve yükseliş fazında değilse güncelleme yapma
        return 0.0, 0.0

    # Kuvvetler (pozitif = yukarı, negatif = aşağı)
    gravity_force = -current_mass * GRAVITY # Yerçekimi her zaman aşağı yönlü
    
    thrust_force = 0.0
    if thrust_active:
        thrust_force = current_mass * ASCENT_ACCELERATION # İtki her zaman yukarı yönlü

    drag_force_magnitude = calculate_drag_force(abs(current_vel), current_mass, 
                                                drag_area_open if parachute_open else drag_area_closed, 
                                                current_alt)
    
    drag_force = 0.0
    if current_vel > 0: # Yukarı doğru hareket ediyorsa, sürüklenme aşağı yönlü
        drag_force = -drag_force_magnitude
    elif current_vel < 0: # Aşağı doğru hareket ediyorsa, sürüklenme yukarı yönlü
        drag_force = drag_force_magnitude
    # current_vel 0 ise, sürüklenme kuvveti de 0.0

    net_force = thrust_force + gravity_force + drag_force
    
    acceleration = net_force / current_mass
    new_velocity = current_vel + acceleration * dt
    new_altitude = current_alt + new_velocity * dt # Hızın işaretine göre irtifa artar veya azalır

    # Yere çarpma kontrolü
    if new_altitude <= 0.0:
        new_altitude = 0.0
        new_velocity = 0.0 # Yere çarptığında durur

    return new_altitude, new_velocity

# Numba ile derleme için @jit dekoratörü eklendi
@jit(nopython=True)
def altitude_to_pressure(altitude):
    """
    İrtifayı atmosferik basınca çevirir (Uluslararası Standart Atmosfer modeli, basit).
    Pascal birimindedir.
    Deniz seviyesi basıncı ve sıcaklığına göre basit bir model.
    """
    if altitude < 0.0:
        return float(BASINC_DENIZ_SEVIYESI)
    
    H_scale = 8000.0 # Yaklaşık ölçek yüksekliği (metre)
    pressure = BASINC_DENIZ_SEVIYESI * np.exp(-altitude / H_scale)
    return float(pressure)

# Numba ile derleme için @jit dekoratörü eklendi
@jit(nopython=True)
def simulate_sensor_data_numba(actual_altitude, current_vel, current_lat, current_lon, current_pitch, current_roll, current_yaw, current_voltage, is_gy_sensor, PEAK_ALTITUDE_VAL, BASINC_SENSOR_NOISE_STD_VAL, SICAKLIK_SENSOR_NOISE_STD_VAL, GPS_NOISE_LAT_LON_VAL, PIL_GERILIM_NOISE_STD_VAL, IMU_NOISE_STD_VAL, BASINC_DENIZ_SEVIYESI_VAL):
    """
    Sensörlerden gelen verileri simüle eder (gürültü dahil).
    Numba'da global değişkenlere doğrudan erişim yerine parametre olarak geçilmesi tercih edilir.
    """
    # Basınç Sensörü (İrtifa ölçümü)
    actual_pressure = altitude_to_pressure(actual_altitude)
    measured_pressure = actual_pressure + np.random.normal(0.0, BASINC_SENSOR_NOISE_STD_VAL)

    # Sıcaklık Sensörü (Uçuş süresince sıcaklık değişimi modellemesi)
    if is_gy_sensor:
        temp_at_peak = 17.0
        temp_at_sea_level = 22.0
        
        A_coeff = (temp_at_sea_level - temp_at_peak) / (PEAK_ALTITUDE_VAL**2)
        simulated_temp = A_coeff * (actual_altitude - PEAK_ALTITUDE_VAL)**2 + temp_at_peak
        
        # np.clip yerine manuel kırpma
        if simulated_temp < temp_at_peak:
            simulated_temp = temp_at_peak
        elif simulated_temp > temp_at_sea_level:
            simulated_temp = temp_at_sea_level
            
        simulated_temp += np.random.normal(0.0, SICAKLIK_SENSOR_NOISE_STD_VAL)
        
    else:
        simulated_temp = 15.0 - (actual_altitude / 150.0) + np.random.normal(0.0, SICAKLIK_SENSOR_NOISE_STD_VAL)
    
    # GPS Sensörü (Gürültü)
    measured_lat = current_lat + np.random.normal(0.0, GPS_NOISE_LAT_LON_VAL)
    measured_lon = current_lon + np.random.normal(0.0, GPS_NOISE_LAT_LON_VAL)
    measured_gps_altitude = actual_altitude + np.random.normal(0.0, BASINC_SENSOR_NOISE_STD_VAL * 2.0)

    # Pil Gerilimi
    measured_voltage = current_voltage + np.random.normal(0.0, PIL_GERILIM_NOISE_STD_VAL)

    # Eğim Açıları (IMU)
    measured_pitch = current_pitch + np.random.normal(0.0, IMU_NOISE_STD_VAL)
    measured_roll = current_roll + np.random.normal(0.0, IMU_NOISE_STD_VAL)
    measured_yaw = current_yaw + np.random.normal(0.0, IMU_NOISE_STD_VAL)
    
    return (max(0.0, round(actual_altitude, 2)), # Numba'da tuple döndürmek daha uygun
            max(0.0, round(measured_pressure, 2)),
            round(simulated_temp, 2),
            round(current_vel, 2),
            round(measured_lat, 6),
            round(measured_lon, 6),
            max(0.0, round(measured_gps_altitude, 2)),
            round(measured_voltage, 2),
            round(measured_pitch, 2),
            round(measured_roll, 2),
            round(measured_yaw, 2))

# simulate_sensor_data wrapper fonksiyonu
def simulate_sensor_data(actual_altitude, current_vel, current_lat, current_lon, current_pitch, current_roll, current_yaw, current_voltage, sim_time, is_gy_sensor=False):
    """
    Sensörlerden gelen verileri simüle eder (gürültü dahil).
    Numba fonksiyonunu çağırır ve sonuçları bir sözlük olarak döndürür.
    """
    # Numba fonksiyonuna sabitleri parametre olarak geç
    result_tuple = simulate_sensor_data_numba(actual_altitude, current_vel, current_lat, current_lon, current_pitch, current_roll, current_yaw, current_voltage, is_gy_sensor, float(PEAK_ALTITUDE), float(BASINC_SENSOR_NOISE_STD), float(SICAKLIK_SENSOR_NOISE_STD), float(GPS_NOISE_LAT_LON), float(PIL_GERILIM_NOISE_STD), float(IMU_NOISE_STD), float(BASINC_DENIZ_SEVIYESI))
    
    return {
        "irtifa": result_tuple[0],
        "basinc": result_tuple[1],
        "sicaklik": result_tuple[2],
        "hiz": result_tuple[3],
        "gps_enlem": result_tuple[4],
        "gps_boylam": result_tuple[5],
        "gps_irtifa": result_tuple[6],
        "pil_gerilimi": result_tuple[7],
        "pitch": result_tuple[8],
        "roll": result_tuple[9],
        "yaw": result_tuple[10]
    }


def simulate_iot_station_temp(base_temp, variation_std):
    """
    IoT istasyonu sıcaklık verilerini simüle eder.
    """
    return base_temp + np.random.normal(0, variation_std)

def calculate_distance_from_origin(lat, lon, origin_lat, origin_lon):
    """ Başlangıç noktasından olan mesafeyi metre cinsinden hesaplar."""
    delta_lat = lat - origin_lat
    delta_lon = lon - origin_lon

    dist_lat = delta_lat * METERS_PER_LAT_DEGREE
    dist_lon = delta_lon * METERS_PER_LON_DEGREE_AT_START # Basit düzlem yaklaşımı

    distance = np.sqrt(dist_lat**2 + dist_lon**2)
    return distance

def generate_telemetry(sim_time, alt_gy, vel_gy, alt_tas, vel_tas,
                         sensor_data_gy, sensor_data_tas, gy_aktif, par_acik_gy, par_acik_tas, ayrilma,
                         iot1_temp, iot2_temp):
    """
    Telemetri paketini oluşturur.
    """
    # Gerçek zamanı hesapla
    current_datetime = START_DATETIME + timedelta(seconds=sim_time)
    real_time_str = current_datetime.strftime("%d/%m/%Y %H:%M:%S")

    # Fırlatma noktasından mesafeler
    dist_gy_from_origin = calculate_distance_from_origin(sensor_data_gy["gps_enlem"], sensor_data_gy["gps_boylam"], initial_gps_latitude, initial_gps_longitude)
    dist_tas_from_origin = calculate_distance_from_origin(sensor_data_tas["gps_enlem"], sensor_data_tas["gps_boylam"], initial_gps_latitude, initial_gps_longitude)

    telemetry_packet = {
        "gercek_zaman": real_time_str, # Gerçek zaman formatında
        "sim_timestamp": round(sim_time, 1),

        "gy_irtifa_m": sensor_data_gy["irtifa"],
        "gy_basinc_pa": sensor_data_gy["basinc"], # Yeni
        "gy_hiz_mps": sensor_data_gy["hiz"],
        "gy_sicaklik_c": sensor_data_gy["sicaklik"],
        "gy_pil_gerilimi_v": sensor_data_gy["pil_gerilimi"], # Yeni
        "gy_gps_enlem": sensor_data_gy["gps_enlem"],
        "gy_gps_boylam": sensor_data_gy["gps_boylam"],
        "gy_gps_irtifa_m": sensor_data_gy["gps_irtifa"], # Yeni (GPS'ten alınan irtifa)
        "gy_pitch_derece": sensor_data_gy["pitch"], # Yeni
        "gy_roll_derece": sensor_data_gy["roll"],   # Yeni
        "gy_yaw_derece": sensor_data_gy["yaw"],     # Yeni
        "gy_gorev_yuku_durum": 1 if gy_aktif else 0, # 1: Aktif, 0: Pasif
        "gy_parasu_durum": 1 if par_acik_gy else 0,  # 1: Açık, 0: Kapalı
        "gy_firlatma_mesafe_m": round(dist_gy_from_origin, 2),

        "tas_irtifa_m": sensor_data_tas["irtifa"],
        "tas_basinc_pa": sensor_data_tas["basinc"], # Yeni
        "tas_hiz_mps": sensor_data_tas["hiz"],
        "tas_sicaklik_c": sensor_data_tas["sicaklik"],
        "tas_pil_gerilimi_v": sensor_data_tas["pil_gerilimi"], # Yeni
        "tas_gps_enlem": sensor_data_tas["gps_enlem"],
        "tas_gps_boylam": sensor_data_tas["gps_boylam"],
        "tas_gps_irtifa_m": sensor_data_tas["gps_irtifa"], # Yeni (GPS'ten alınan irtifa)
        "tas_pitch_derece": sensor_data_tas["pitch"], # Yeni
        "tas_roll_derece": sensor_data_tas["roll"],   # Yeni
        "tas_yaw_derece": sensor_data_tas["yaw"],     # Yeni
        "tas_parasu_durum": 1 if par_acik_tas else 0, # 1: Açık, 0: Kapalı
        "tas_firlatma_mesafe_m": round(dist_tas_from_origin, 2),

        "ayrilma_durum": 1 if ayrilma else 0, # 1: Gerçekleşti, 0: Bekleniyor
        "irtifa_farki_m": round(abs((alt_gy if alt_gy is not None else 0) - (alt_tas if alt_tas is not None else 0)), 2),
        # "aralarindaki_mesafe_m": round(dist_between_systems, 2), # Kaldırıldı

        "iot1_sicaklik_c": round(iot1_temp, 2), # Yeni
        "iot2_sicaklik_c": round(iot2_temp, 2)  # Yeni
    }
    # Basit bir checksum hesaplaması (tüm sayısal değerlerin toplamı mod 256)
    checksum_val = 0
    for key, value in telemetry_packet.items():
        if isinstance(value, (int, float)):
            checksum_val += int(value * 100) if isinstance(value, float) else value
    telemetry_packet["checksum"] = checksum_val % 256

    return telemetry_packet

def process_telecommand(command, current_state):
    """
    Yer istasyonundan gelen telekomutları işler.
    """
    response_message = ""
    # Sadece manuel ayrılma komutu (yedekleme senaryosu)
    if command == "AYRILMA_MANUEL": # Ayrılma manuel komutu
        if not current_state["ayrilma_gerceklesti"]:
            current_state["ayrilma_gerceklesti"] = True
            current_state["gorev_yuku_aktif"] = True # Manuel ayrılmada da görev yükü aktifleşsin
            response_message = "Manuel Ayrılma Başlatıldı ve Görev Yükü Aktif Edildi."
            print(f"[{current_state['current_time']:.1f}s] Telekomut alındı: {response_message}")
        else:
            response_message = "Ayrılma Zaten Gerçekleşti."
    else:
        response_message = "Bilinmeyen veya Desteklenmeyen Komut."
    return current_state, response_message

# -----------------------------------------------------------
# 4. Simülasyon Döngüsü
# -----------------------------------------------------------

print("\n--- Model Uydu Dijital İkiz Simülasyonu Başlıyor ---")

# Manuel ayrılma komutunun tetiklenip tetiklenmediğini takip etmek için bayrak
manuel_ayrilma_tetiklendi = False

# Uçuş fazı stringlerini int karşılıklarına dönüştür (Numba için)
PHASE_MAP = {'ASCENT': 0, 'COAST': 1, 'DESCENT_PRE_SEPARATION': 2, 'DESCENT_POST_SEPARATION': 3}
REVERSE_PHASE_MAP = {0: 'ASCENT', 1: 'COAST', 2: 'DESCENT_PRE_SEPARATION', 3: 'DESCENT_POST_SEPARATION'}

current_flight_phase_int = PHASE_MAP[FLIGHT_PHASE] # Başlangıç fazını int olarak ayarla

# Simülasyonu uydu yere inene kadar çalıştır
while True:
    current_time = simulation_step_count * TIME_STEP

    # --- Pil Gerilimi Güncellemesi ---
    if current_battery_voltage > 0:
        current_battery_voltage = max(0.0, current_battery_voltage - PIL_DEGRASIN_ORANI * TIME_STEP)
    
    # --- Eğim Açıları (Pitch, Roll, Yaw) Güncellemesi ---
    current_pitch += np.random.normal(0.0, IMU_NOISE_STD * 0.5)
    current_roll += np.random.normal(0.0, IMU_NOISE_STD * 0.5)
    current_yaw += np.random.normal(0.0, IMU_NOISE_STD * 0.5)
    
    # Numba uyumluluğu için np.clip yerine manuel kırpma
    current_pitch = max(-IMU_MAX_PITCH_ROLL, min(current_pitch, IMU_MAX_PITCH_ROLL))
    current_roll = max(-IMU_MAX_PITCH_ROLL, min(current_roll, IMU_MAX_PITCH_ROLL))
    current_yaw = current_yaw % 360.0

    if ayrilma_gerceklesti:
        current_pitch = current_pitch * 0.95
        current_roll = current_roll * 0.95
        current_yaw += np.random.normal(0.0, IMU_NOISE_STD * 0.1)
        current_yaw = current_yaw % 360.0

    # 4.1. Fizik Güncellemesi (Uçuş Fazına Göre)
    if current_flight_phase_int == PHASE_MAP['ASCENT']:
        altitude_initial_system, velocity_initial_system = update_physics(
            TIME_STEP, altitude_initial_system, velocity_initial_system,
            TOPLAM_BASLANGIC_KUTLE,
            0.0, # Paraşüt açık değil
            KESIT_ALANI_MODEL_UYDU * CD_MODEL_UYDU_GOKYUZU, # Kendi aerodinamik sürtünmesi
            False, # Paraşüt kapalı
            current_flight_phase_int,
            True # İtki aktif
        )
        velocities_initial_system_flight.append(velocity_initial_system)

        # Yakıt bitişi ve süzülme fazına geçiş
        if current_time >= ASCENT_DURATION:
            current_flight_phase_int = PHASE_MAP['COAST']
            print(f"[{current_time:.1f}s] Olay: Roket Yakıtı Bitti, Yükseliş İvmesi Durdu. Süzülme Başlıyor.")

    elif current_flight_phase_int == PHASE_MAP['COAST']:
        altitude_initial_system, velocity_initial_system = update_physics(
            TIME_STEP, altitude_initial_system, velocity_initial_system,
            TOPLAM_BASLANGIC_KUTLE,
            0.0, # Paraşüt açık değil
            KESIT_ALANI_MODEL_UYDU * CD_MODEL_UYDU_GOKYUZU, # Kendi aerodinamik sürtünmesi
            False, # Paraşüt kapalı
            current_flight_phase_int,
            False # İtki pasif
        )
        velocities_initial_system_flight.append(velocity_initial_system)

        # Hız sıfıra veya negatife düştüğünde iniş fazına geçiş (apogee aşıldı)
        if velocity_initial_system <= 0.0 and altitude_initial_system > 0.0:
            current_flight_phase_int = PHASE_MAP['DESCENT_PRE_SEPARATION']
            print(f"[{current_time:.1f}s] Olay: Apogee Aşıldı, İniş Başlıyor. (Şu anki İrtifa: {altitude_initial_system:.2f}m)")

    elif current_flight_phase_int == PHASE_MAP['DESCENT_PRE_SEPARATION']:
        altitude_initial_system, velocity_initial_system = update_physics(
            TIME_STEP, altitude_initial_system, velocity_initial_system,
            TOPLAM_BASLANGIC_KUTLE,
            PARASUT_ALANI_BASLANGIC * CD_PARASUT, # Bu fazda paraşüt kapalı düşüyor varsayımı
            KESIT_ALANI_MODEL_UYDU * CD_MODEL_UYDU_GOKYUZU,
            False,
            current_flight_phase_int,
            False # İtki pasif
        )
        velocities_initial_system_flight.append(velocity_initial_system)

        # Otomatik Ayrılma ve Paraşüt Açılma Kontrolü
        if altitude_initial_system <= AYRILMA_IRTIFASI and not ayrilma_gerceklesti:
            ayrilma_gerceklesti = True
            gorev_yuku_aktif = True
            print(f"[{current_time:.1f}s] Olay: OTOMATİK Ayrılma Gerçekleşti (Hedef İrtifa: {AYRILMA_IRTIFASI}m)!")

            altitude_tas = altitude_initial_system
            velocity_tas = velocity_initial_system
            parasut_acik_tas = True

            altitude_gy = altitude_initial_system
            velocity_gy = velocity_initial_system
            parasut_acik_gy = True
            print(f"[{current_time:.1f}s] Olay: Taşıyıcı ve Görev Yükü Paraşütleri Açıldı!")

            altitude_initial_system = -1.0 # Başlangıç sistemini artık izleme (Numba için float)
            current_flight_phase_int = PHASE_MAP['DESCENT_POST_SEPARATION']

    elif current_flight_phase_int == PHASE_MAP['DESCENT_POST_SEPARATION']:
        # Taşıyıcı güncellemesi
        if altitude_tas is not None and altitude_tas > 0.0:
            altitude_tas, velocity_tas = update_physics(
                TIME_STEP, altitude_tas, velocity_tas,
                KUTLE_TASIYICI, PARASUT_ALANI_TASIYICI * CD_PARASUT,
                0.0, # Kapalı hali düşünülmeyecek, hep paraşütlü
                parasut_acik_tas,
                current_flight_phase_int,
                False
            )
            velocities_tas_flight.append(velocity_tas)
        elif altitude_tas is not None and altitude_tas <= 0.0:
            altitude_tas = 0.0
            velocity_tas = 0.0

        # Görev Yükü güncellemesi
        if altitude_gy is not None and altitude_gy > 0.0:
            altitude_gy, velocity_gy = update_physics(
                TIME_STEP, altitude_gy, velocity_gy,
                KUTLE_GOREV_YUKU, PARASUT_ALANI_GOREV_YUKU * CD_PARASUT,
                0.0, # Kapalı hali düşünülmeyecek, hep paraşütlü
                parasut_acik_gy,
                current_flight_phase_int,
                False
            )
            velocities_gy_flight.append(velocity_gy)
        elif altitude_gy is not None and altitude_gy <= 0.0:
            altitude_gy = 0.0
            velocity_gy = 0.0

    # 4.2. GPS Güncellemesi (Hava koşulları/rüzgar etkisi ile yatayda sürüklenme)
    horizontal_speed_factor = 0.0000005

    if current_flight_phase_int == PHASE_MAP['ASCENT'] or current_flight_phase_int == PHASE_MAP['COAST'] or current_flight_phase_int == PHASE_MAP['DESCENT_PRE_SEPARATION']:
        if velocity_initial_system > 0.0 or current_flight_phase_int == PHASE_MAP['COAST']: # Sadece hareket varsa
            current_gps_latitude_gy += np.random.normal(0.0, GPS_NOISE_LAT_LON) * (abs(velocity_initial_system) * horizontal_speed_factor) * TIME_STEP
            current_gps_longitude_gy += np.random.normal(0.0, GPS_NOISE_LAT_LON) * (abs(velocity_initial_system) * horizontal_speed_factor * 1.5) * TIME_STEP
            current_gps_latitude_tas = current_gps_latitude_gy
            current_gps_longitude_tas = current_gps_longitude_gy
    elif current_flight_phase_int == PHASE_MAP['DESCENT_POST_SEPARATION']:
        if velocity_gy is not None and velocity_gy != 0.0:
            current_gps_latitude_gy += np.random.normal(0.0, GPS_NOISE_LAT_LON) * (abs(velocity_gy) * horizontal_speed_factor * 1.1) * TIME_STEP
            current_gps_longitude_gy += np.random.normal(0.0, GPS_NOISE_LAT_LON) * (abs(velocity_gy) * horizontal_speed_factor * 0.9) * TIME_STEP
        if velocity_tas is not None and velocity_tas != 0.0:
            current_gps_latitude_tas += np.random.normal(0.0, GPS_NOISE_LAT_LON) * (abs(velocity_tas) * horizontal_speed_factor * 0.8) * TIME_STEP
            current_gps_longitude_tas += np.random.normal(0.0, GPS_NOISE_LAT_LON) * (abs(velocity_tas) * horizontal_speed_factor * 1.2) * TIME_STEP

    # 4.3. Sensör Verisi Simülasyonu
    if current_flight_phase_int == PHASE_MAP['ASCENT'] or current_flight_phase_int == PHASE_MAP['COAST'] or current_flight_phase_int == PHASE_MAP['DESCENT_PRE_SEPARATION']:
        sensor_readings_gy = simulate_sensor_data(altitude_initial_system, velocity_initial_system, 
                                                  current_gps_latitude_gy, current_gps_longitude_gy,
                                                  current_pitch, current_roll, current_yaw, current_battery_voltage, current_time, is_gy_sensor=True)
        sensor_readings_tas = simulate_sensor_data(altitude_initial_system, velocity_initial_system, 
                                                   current_gps_latitude_tas, current_gps_longitude_tas,
                                                   current_pitch, current_roll, current_yaw, current_battery_voltage, current_time, is_gy_sensor=False)
    elif current_flight_phase_int == PHASE_MAP['DESCENT_POST_SEPARATION']:
        sensor_readings_gy = simulate_sensor_data(altitude_gy if altitude_gy is not None else 0.0, velocity_gy if velocity_gy is not None else 0.0, 
                                                  current_gps_latitude_gy, current_gps_longitude_gy,
                                                  current_pitch, current_roll, current_yaw, current_battery_voltage, current_time, is_gy_sensor=True)
        sensor_readings_tas = simulate_sensor_data(altitude_tas if altitude_tas is not None else 0.0, velocity_tas if velocity_tas is not None else 0.0, 
                                                   current_gps_latitude_tas, current_gps_longitude_tas,
                                                   current_pitch, current_roll, current_yaw, current_battery_voltage, current_time, is_gy_sensor=False)

    # --- IoT İstasyonu Sıcaklıkları ---
    iot1_temp = simulate_iot_station_temp(IOT_STATION_1_TEMP_BASE, IOT_STATION_1_TEMP_VARIATION)
    iot2_temp = simulate_iot_station_temp(IOT_STATION_2_TEMP_BASE, IOT_STATION_2_TEMP_VARIATION)


    # 4.4. Telemetri Oluşturma ve Veri Depolama (Belirli aralıklarla)
    if abs(current_time % TELEMETRI_GONDERIM_SIKLIGI) < (TIME_STEP / 2) or simulation_step_count == 0:
        if current_flight_phase_int == PHASE_MAP['ASCENT'] or current_flight_phase_int == PHASE_MAP['COAST'] or current_flight_phase_int == PHASE_MAP['DESCENT_PRE_SEPARATION']:
            telemetry_packet = generate_telemetry(current_time, altitude_initial_system, velocity_initial_system,
                                                    altitude_initial_system, velocity_initial_system, # Aynı veriler
                                                    sensor_readings_gy, sensor_readings_tas,
                                                    gorev_yuku_aktif, False, False, ayrilma_gerceklesti,
                                                    iot1_temp, iot2_temp)
            # Sadece telemetri gönderildiği zaman verileri depola
            time_data.append(current_time)
            altitude_initial_system_data.append(altitude_initial_system)
            velocity_initial_system_data.append(velocity_initial_system)
            altitude_tas_data.append(np.nan)
            velocity_tas_data.append(np.nan)
            altitude_gy_data.append(np.nan)
            velocity_gy_data.append(np.nan)

        elif current_flight_phase_int == PHASE_MAP['DESCENT_POST_SEPARATION']:
            telemetry_packet = generate_telemetry(current_time, altitude_gy, velocity_gy,
                                                    altitude_tas, velocity_tas,
                                                    sensor_readings_gy, sensor_readings_tas,
                                                    gorev_yuku_aktif, parasut_acik_gy, parasut_acik_tas, ayrilma_gerceklesti,
                                                    iot1_temp, iot2_temp)
            # Sadece telemetri gönderildiği zaman verileri depola
            time_data.append(current_time)
            altitude_initial_system_data.append(np.nan)
            velocity_initial_system_data.append(np.nan)
            altitude_tas_data.append(altitude_tas)
            velocity_tas_data.append(velocity_tas)
            altitude_gy_data.append(altitude_gy)
            velocity_gy_data.append(velocity_gy)

        sim_telemetry_data.append(telemetry_packet)

        gps_latitude_data.append(sensor_readings_gy["gps_enlem"])
        gps_longitude_data.append(sensor_readings_gy["gps_boylam"])
        gps_altitude_data.append(sensor_readings_gy["gps_irtifa"])


    # 4.5. Telekomut Simülasyonu (Manuel ayrılma komutu)
    if current_flight_phase_int == PHASE_MAP['DESCENT_PRE_SEPARATION'] and altitude_initial_system <= AYRILMA_IRTIFASI - 10 and not manuel_ayrilma_tetiklendi:
        current_state_for_tc = {
            "current_time": current_time,
            "ayrilma_gerceklesti": ayrilma_gerceklesti,
            "gorev_yuku_aktif": gorev_yuku_aktif
        }
        updated_state, response = process_telecommand("AYRILMA_MANUEL", current_state_for_tc)
        ayrilma_gerceklesti = updated_state["ayrilma_gerceklesti"]
        gorev_yuku_aktif = updated_state["gorev_yuku_aktif"]
        manuel_ayrilma_tetiklendi = True

        if ayrilma_gerceklesti:
            altitude_tas = altitude_initial_system
            velocity_tas = velocity_initial_system
            parasut_acik_tas = True

            altitude_gy = altitude_initial_system
            velocity_gy = velocity_initial_system
            parasut_acik_gy = True
            print(f"[{current_time:.1f}s] MANUEL KOMUT İLE Taşıyıcı ve Görev Yükü Paraşütleri Açıldı!")
            altitude_initial_system = -1.0 # Başlangıç sistemini artık izleme
            current_flight_phase_int = PHASE_MAP['DESCENT_POST_SEPARATION']


    # İniş kontrolü: Tüm aktif parçalar yere indiğinde döngüyü kır
    if current_flight_phase_int == PHASE_MAP['ASCENT'] or current_flight_phase_int == PHASE_MAP['COAST'] or current_flight_phase_int == PHASE_MAP['DESCENT_PRE_SEPARATION']:
        if altitude_initial_system <= 0.0 and velocity_initial_system == 0.0:
            print(f"[{current_time:.1f}s] Simülasyon Durdu: Başlangıç Sistemi Başarılı Bir Şekilde İniş Yaptı (Ayrılma Gerçekleşmedi).")
            break
    elif current_flight_phase_int == PHASE_MAP['DESCENT_POST_SEPARATION']:
        tas_landed = (altitude_tas <= 0.0 and velocity_tas == 0.0) if altitude_tas is not None else False
        gy_landed = (altitude_gy <= 0.0 and velocity_gy == 0.0) if altitude_gy is not None else False

        if tas_landed and gy_landed:
            print(f"[{current_time:.1f}s] Simülasyon Durdu: Taşıyıcı ve Görev Yükü Başarılı Bir Şekilde İniş Yaptı.")
            break
    
    simulation_step_count += 1

print("\n--- Simülasyon Tamamlandı ---")

# -----------------------------------------------------------
# 5. Veri Analizi ve Görselleştirme
# -----------------------------------------------------------

# Telemetri verilerini DataFrame'e dönüştür (daha kolay analiz için)
telemetry_df = pd.DataFrame(sim_telemetry_data)
print("\n--- SİMÜLE EDİLMİŞ TÜM TELEMETRİ VERİLERİ (Baştan Sona) ---")
print(telemetry_df.to_string())

# Grafikler
plt.figure(figsize=(18, 12))

# İrtifa Grafiği (Yükseliş ve İniş Fazları ile)
plt.subplot(2, 1, 1)
plt.plot(time_data, altitude_initial_system_data, label='Başlangıç Sistemi İrtifa', color='gray', linestyle='--', alpha=0.7)
plt.plot(time_data, altitude_tas_data, label='Taşıyıcı İrtifa', color='blue')
plt.plot(time_data, altitude_gy_data, label='Görev Yükü İrtifa', color='green')
plt.axhline(y=PEAK_ALTITUDE, color='darkgreen', linestyle=':', label=f'Hedef Maksimum İrtifa ({PEAK_ALTITUDE}m)')
plt.axhline(y=AYRILMA_IRTIFASI, color='r', linestyle=':', label=f'Ayrılma İrtifası ({AYRILMA_IRTIFASI}m)')
plt.title('İrtifa Değişimi', fontsize=16)
plt.xlabel('Zaman (s)', fontsize=12)
plt.ylabel('İrtifa (m)', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.ylim(bottom=0)

# Hız Grafiği
plt.subplot(2, 1, 2)
plt.plot(time_data, velocity_initial_system_data, label='Başlangıç Sistemi Hız', color='gray', linestyle='--', alpha=0.7)
plt.plot(time_data, velocity_tas_data, label='Taşıyıcı Hız', color='orange')
plt.plot(time_data, velocity_gy_data, label='Görev Yükü Hız', color='purple')
plt.axhline(y=HEDEF_INIS_HIZI_BASLANGIC, color='gray', linestyle=':', label=f'Hedef Başlangıç Hızı ({HEDEF_INIS_HIZI_BASLANGIC} m/s)')
plt.axhline(y=HEDEF_INIS_HIZI_TASIYICI, color='orange', linestyle=':', label=f'Hedef Taşıyıcı Hızı ({HEDEF_INIS_HIZI_TASIYICI} m/s)')
plt.axhline(y=HEDEF_INIS_HIZI_GOREV_YUKU, color='purple', linestyle=':', label=f'Hedef Görev Yükü Hızı ({HEDEF_INIS_HIZI_GOREV_YUKU} m/s)')
plt.title('Hız Değişimi', fontsize=16)
plt.xlabel('Zaman (s)', fontsize=12)
plt.ylabel('Hız (m/s)', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.ylim(bottom=min(0, np.nanmin(velocity_initial_system_data))) # Hızın negatif olabileceği için y ekseni alt sınırı ayarlandı

plt.tight_layout()
plt.show()

# Sensörden Alınan Yükseklik Verileri Grafiği
plt.figure(figsize=(15, 8))
if not telemetry_df['gy_irtifa_m'].isnull().all():
    plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_irtifa_m'], marker='o', linestyle='-', markersize=2, label='Görev Yükü Yükseklik (Sensör)')
if not telemetry_df['tas_irtifa_m'].isnull().all():
    plt.plot(telemetry_df['sim_timestamp'], telemetry_df['tas_irtifa_m'], marker='x', linestyle='--', markersize=2, label='Taşıyıcı Yükseklik (Sensör)')
plt.title('Sensörden Alınan Yükseklik Verileri', fontsize=16)
plt.xlabel('Zaman (s)', fontsize=12)
plt.ylabel('Yükseklik (m)', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.ylim(bottom=0)
plt.show()


# Yeni Eklenen Grafikler
plt.figure(figsize=(18, 12))

# Basınç Değişimi Grafiği (U şeklinde olacak)
plt.subplot(2, 2, 1)
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_basinc_pa'], label='Görev Yükü Basıncı', color='darkred')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['tas_basinc_pa'], label='Taşıyıcı Basıncı', color='lightcoral', linestyle='--')
plt.axhline(y=BASINC_DENIZ_SEVIYESI, color='blue', linestyle=':', label=f'Deniz Seviyesi Basıncı ({BASINC_DENIZ_SEVIYESI} Pa)')
plt.title('Basınç Değişimi', fontsize=14)
plt.xlabel('Zaman (s)', fontsize=10)
plt.ylabel('Basınç (Pa)', fontsize=10)
plt.grid(True)
plt.legend(fontsize=8)

# Pil Gerilimi Grafiği
plt.subplot(2, 2, 2)
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_pil_gerilimi_v'], label='Görev Yükü Pil Gerilimi', color='darkgreen')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['tas_pil_gerilimi_v'], label='Taşıyıcı Pil Gerilimi', color='lightgreen', linestyle='--')
plt.axhline(y=10.2, color='red', linestyle=':', label='Hedef Bitiş Gerilimi (~10.2V)')
plt.title('Pil Gerilimi', fontsize=14)
plt.xlabel('Zaman (s)', fontsize=10)
plt.ylabel('Gerilim (V)', fontsize=10)
plt.grid(True)
plt.legend(fontsize=8)
plt.ylim(bottom=0)


# Eksen Verileri Grafiği (Pitch, Roll, Yaw) - Sadece GY verileri
plt.subplot(2, 2, 3)
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_pitch_derece'], label='GY Pitch', color='blue')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_roll_derece'], label='GY Roll', color='purple')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_yaw_derece'], label='GY Yaw', color='orange')
plt.title('Eksen Verileri (Görev Yükü)', fontsize=14)
plt.xlabel('Zaman (s)', fontsize=10)
plt.ylabel('Açı (derece)', fontsize=10)
plt.grid(True)
plt.legend(fontsize=8)

# Ortam Sıcaklıkları Grafiği - Sadece 3 çizgi ve GY sıcaklığı U şeklinde
plt.subplot(2, 2, 4)
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['iot1_sicaklik_c'], label='IoT 1 Sıcaklık', color='brown')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['iot2_sicaklik_c'], label='IoT 2 Sıcaklık', color='darkorange', linestyle='--')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_sicaklik_c'], label='Görev Yükü Sıcaklığı', color='navy', linestyle=':')
plt.title('Ortam Sıcaklıkları', fontsize=14)
plt.xlabel('Zaman (s)', fontsize=10)
plt.ylabel('Sıcaklık (°C)', fontsize=10)
plt.grid(True)
plt.legend(fontsize=8)


plt.tight_layout()
plt.show()


# Yükseklik Verileri - İrtifa Farkı Analizi
plt.figure(figsize=(12, 7))
if not telemetry_df['irtifa_farki_m'].isnull().all():
    plt.plot(telemetry_df['sim_timestamp'], telemetry_df['irtifa_farki_m'], label='İrtifa Farkı') 
plt.title('Yükseklik Verileri', fontsize=16) 
plt.xlabel('Zaman (s)', fontsize=12)
plt.ylabel('Yükseklik Farkı (m)', fontsize=12) 
plt.grid(True)
plt.legend(fontsize=10)
plt.ylim(bottom=0)
plt.show()

# GPS Verileri Grafiği (Enlem, Boylam, İrtifa) - Sadece Görev Yükü için
plt.figure(figsize=(15, 8))
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_gps_enlem'], label='GY GPS Enlem', color='red', linestyle='-')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_gps_boylam'], label='GY GPS Boylam', color='green', linestyle='--')
plt.plot(telemetry_df['sim_timestamp'], telemetry_df['gy_gps_irtifa_m'], label='GY GPS İrtifa', color='blue', linestyle=':')
plt.title('GPS Verileri (Görev Yükü)', fontsize=16)
plt.xlabel('Zaman (s)', fontsize=12)
plt.ylabel('Değer', fontsize=12)
plt.grid(True)
plt.legend(fontsize=10)
plt.show()


# Görev Yükü ve Taşıyıcı Arasındaki Mesafe Grafiği - KALDIRILDI
# plt.figure(figsize=(12, 7))
# if not telemetry_df['aralarindaki_mesafe_m'].isnull().all():
#     plt.plot(telemetry_df['sim_timestamp'], telemetry_df['aralarindaki_mesafe_m'], label='Aradaki Mesafe', color='purple')
# plt.title('Görev Yükü ve Taşıyıcı Arası Mesafe', fontsize=16)
# plt.xlabel('Zaman (s)', fontsize=12)
# plt.ylabel('Mesafe (m)', fontsize=12)
# plt.grid(True)
# plt.legend(fontsize=10)
# plt.ylim(bottom=0)
# plt.show()


# --- SİMÜLASYON ANALİZ VE SONUÇ ÖZETİ ---
print("\n--- SİMÜLASYON ANALİZ VE SONUÇ ÖZETİ ---")

# Simülasyon sonunda ulaşılan hızlar
final_vel_initial_system = velocities_initial_system_flight[-1] if velocities_initial_system_flight else np.nan
final_vel_tas = velocities_tas_flight[-1] if velocities_tas_flight else np.nan
final_vel_gy = velocities_gy_flight[-1] if velocities_gy_flight else np.nan

# Ortalama hızlar
avg_vel_initial_system = np.mean(velocities_initial_system_flight) if velocities_initial_system_flight else np.nan
avg_vel_tas = np.mean(velocities_tas_flight) if velocities_tas_flight else np.nan
avg_vel_gy = np.mean(velocities_gy_flight) if velocities_gy_flight else np.nan


print("\n--- İniş Hızları Karşılaştırması ---")
print(f"  Başlangıç Sistemi (Yükseliş/İniş Öncesi Ayrılma):")
print(f"    Teorik Hedef Hız: {HEDEF_INIS_HIZI_BASLANGIC:.2f} m/s (Bu fazda farklı bir hedef olabilir)")
if not pd.isna(final_vel_initial_system):
    print(f"    Simülasyon Son Hızı (Faz Sonu): {final_vel_initial_system:.2f} m/s")
    print(f"    Simülasyon Ortalama Hız: {avg_vel_initial_system:.2f} m/s")
    print(f"    Hedef ile Fark (Ortalama Hız): {abs(HEDEF_INIS_HIZI_BASLANGIC - avg_vel_initial_system):.2f} m/s")
else:
    print("    Simülasyon verisi bulunamadı (sistem ayrılmış olabilir veya uçuşu kısa sürmüş).")

print(f"\n  Taşıyıcı (400m sonrası):")
print(f"    Teorik Hedef Hız: {HEDEF_INIS_HIZI_TASIYICI:.2f} m/s")
if not pd.isna(final_vel_tas):
    print(f"    Simülasyon Son Hızı (İniş Öncesi): {final_vel_tas:.2f} m/s")
    print(f"    Simülasyon Ortalama Hız: {avg_vel_tas:.2f} m/s")
    print(f"    Hedef ile Fark (Ortalama HEDEF_INIS_HIZI_TASIYICI): {abs(HEDEF_INIS_HIZI_TASIYICI - avg_vel_tas):.2f} m/s")
else:
    print("    Simülasyon verisi bulunamadı (sistem ayrılmamış olabilir veya yere inmeyebilir).")

print(f"\n  Görev Yükü (400m sonrası):")
print(f"    Teorik Hedef Hız: {HEDEF_INIS_HIZI_GOREV_YUKU:.2f} m/s")
if not pd.isna(final_vel_gy):
    print(f"    Simülasyon Son Hızı (İniş Öncesi): {final_vel_gy:.2f} m/s")
    print(f"    Simülasyon Ortalama Hız: {avg_vel_gy:.2f} m/s")
    print(f"    Hedef ile Fark (Ortalama HEDEF_INIS_HIZI_GOREV_YUKU): {abs(HEDEF_INIS_HIZI_GOREV_YUKU - avg_vel_gy):.2f} m/s") # Düzeltilen kısım
else:
    print("    Simülasyon verisi bulunamadı (sistem ayrılmamış olabilir veya yere inmeyebilir).")


print("\n--- Genel Telemetri Analizi ---")
print(f"Toplam telemetri paketi sayısı: {len(sim_telemetry_data)}")
if len(sim_telemetry_data) > 0:
    telemetry_frequency = len(sim_telemetry_data) / sim_telemetry_data[-1]['sim_timestamp'] if sim_telemetry_data[-1]['sim_timestamp'] > 0 else 0
    print(f"Ortalama telemetri gönderim sıklığı: {telemetry_frequency:.2f} paket/s (Beklenen: {1/TELEMETRI_GONDERIM_SIKLIGI:.1f} paket/s)")


print("\n--- Görev Yükü Aktif/Pasif Durum Analizi ---")
gy_aktif_timestamps = telemetry_df[telemetry_df['gy_gorev_yuku_durum'] == 1]['sim_timestamp'].min()
if not pd.isna(gy_aktif_timestamps):
    print(f"  Görev Yükü ilk aktifleştiği zaman: {gy_aktif_timestamps:.1f} s")
else:
    print("  Görev Yükü simülasyon boyunca hiç aktifleşmedi (bu bir sorun olabilir!).")


print("\n--- Ayrılma Durum Analizi ---")
ayrilma_timestamps = telemetry_df[telemetry_df['ayrilma_durum'] == 1]['sim_timestamp'].min()
if not pd.isna(ayrilma_timestamps):
    print(f"  Ayrılma işlemi gerçekleştiği zaman: {ayrilma_timestamps:.1f} s")
else:
    print("  Ayrılma işlemi simülasyon boyunca gerçekleşmedi (bu bir sorun olabilir!).")


print("\n--- Konum ve Mesafe Analizi ---")
if ayrilma_gerceklesti and len(telemetry_df) > 0:
    final_gy_dist = telemetry_df['gy_firlatma_mesafe_m'].iloc[-1]
    final_tas_dist = telemetry_df['tas_firlatma_mesafe_m'].iloc[-1]
    # final_distance_between = telemetry_df['aralarindaki_mesafe_m'].iloc[-1] # Kaldırıldı

    print(f"  Görev Yükünün Fırlatma Yerine Son Mesafesi: {final_gy_dist:.2f} m")
    print(f"  Taşıyıcının Fırlatma Yerine Son Mesafesi: {final_tas_dist:.2f} m")
    # print(f"  Görev Yükü ve Taşıyıcı Arasındaki Son Mesafe: {final_distance_between:.2f} m") # Kaldırıldı
else:
    print("  Konum ve mesafe analizi için yeterli veri bulunamadı (ayrılma gerçekleşmemiş veya sistem yere inmemiş olabilir).")
