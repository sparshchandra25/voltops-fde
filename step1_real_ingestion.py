# step1_real_ingestion.py
import os
import urllib.request
import requests
import scipy.io
import numpy as np
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Verified raw mirrors for NASA Ames PCoE Battery Dataset
NASA_MIRRORS = {
    "B0005.mat": [
        "https://raw.githubusercontent.com/changyeon99/Battery-Data-Set/main/B0005.mat",
        "https://raw.githubusercontent.com/Maruthi-prasanth/Battery-Health-Prophecy/main/B0005.mat",
        "https://raw.githubusercontent.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset/master/B0005.mat",
    ],
    "B0006.mat": [
        "https://raw.githubusercontent.com/changyeon99/Battery-Data-Set/main/B0006.mat",
        "https://raw.githubusercontent.com/Maruthi-prasanth/Battery-Health-Prophecy/main/B0006.mat",
        "https://raw.githubusercontent.com/anirudhkhatry/SOH-prediction-using-NASA-Dataset/master/B0006.mat",
    ],
}

# Pan-India EV Manufacturing, Fleet & Extreme Climate Testing Hubs
INDIAN_EV_HUBS = {
    "Chennai (SRM KTR / TN EV Corridor)": {"lat": 12.8230, "lon": 80.0444, "state": "Tamil Nadu", "zone": "Coastal Humid"},
    "Bengaluru (Ather / Ola Futurefactory Hub)": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "zone": "Deccan Plateau"},
    "Pune (Tata Motors / Chakan Auto Cluster)": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "zone": "Western Ghats"},
    "Delhi NCR (Gurugram High-Heat EV Fleets)": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi NCR", "zone": "Extreme North Summer"},
    "Mumbai (BKC / Coastal Monsoon Fleets)": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "zone": "High-Rainfall Coastal"},
    "Hyderabad (Telangana Mobility Valley)": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "zone": "Semi-Arid Central"},
    "Ahmedabad (Sanand / Gujarat EV Hub)": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "zone": "Dry Arid Heat"},
    "Kolkata (East India Last-Mile EV Hub)": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "zone": "Tropical Wet"},
    "Jaipur (Rajasthan Desert Thermal Testing)": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "zone": "Arid Desert Heat"},
    "Kochi (Kerala High-Humidity EV Grid)": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala", "zone": "Equatorial Monsoon"},
    "Guwahati (Northeast Hilly Terrain Hub)": {"lat": 26.1445, "lon": 91.7362, "state": "Assam", "zone": "Sub-Himalayan Grade"},
    "Leh - Ladakh (High-Altitude Cold BMS Test)": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh", "zone": "Sub-Zero High Altitude"},
}


def download_nasa_mat_files():
    """Downloads real NASA binary .mat files if not already cached locally."""
    for filename, urls in NASA_MIRRORS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 100_000:
            print(f"[OK] {filename} already exists ({os.path.getsize(filepath) / 1e6:.2f} MB)")
            continue

        for url in urls:
            try:
                print(f"[DOWNLOADING] {filename} from {url} ...")
                urllib.request.urlretrieve(url, filepath)
                if os.path.getsize(filepath) > 100_000:
                    print(f"[SUCCESS] Saved {filepath} ({os.path.getsize(filepath) / 1e6:.2f} MB)")
                    break
            except Exception as e:
                print(f"[WARN] Mirror failed ({e}), trying next...")


def parse_nasa_mat(battery_id: str):
    """
    Unpacks NASA MATLAB structs into:
    1. cycle_df: 1 row per discharge cycle (for LightGBM RUL & Thermal ML training)
    2. stream_df: high-frequency second-by-second V, I, T curves (for DSA Sliding Window)
    """
    filepath = os.path.join(DATA_DIR, f"{battery_id}.mat")
    mat = scipy.io.loadmat(filepath)
    raw_cycles = mat[battery_id][0, 0]["cycle"][0]

    cycle_summaries = []
    timeseries_rows = []
    discharge_idx = 0

    for cycle in raw_cycles:
        cycle_type = str(cycle["type"][0])
        if cycle_type != "discharge":
            continue

        discharge_idx += 1
        ambient_temp = float(cycle["ambient_temperature"][0, 0])
        data = cycle["data"][0, 0]

        v_meas = data["Voltage_measured"][0]
        i_meas = data["Current_measured"][0]
        t_meas = data["Temperature_measured"][0]
        v_load = data["Voltage_load"][0]
        time_s = data["Time"][0]
        capacity_ah = float(data["Capacity"][0, 0]) if data["Capacity"].size > 0 else np.nan

        duration_s = float(time_s[-1] - time_s[0])
        temp_rise_c = float(np.max(t_meas) - t_meas[0])
        mean_v = float(np.mean(v_meas))
        min_v = float(np.min(v_meas))
        ir_proxy = float(abs(v_meas[0] - v_meas[min(5, len(v_meas) - 1)]) / (abs(np.mean(i_meas)) + 1e-5))

        # Clamp SoH at 100.0% max for fresh cells exceeding nominal 2.0 Ah nameplate
        raw_soh = (capacity_ah / 2.0) * 100.0
        clamped_soh = round(min(100.0, max(0.0, raw_soh)), 2)

        cycle_summaries.append({
            "battery_id": battery_id,
            "discharge_cycle": discharge_idx,
            "ambient_temp_c": ambient_temp,
            "capacity_ah": capacity_ah,
            "soh_pct": clamped_soh,
            "mean_voltage_v": round(mean_v, 4),
            "min_voltage_v": round(min_v, 4),
            "max_temp_c": round(float(np.max(t_meas)), 2),
            "temp_rise_c": round(temp_rise_c, 2),
            "discharge_duration_s": round(duration_s, 1),
            "ir_proxy_ohm": round(ir_proxy, 4),
        })

        if discharge_idx in (1, 10, 50, 100, 118, 150, 168):
            for t, v, i, temp, vl in zip(time_s, v_meas, i_meas, t_meas, v_load):
                timeseries_rows.append({
                    "battery_id": battery_id,
                    "discharge_cycle": discharge_idx,
                    "time_s": round(float(t), 2),
                    "voltage_v": round(float(v), 4),
                    "current_a": round(float(i), 4),
                    "temp_c": round(float(temp), 3),
                    "voltage_load_v": round(float(vl), 4),
                    "capacity_ah": round(capacity_ah, 4),
                })

    cycle_df = pd.DataFrame(cycle_summaries)
    eol_cycles = cycle_df[cycle_df["capacity_ah"] <= 1.40]["discharge_cycle"]
    eol_cycle_num = int(eol_cycles.iloc[0]) if len(eol_cycles) > 0 else int(cycle_df["discharge_cycle"].max())
    cycle_df["rul_cycles"] = np.maximum(0, eol_cycle_num - cycle_df["discharge_cycle"])

    return cycle_df, pd.DataFrame(timeseries_rows)


def fetch_live_indian_ambient_temp(city_key: str = "Chennai (SRM KTR / TN EV Corridor)"):
    """
    Fetches live ambient temperature & humidity for any preset Indian EV hub
    OR dynamically geocodes any custom Indian city name via Open-Meteo API.
    """
    if city_key in INDIAN_EV_HUBS:
        hub = INDIAN_EV_HUBS[city_key]
        lat, lon, state, zone = hub["lat"], hub["lon"], hub["state"], hub["zone"]
    else:
        # Dynamic geocoding for any custom Indian city typed by user
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city_key)}&count=1&language=en&format=json"
            geo_res = requests.get(geo_url, timeout=4).json()
            if geo_res.get("results"):
                loc = geo_res["results"][0]
                lat, lon = loc["latitude"], loc["longitude"]
                state = loc.get("admin1", "India")
                zone = "Custom Live Geocode"
            else:
                hub = INDIAN_EV_HUBS["Chennai (SRM KTR / TN EV Corridor)"]
                lat, lon, state, zone = hub["lat"], hub["lon"], hub["state"], hub["zone"]
        except Exception:
            hub = INDIAN_EV_HUBS["Chennai (SRM KTR / TN EV Corridor)"]
            lat, lon, state, zone = hub["lat"], hub["lon"], hub["state"], hub["zone"]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
    try:
        resp = requests.get(url, timeout=5).json()
        temp = float(resp["current"]["temperature_2m"])
        hum = int(resp["current"]["relative_humidity_2m"])
        print(f"[LIVE WEATHER] {city_key} ({state} - {zone}): {temp}°C | Humidity: {hum}%")
        return {
            "city": city_key,
            "state": state,
            "zone": zone,
            "temp_c": temp,
            "humidity_pct": hum,
            "lat": lat,
            "lon": lon,
        }
    except Exception as e:
        print(f"[WARN] Weather API fallback triggered for {city_key}: {e}")
        return {
            "city": city_key,
            "state": state,
            "zone": zone,
            "temp_c": 32.0,
            "humidity_pct": 65,
            "lat": lat,
            "lon": lon,
        }


def fetch_live_nhtsa_ev_recalls(make="CHEVROLET", model="BOLT EV", year="2021"):
    """Fetches real government High-Voltage Battery recall bulletins from US DOT NHTSA API."""
    url = f"https://api.nhtsa.gov/recalls/recallsByVehicle?make={make}&model={model}&modelYear={year}"
    try:
        resp = requests.get(url, timeout=8).json()
        results = resp.get("results", [])
        print(f"[LIVE NHTSA API] Fetched {len(results)} official recalls for {year} {make} {model}")
        return results
    except Exception as e:
        print(f"[WARN] NHTSA API error: {e}")
        return []


if __name__ == "__main__":
    download_nasa_mat_files()

    all_cycles, all_streams = [], []
    for bat in ["B0005", "B0006"]:
        c_df, s_df = parse_nasa_mat(bat)
        all_cycles.append(c_df)
        all_streams.append(s_df)

    full_cycle_df = pd.concat(all_cycles, ignore_index=True)
    full_stream_df = pd.concat(all_streams, ignore_index=True)

    full_cycle_df.to_csv(os.path.join(DATA_DIR, "nasa_cycles_processed.csv"), index=False)
    full_stream_df.to_csv(os.path.join(DATA_DIR, "nasa_streams_processed.csv"), index=False)

    print(f"\n[SAVED] {len(full_cycle_df)} real discharge cycles -> data/nasa_cycles_processed.csv")
    print(f"[SAVED] {len(full_stream_df)} real high-freq telemetry rows -> data/nasa_streams_processed.csv\n")

    for test_hub in ["Chennai (SRM KTR / TN EV Corridor)", "Delhi NCR (Gurugram High-Heat EV Fleets)", "Leh - Ladakh (High-Altitude Cold BMS Test)"]:
        fetch_live_indian_ambient_temp(test_hub)
    fetch_live_nhtsa_ev_recalls("CHEVROLET", "BOLT EV", "2021")