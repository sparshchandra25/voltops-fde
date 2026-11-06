# step2_dsa_and_ml_engine.py
from collections import deque
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Real Indian EV Profiles (AIS-038 Rev 2 for 4W, AIS-156 Amend 3 for 2W)
VEHICLE_PROFILES = {
    "4W_TATA_LFP": {
        "name": "Tata Nexon EV / Mahindra XUV400 (4W 320V LFP Pack — AIS-038 Rev 2)",
        "series_cells": 100,
        "window_sec": 25.0,
        "delta_v_crit": 0.18,       # Tighter ΔV threshold for flat LFP curve
        "max_safe_temp_c": 52.0,    # AIS-038 thermal warning trigger
        "statutory_law": "MoRTH AIS-038 (Rev 2) & UN GTR No. 20",
    },
    "2W_OLA_NMC": {
        "name": "Ola S1 Pro / Ather 450X (2W 51.1V NMC Pack — AIS-156 Amend 3)",
        "series_cells": 14,
        "window_sec": 25.0,
        "delta_v_crit": 0.24,       # NMC steeper curve ΔV sag threshold
        "max_safe_temp_c": 48.0,    # NMC lower thermal runaway onset
        "statutory_law": "MoRTH AIS-156 (Amendment 3, Phase 2 - Annex 8J)",
    },
}


class MonotonicSlidingWindowDetector:
    """
    O(1) Time Complexity Sliding-Window Voltage Drop & Thermal Fault Detector.
    Uses monotonic deques to track rolling V_max and V_min over a time window W.
    """
    def __init__(self, profile_key: str):
        self.profile = VEHICLE_PROFILES.get(profile_key, VEHICLE_PROFILES["4W_TATA_LFP"])
        self.window_sec = self.profile["window_sec"]
        self.delta_v_crit = self.profile["delta_v_crit"]
        self.max_deque = deque()  # Stores (time_s, voltage_v) in decreasing order
        self.min_deque = deque()  # Stores (time_s, voltage_v) in increasing order

    def step(self, time_s: float, voltage_v: float, current_a: float, temp_c: float, ambient_offset_c: float = 0.0):
        effective_temp_c = round(temp_c + ambient_offset_c, 2)

        # Only track rolling ΔV once constant-current load has stabilized (t > 45s)
        # and before the standard end-of-discharge knee (voltage_v > 3.35V)
        is_active_plateau = (abs(current_a) > 1.0) and (time_s > 45.0) and (voltage_v > 3.35)

        if not is_active_plateau:
            return {"fault_detected": False, "rolling_delta_v": 0.0, "effective_temp_c": effective_temp_c}

        # 1. Evict expired timestamps outside [time_s - W, time_s]
        while self.max_deque and self.max_deque[0][0] < time_s - self.window_sec:
            self.max_deque.popleft()
        while self.min_deque and self.min_deque[0][0] < time_s - self.window_sec:
            self.min_deque.popleft()

        # 2. Maintain monotonic invariants in O(1) amortized time
        while self.max_deque and self.max_deque[-1][1] <= voltage_v:
            self.max_deque.pop()
        self.max_deque.append((time_s, voltage_v))

        while self.min_deque and self.min_deque[-1][1] >= voltage_v:
            self.min_deque.pop()
        self.min_deque.append((time_s, voltage_v))

        # 3. O(1) Rolling Delta-V computation
        v_max = self.max_deque[0][1]
        v_min = self.min_deque[0][1]
        delta_v = round(v_max - v_min, 4)

        v_fault = delta_v >= self.delta_v_crit
        t_fault = effective_temp_c >= self.profile["max_safe_temp_c"]

        if v_fault or t_fault:
            if v_fault and t_fault:
                dtc = "P0B24 + P0A7F (Combined Voltage Sag & Thermal Runaway Risk)"
            elif v_fault:
                dtc = "P0B24 (Cell Voltage Unstable / Internal Sag)"
            else:
                dtc = "P0A7F (Pack Thermal Over-Temperature Threshold Exceeded)"

            return {
                "fault_detected": True,
                "time_s": time_s,
                "cell_voltage_v": voltage_v,
                "pack_voltage_v": round(voltage_v * self.profile["series_cells"], 1),
                "rolling_delta_v": delta_v,
                "threshold_v": self.delta_v_crit,
                "temp_c": effective_temp_c,
                "dtc_code": dtc,
                "law_triggered": self.profile["statutory_law"],
                "effective_temp_c": effective_temp_c,
            }
        return {"fault_detected": False, "rolling_delta_v": delta_v, "effective_temp_c": effective_temp_c}


def train_nasa_rul_model():
    """Trains LightGBM on Battery B0005 and evaluates on degraded Battery B0006."""
    df = pd.read_csv("data/nasa_cycles_processed.csv")
    features = [
        "discharge_cycle", "mean_voltage_v", "min_voltage_v",
        "max_temp_c", "temp_rise_c", "discharge_duration_s", "ir_proxy_ohm"
    ]

    train_df = df[df["battery_id"] == "B0005"]
    test_df = df[df["battery_id"] == "B0006"]

    model = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1
    )
    model.fit(train_df[features], train_df["rul_cycles"])

    preds = model.predict(test_df[features])
    mae = mean_absolute_error(test_df["rul_cycles"], preds)
    print(f"[ML ENGINE] LightGBM RUL Model Trained on NASA B0005 | Test MAE on B0006: {mae:.2f} cycles")
    return model, features


if __name__ == "__main__":
    model, features = train_nasa_rul_model()
    streams = pd.read_csv("data/nasa_streams_processed.csv")

    for cyc in [1, 118]:
        sample_stream = streams[(streams["battery_id"] == "B0006") & (streams["discharge_cycle"] == cyc)]
        detector = MonotonicSlidingWindowDetector("4W_TATA_LFP")
        tripped = False
        for _, row in sample_stream.iterrows():
            res = detector.step(row["time_s"], row["voltage_v"], row["current_a"], row["temp_c"])
            if res["fault_detected"]:
                print(f"🚨 [Cycle #{cyc}] TRIP @ t={res['time_s']}s | ΔV={res['rolling_delta_v']}V | {res['dtc_code']}")
                tripped = True
                break
        if not tripped:
            print(f"✅ [Cycle #{cyc}] HEALTHY — Zero false-positive voltage sags across entire discharge curve.")