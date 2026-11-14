from collections import deque
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

VEHICLE_PROFILES = {
    "4W_TATA_LFP": {
        "name": "Tata Nexon EV (4W LFP Pack)",
        "series_cells": 100,
        "nominal_v": 320.0,
        "window_sec": 25.0,
        "delta_v_crit": 0.18,
        "temp_crit_c": 52.0,
        "statutory_law": "MoRTH AIS-038 Rev 2 (Clause 5.1.3)",
    },
    "2W_OLA_NMC": {
        "name": "Ola S1 Pro / Ather 450X (2W NMC Pack)",
        "series_cells": 14,
        "nominal_v": 51.1,
        "window_sec": 25.0,
        "delta_v_crit": 0.22,
        "temp_crit_c": 55.0,
        "statutory_law": "MoRTH AIS-156 Amend 3 Phase 2 (Annex 8J)",
    },
}


class MonotonicSlidingWindowDetector:
    """
    O(1) amortized monotonic deque sliding-window detector for rapid cell voltage sag (dV/dt).
    """
    def __init__(self, vehicle_mode: str = "4W_TATA_LFP"):
        self.vehicle_mode = vehicle_mode if vehicle_mode in VEHICLE_PROFILES else "4W_TATA_LFP"
        self.profile = VEHICLE_PROFILES[self.vehicle_mode]
        self.window_sec = self.profile["window_sec"]
        self.max_deque = deque()
        self.min_deque = deque()

    def step(self, time_s: float, voltage_v: float, current_a: float, temp_c: float, ambient_offset_c: float = 0.0):
        cutoff_t = time_s - self.window_sec

        while self.max_deque and self.max_deque[0][0] < cutoff_t:
            self.max_deque.popleft()
        while self.min_deque and self.min_deque[0][0] < cutoff_t:
            self.min_deque.popleft()

        while self.max_deque and self.max_deque[-1][1] <= voltage_v:
            self.max_deque.pop()
        self.max_deque.append((time_s, voltage_v))

        while self.min_deque and self.min_deque[-1][1] >= voltage_v:
            self.min_deque.pop()
        self.min_deque.append((time_s, voltage_v))

        rolling_max_v = self.max_deque[0][1]
        rolling_delta_v = round(max(0.0, rolling_max_v - voltage_v), 4)
        effective_temp_c = round(temp_c + ambient_offset_c, 2)

        under_active_load = (current_a < -0.5) and (30.0 <= time_s <= 2350.0) and (voltage_v > 2.95)
        sag_fault = under_active_load and (rolling_delta_v > self.profile["delta_v_crit"])
        temp_fault = effective_temp_c >= self.profile["temp_crit_c"]
        fault_detected = bool(sag_fault or temp_fault)

        dtc_code = None
        if sag_fault:
            dtc_code = "P0B24 (Hybrid/EV Battery 'A' Voltage Unstable)"
        elif temp_fault:
            dtc_code = "P0A7E (Hybrid/EV Battery Pack Over-Temperature)"

        return {
            "time_s": round(time_s, 2),
            "cell_voltage_v": round(voltage_v, 4),
            "current_a": round(current_a, 3),
            "effective_temp_c": effective_temp_c,
            "rolling_delta_v": rolling_delta_v,
            "threshold_v": self.profile["delta_v_crit"],
            "fault_detected": fault_detected,
            "dtc_code": dtc_code,
            "law_triggered": self.profile["statutory_law"] if fault_detected else None,
        }


def train_nasa_rul_model(cycles_csv_path: str = "data/nasa_cycles_processed.csv"):
    df = pd.read_csv(cycles_csv_path)
    features = [
        "discharge_cycle",
        "capacity_ah",
        "soh_pct",
        "min_voltage_v",
        "mean_voltage_v",
        "max_temp_c",
        "temp_rise_c",
        "ir_proxy_ohm",
        "discharge_duration_s",
    ]
    train_df = df[df["battery_id"] == "B0005"].copy()
    if train_df.empty:
        train_df = df.copy()

    model = GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.07,
        max_depth=4,
        random_state=42,
    )
    model.fit(train_df[features], train_df["rul_cycles"])
    return model, features
