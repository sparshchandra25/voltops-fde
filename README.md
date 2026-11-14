
# VOLTOPS // 3D BMS Diagnostic & Statutory Hybrid RAG Workstation

**Live Deployment:** [https://voltops-fde.vercel.app](https://voltops-fde.vercel.app)

An end-to-end Forward Deployed Engineering (FDE) telemetry and diagnostics pipeline for Electric Vehicle High-Voltage (HV) Battery Management Systems (BMS). VoltOps ingests physical lithium-ion aging logs from the **NASA Ames Prognostics Center of Excellence (PCoE)**, couples them with live ambient weather telemetry across **12 Pan-India EV operating hubs**, runs an **$O(1)$ amortized Monotonic Deque sliding-window fault detector**, predicts **Remaining Useful Life (RUL)** via Gradient Boosted Decision Trees, and synthesizes statutory repair work orders grounded in **India MoRTH (AIS-156 / AIS-038)**, **UN GTR No. 20**, and **US DOT NHTSA** recall bulletins.

---

## System Architecture (5-Stage FDE Pipeline)

```text
[NASA PCoE .mat/.csv Telemetry] + [Open-Meteo Live Pan-India Weather API]
                                  │
                                  ▼
      ┌───────────────────────────────────────────────────────┐
      │ STAGE 01: Real-Time Sensor & Thermal Offset Ingestion │
      └───────────────────────────────────────────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│ STAGE 02: O(1) Monotonic Deque   │  │ STAGE 03: Gradient Boosted RUL   │
│ 25.0s Sliding-Window dV/dt Guard │  │ & Ohmic Internal Resistance (IR) │
│ -> Trips OBD-II P0B24 / P0A7E    │  │ -> Predicts SoH % & Cycle RUL    │
└──────────────────────────────────┘  └──────────────────────────────────┘
                 └────────────────┬────────────────┘
                                  ▼
      ┌───────────────────────────────────────────────────────┐
      │ STAGE 04: Category-Aware Hybrid RAG Engine            │
      │ BM25 Lexical + TruncatedSVD Dense Vectors + RRF       │
      │ Corpus: MoRTH AIS-156, AIS-038 Rev 2, UN GTR-20, NHTSA│
      └───────────────────────────────────────────────────────┘
                                  │
                                  ▼
      ┌───────────────────────────────────────────────────────┐
      │ STAGE 05: Agentic Work Order & HV Safety Evals Gate   │
      │ Verifies 1000V Class-0 PPE, MSD Pull & <30V DC Bleed  │
      └───────────────────────────────────────────────────────┘
