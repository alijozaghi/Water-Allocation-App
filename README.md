# 💧 Water Allocation + Desalination Web App

A **Streamlit-based decision-support application** for projecting and optimizing water treatment plant (WTP) demands under multiple constraints — including **Bois d’Arc annual average limits**, **desalination efficiency**, and **capacity-based allocations**. The app enables engineers and planners to evaluate different water-supply and desalination strategies in a transparent, interactive, and data-driven way.

---

## 🚀 Key Features

### 🔹 Core Functionality
- **Daily Water Demand Simulation**
  Computes plant-level (Wylie, Leonard, Tawakoni) daily flows using user-defined ratios, SCADA data, and pipeline capacities.

- **Interval-Aware Bois d’Arc Cap Enforcement**
  Automatically adjusts Leonard and Wylie flows within a selected interval to maintain compliance with the annual Bois d’Arc average limit (MGD).

- **Desalination Integration**
  Simulates desalination processes with customizable efficiency (fraction or decimal) and “offset” policies (prioritizing Wylie or Tawakoni reductions first).

- **Optimization Engine**
  Solves for the optimal distribution ratios (Wylie/Leonard/Tawakoni) to **maximize Texoma usage** while maintaining plant capacity, average flow floors, and operational constraints.

### 🔹 Interactive Interface
- Clean, responsive **Streamlit layout** with collapsible parameter panels.
- **Sidebar configuration** for all model inputs, including capacities, mixing ratios, interval controls, and desalination options.
- **Selectable ratio modes**:
  - Manual (user-defined)
  - Capacity-based (proportional to plant capacity)
  - Optimized (computed via constrained optimization)

### 🔹 Outputs & Visualization
- Real-time **KPI metrics** for each WTP (annual averages, peaks, and totals).
- **Daily time-series plots** for all plants and flows (Texoma, Lavon, Bois d’Arc, Tawakoni, Desal).
- Interactive data table and **CSV export** for results.

---

## 🧩 Sidebar Parameters

| Category | Description |
|-----------|-------------|
| **Plant Capacities (MGD)** | Wylie, Leonard, and Tawakoni design capacities (moved here from main UI). |
| **Mix Ratios** | Lavon:Texoma and Bois d’Arc:Texoma blending ratios. |
| **Pipe Capacities** | Defines transmission limits (Texoma→Wylie, Texoma→Leonard). |
| **Bois d’Arc Limit** | Annual average limit (MGD) for Bois d’Arc withdrawals. |
| **Interval Controls** | Defines time window and direction for Leonard→Wylie rebalancing. |
| **Desalination Settings** | Policy, efficiency, and per-facility floor flows. |
| **Optimization Floors** | Minimum average MGD thresholds for each plant during optimization. |

---

## ⚙️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/water-allocation-desal.git
cd water-allocation-desal

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

---

## 📁 Input Data

Upload a **SCADA CSV file** containing daily total system demand data.
Each column should represent a single year, with numeric daily values (MGD):

```text
| DATE | 2018 | 2019 | 2020 | 2021 |
|------|------|------|------|------|
| 1/1  | 785  | 812  | 830  | 840  |
| 1/2  | 790  | 820  | 835  | 845  |
```

The app automatically validates the selected year column for numeric completeness.

---

## 📊 Results & Outputs

- **KPI Dashboard:** Annual averages, peak-day demands, and Texoma utilization summary.
- **Interactive Plots:** Multi-plant time series with optional overlay lines (targets or averages).
- **Results Table:** Daily detailed outputs (MGD) for all sources and facilities.
- **Export:** Download all computed data as a CSV file.

---

## 🧠 Optimization Overview

The optimization routine uses **SciPy’s SLSQP solver** to find the ratio vector **(rW, rL, rT)** that:
- Maximizes total water imported from Texoma,
- Respects individual WTP capacities and average MGD floors,
- Satisfies the Bois d’Arc constraint across the specified interval.

Constraint types include:
- Equality: (rW + rL + rT = 1)
- Inequalities: (r_i ≥ 0), (r_i ≤ r_max,i), and (avg flow_i ≥ floor_i)

---

## 🧾 Example Use Cases

- Evaluate desalination efficiency impacts (e.g., 0.85).
- Test capacity expansions or operational limits.
- Simulate seasonal water rebalancing between Leonard and Wylie.
- Optimize daily blending ratios to maximize Texoma intake.

---

## 🧩 Dependencies

- Python ≥ 3.8
- streamlit
- pandas, numpy, matplotlib
- scipy
- datetime

All dependencies are listed in the provided `requirements.txt`.

---

## 📜 License

This project is distributed under the **MIT License**.
Feel free to use, modify, and distribute with proper attribution.

---

