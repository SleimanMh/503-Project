"""
EV Charging Optimization Dashboard — Streamlit
3-way comparison: AI+ML vs AI-only vs FCFS, with ML monitoring panel.
"""

import json
import random
from datetime import datetime, timedelta, timezone

import time

import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as _components

GATEWAY_URL = "http://localhost:8000"

st.set_page_config(page_title="EV Charging System", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    _view = st.radio(
        "nav",
        [
            "📊 Optimizer Simulation",
            "🔌 OCPP Live Monitor",
            "📈 MLflow Tracking",
            "📡 Grafana Monitoring",
            "🔬 Prometheus Metrics",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.header("System Status")
    try:
        health = requests.get(f"{GATEWAY_URL}/health", timeout=3).json()
        st.success("Gateway: UP")
        for svc, status in health.get("services", {}).items():
            if status == "up":
                st.success(f"{svc}: UP")
            else:
                st.warning(f"{svc}: DOWN")
    except Exception:
        st.error("Gateway: UNREACHABLE")

    st.divider()
    st.header("Grid Parameters")
    transformer_kw = st.number_input("Transformer Capacity (kW)", 50, 2000, 150)
    building_load_kw = st.number_input("Building Base Load (kW)", 10, 1000, 60)
    duration_hours = st.slider("Simulation Duration (hours)", 1, 48, 8)
    time_step = st.selectbox("Time Step (min)", [15, 30, 60], index=0)
    compare_baseline = st.checkbox("Compare with FCFS Baseline", value=True)

    st.divider()
    st.header("Grid Layout")
    ev_chargers_a = st.slider("Section A — chargers", 1, 30, 10)
    ev_chargers_b = st.slider("Section B — chargers", 1, 30, 10)
    feeder_length_m = st.slider("Feeder length (m)", 10, 200, 30, step=5)

    st.divider()
    st.header("Electric Vehicles")
    num_evs = st.slider("Number of EVs", 1, 50, 10)

    st.divider()
    st.header("Arrival Pattern")
    arrival_mode = st.selectbox(
        "Arrival timing",
        ["All arrive together", "Staggered (random)", "Staggered (evenly spaced)"],
        index=1,
        help="Controls when each EV arrives at the charging hub",
    )
    if arrival_mode != "All arrive together":
        spread_hours = st.slider(
            "Arrival spread (hours)", 0.5, 8.0, 3.0, step=0.5,
            help="Total time window over which EVs arrive",
        )
    else:
        spread_hours = 0.0

# Realistic EV profiles used for randomisation
_EV_PROFILES = [
    {"name": "Nissan Leaf",       "capacity": 40,  "max_kw": 7.2},
    {"name": "Renault Zoe",       "capacity": 50,  "max_kw": 11.0},
    {"name": "Tesla Model 3 SR",  "capacity": 60,  "max_kw": 11.0},
    {"name": "Tesla Model 3 LR",  "capacity": 75,  "max_kw": 11.0},
    {"name": "BMW i3",            "capacity": 40,  "max_kw": 7.2},
    {"name": "VW ID.3",           "capacity": 58,  "max_kw": 11.0},
    {"name": "VW ID.4",           "capacity": 77,  "max_kw": 11.0},
    {"name": "Hyundai Ioniq 5",   "capacity": 72,  "max_kw": 22.0},
    {"name": "Kia EV6",           "capacity": 77,  "max_kw": 22.0},
    {"name": "Tesla Model Y",     "capacity": 75,  "max_kw": 11.0},
    {"name": "Audi e-tron",       "capacity": 95,  "max_kw": 22.0},
    {"name": "Peugeot e-208",     "capacity": 50,  "max_kw": 7.2},
    {"name": "Skoda Enyaq",       "capacity": 77,  "max_kw": 11.0},
    {"name": "Mini Electric",     "capacity": 33,  "max_kw": 7.2},
    {"name": "Mercedes EQC",      "capacity": 80,  "max_kw": 11.0},
]

def _random_ev_defaults(idx):
    """Return realistic random values for one EV slot."""
    rng = random.Random(st.session_state.get("rand_seed", 0) + idx)
    profile = rng.choice(_EV_PROFILES)
    soc = rng.randint(8, 55)          # typical arrival SoC: low–medium
    target = rng.randint(80, 100)     # drivers want near-full charge
    return soc, target, profile["max_kw"], profile["capacity"], profile["name"]


def _topology_figure(params: dict, live_gv: dict | None = None) -> go.Figure:
    """
    Interactive single-line diagram of the 5-bus EV charging distribution network.
    Draws a proper electrical-style layout: HV source at top, transformer in the
    middle, LV busbar as a thick horizontal bar, with feeders dropping down to
    EV Sections and the Building.  Includes visual icons (⚡🔌🏢) and clear
    voltage / loading annotations.
    """
    # ── Colour palette ──
    IDLE = "#b0c4de"   # steel-blue — no simulation data yet
    OK   = "#27ae60"   # green — nominal
    WARN = "#f39c12"   # amber — warning
    CRIT = "#e74c3c"   # red — critical
    BG   = "#fafbfc"
    WIRE = "#555"

    bv       = (live_gv or {}).get("bus_voltages_pu", {})
    t_pct    = (live_gv or {}).get("max_transformer_loading_pct")
    losses   = (live_gv or {}).get("network_losses_kw")
    feasible = (live_gv or {}).get("feasible")
    kva      = params["transformer_kw"] / 0.95

    def _vc(key):
        if key not in bv:
            return IDLE
        v = bv[key]
        return OK if v >= 0.95 else WARN if v >= 0.90 else CRIT

    def _tc():
        if t_pct is None:
            return IDLE
        return OK if t_pct < 80 else WARN if t_pct < 100 else CRIT

    fig = go.Figure()

    # ── Layout coordinates ──
    # Y: 10 (top) → 0 (bottom).   X centred on 5, range 0–10.
    HV_Y    = 9.0
    XFMR_Y  = 7.0
    BUS_Y   = 5.0
    LOAD_Y  = 2.0
    CX      = 5.0     # centre x
    SEC_A_X = 1.8
    BLDG_X  = 5.0
    SEC_B_X = 8.2

    # ── Helper: draw a wire (line segment) ──
    def wire(x0, y0, x1, y1, color=WIRE, width=2, dash="solid", label=None):
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hoverinfo="skip", showlegend=False,
        ))
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_annotation(
                x=mx + (0.25 if x0 == x1 else 0), y=my,
                text=label, showarrow=False,
                font=dict(size=8, color="#777"),
                bgcolor="rgba(255,255,255,0.85)", borderpad=2,
            )

    # ── Helper: draw a node (marker + label) ──
    def node(x, y, label, sublabel, color, symbol="circle", size=36,
             textpos="top center", hover_lines=None):
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(color=color, size=size, symbol=symbol,
                        line=dict(color="white", width=2.5)),
            text=[f"<b>{label}</b><br><span style='font-size:9px'>{sublabel}</span>"],
            textposition=textpos,
            textfont=dict(size=10),
            hovertext="<br>".join(hover_lines or [label]),
            hoverinfo="text", showlegend=False,
        ))

    # ════════════════════════════════════════════════════════
    #  1) HV Grid source
    # ════════════════════════════════════════════════════════
    hv_color = _vc("HV Bus")
    hv_v = f" • {bv['HV Bus']:.4f} p.u." if "HV Bus" in bv else ""
    node(CX, HV_Y, "⚡ HV Grid", f"4.16 kV{hv_v}", hv_color,
         symbol="diamond", size=40, textpos="top center",
         hover_lines=["<b>HV Grid Bus</b>", "Nominal: 4.16 kV",
                       f"Voltage: {bv.get('HV Bus', '—')} p.u."])

    # ════════════════════════════════════════════════════════
    #  2) HV → Transformer wire
    # ════════════════════════════════════════════════════════
    wire(CX, HV_Y - 0.45, CX, XFMR_Y + 0.55, color="#444", width=3,
         label="HV Feeder")

    # ════════════════════════════════════════════════════════
    #  3) Transformer
    # ════════════════════════════════════════════════════════
    xfmr_color = _tc()
    xfmr_load_str = f" • {t_pct:.0f}%" if t_pct is not None else ""
    node(CX, XFMR_Y, "🔄 Transformer", f"{kva:.0f} kVA{xfmr_load_str}",
         xfmr_color, symbol="square", size=48, textpos="middle right",
         hover_lines=["<b>Transformer</b>", f"Rated: {kva:.0f} kVA",
                       f"Loading: {t_pct:.1f}%" if t_pct else "Loading: —",
                       "4.16 kV → 480 V"])
    # Transformer symbol: two overlapping circles (drawn as annotations)
    for dx in (-0.2, 0.2):
        fig.add_shape(type="circle",
                      x0=CX + dx - 0.25, y0=XFMR_Y - 0.25,
                      x1=CX + dx + 0.25, y1=XFMR_Y + 0.25,
                      line=dict(color=xfmr_color, width=2),
                      fillcolor="rgba(255,255,255,0.6)")

    # ════════════════════════════════════════════════════════
    #  4) Transformer → LV Busbar wire
    # ════════════════════════════════════════════════════════
    wire(CX, XFMR_Y - 0.55, CX, BUS_Y + 0.15, color="#444", width=3,
         label="LV Output • 480 V")

    # ════════════════════════════════════════════════════════
    #  5) LV Main Busbar (thick horizontal bar)
    # ════════════════════════════════════════════════════════
    bus_color = _vc("LV Main Busbar")
    bus_v = f"  {bv['LV Main Busbar']:.4f} p.u." if "LV Main Busbar" in bv else ""
    lv_left, lv_right = 0.8, 9.2
    fig.add_shape(type="rect",
                  x0=lv_left, y0=BUS_Y - 0.12,
                  x1=lv_right, y1=BUS_Y + 0.12,
                  fillcolor=bus_color,
                  line=dict(color=bus_color, width=0),
                  layer="below")
    fig.add_annotation(x=lv_right + 0.1, y=BUS_Y,
                       text=f"<b>LV Busbar</b>  480 V{bus_v}",
                       showarrow=False, xanchor="left",
                       font=dict(size=9, color="#333"))
    if losses is not None:
        fig.add_annotation(x=lv_left - 0.1, y=BUS_Y,
                           text=f"Losses: {losses:.1f} kW",
                           showarrow=False, xanchor="right",
                           font=dict(size=8, color="#999"))

    # ════════════════════════════════════════════════════════
    #  6) Feeders from busbar down to loads
    # ════════════════════════════════════════════════════════
    feeder_lbl = f"{params['feeder_length_m']:.0f} m"

    # Busbar → Section A (L-shaped: vertical down from busbar, then horiz)
    wire(SEC_A_X, BUS_Y - 0.12, SEC_A_X, LOAD_Y + 0.7,
         color="#2980b9", width=2, dash="dashdot", label=f"Feeder A • {feeder_lbl}")

    # Busbar → Building (straight down)
    wire(BLDG_X, BUS_Y - 0.12, BLDG_X, LOAD_Y + 0.7,
         color="#8e44ad", width=2, dash="dashdot", label=f"Bldg Feeder • {feeder_lbl}")

    # Busbar → Section B
    wire(SEC_B_X, BUS_Y - 0.12, SEC_B_X, LOAD_Y + 0.7,
         color="#2980b9", width=2, dash="dashdot", label=f"Feeder B • {feeder_lbl}")

    # ════════════════════════════════════════════════════════
    #  7) Load nodes (EV Sections + Building)
    # ════════════════════════════════════════════════════════
    sec_a_color = _vc("EV Section A")
    sec_a_v = f" • {bv['EV Section A']:.4f} p.u." if "EV Section A" in bv else ""
    n_a = params["ev_chargers_a"]
    node(SEC_A_X, LOAD_Y, f"🔌 EV Zone A", f"{n_a} chargers{sec_a_v}",
         sec_a_color, symbol="circle",
         size=max(32, min(56, 18 + n_a * 2)),
         textpos="bottom center",
         hover_lines=[f"<b>EV Section A</b>", f"Chargers: {n_a}",
                       f"Voltage: {bv.get('EV Section A', '—')} p.u."])

    bldg_color = _vc("Building")
    bldg_v = f" • {bv['Building']:.4f} p.u." if "Building" in bv else ""
    node(BLDG_X, LOAD_Y, "🏢 Building", f"{params['building_load_kw']:.0f} kW{bldg_v}",
         bldg_color, symbol="pentagon", size=38,
         textpos="bottom center",
         hover_lines=[f"<b>Building Load</b>",
                       f"Base load: {params['building_load_kw']:.0f} kW",
                       f"Voltage: {bv.get('Building', '—')} p.u."])

    sec_b_color = _vc("EV Section B")
    sec_b_v = f" • {bv['EV Section B']:.4f} p.u." if "EV Section B" in bv else ""
    n_b = params["ev_chargers_b"]
    node(SEC_B_X, LOAD_Y, f"🔌 EV Zone B", f"{n_b} chargers{sec_b_v}",
         sec_b_color, symbol="circle",
         size=max(32, min(56, 18 + n_b * 2)),
         textpos="bottom center",
         hover_lines=[f"<b>EV Section B</b>", f"Chargers: {n_b}",
                       f"Voltage: {bv.get('EV Section B', '—')} p.u."])

    # ════════════════════════════════════════════════════════
    #  8) Feasibility banner (top)
    # ════════════════════════════════════════════════════════
    if live_gv is not None:
        banner = "✅  Grid Feasible" if feasible else "❌  Grid Not Feasible"
        if losses is not None:
            banner += f"  •  Losses: {losses:.1f} kW"
        banner_color = "#27ae60" if feasible else "#e74c3c"
        fig.add_annotation(
            x=CX, y=10.2,
            text=f"<b>{banner}</b>",
            showarrow=False,
            font=dict(size=13, color=banner_color),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=banner_color, borderwidth=1, borderpad=6,
        )

    # ── Colour legend ──
    for lc, ln in [
        (IDLE, "No data yet"),
        (OK,   "Normal  (V ≥ 0.95 p.u. / load < 80%)"),
        (WARN, "Warning  (V 0.90–0.95 / load 80–100%)"),
        (CRIT, "Critical  (V < 0.90 / load > 100%)"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=lc, size=10, symbol="circle"),
            name=ln, showlegend=True,
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 11.0]),
        yaxis=dict(visible=False, range=[0.5, 10.8], scaleanchor="x", scaleratio=1),
        height=620,
        plot_bgcolor=BG,
        paper_bgcolor="white",
        margin=dict(t=15, b=5, l=5, r=5),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.06,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        hovermode="closest",
    )
    return fig


# ── OCPP Live Monitor (renders and stops when that view is selected) ──────
if _view == "🔌 OCPP Live Monitor":
    st.title("🔌 OCPP Live Monitor")
    st.caption("Real-time OCPP 1.6J charge-point sessions — auto-refreshes while sessions are active")

    _oc1, _oc2, _oc3 = st.columns([1, 2, 1])
    with _oc1:
        _auto_refresh = st.toggle("Auto-refresh", value=True)
    with _oc2:
        _refresh_s = st.slider("Refresh interval (s)", 2, 30, 5, label_visibility="collapsed")
    with _oc3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    _sessions: list[dict] = []
    _gw_up = False
    try:
        _r = requests.get(f"{GATEWAY_URL}/ocpp/sessions", timeout=5)
        _r.raise_for_status()
        _sessions = _r.json().get("sessions", [])
        _gw_up = True
    except requests.ConnectionError:
        st.error(f"Cannot connect to gateway at `{GATEWAY_URL}`. Make sure the containers are running.")
    except Exception as _ex:
        st.error(f"Error fetching sessions: {_ex}")

    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("Gateway", "🟢 UP" if _gw_up else "🔴 DOWN")
    _k2.metric("Active Sessions", len(_sessions))
    if _sessions:
        _total_kw = sum(s.get("current_power_kw", 0.0) for s in _sessions)
        _total_en  = sum(s.get("energy_needed_kwh", 0.0) for s in _sessions)
        _total_del = sum((s.get("last_meter_wh", 0) - s.get("meter_start_wh", 0)) / 1000.0 for s in _sessions)
        _k3.metric("Total Grid Load", f"{_total_kw:.1f} kW")
        _k4.metric("Fleet Energy Progress", f"{_total_del:.2f} / {_total_en:.1f} kWh")
    else:
        _k3.metric("Total Grid Load", "0 kW")
        _k4.metric("Fleet Energy Progress", "—")

    st.divider()

    # ── Simulation launcher ───────────────────────────────────────────────────
    try:
        _sim_st = requests.get(f"{GATEWAY_URL}/ocpp/simulate/status", timeout=3).json()
    except Exception:
        _sim_st = {"running": False, "vehicles": [], "error": None}

    _sim_running = _sim_st.get("running", False)

    with st.expander("🚀 Launch Simulation" if not _sim_running else "🔴 Simulation Running", expanded=not _sim_running and not _sessions):
        if _sim_running:
            _started = _sim_st.get("started_at", "")
            st.success(f"Fleet simulation is running (started {_started[:19].replace('T',' ')} UTC)")
            _vehs = _sim_st.get("vehicles", [])
            if _vehs:
                _vcols = st.columns(min(len(_vehs), 4))
                for _vi, _vv in enumerate(_vehs):
                    with _vcols[_vi % 4]:
                        _done_icon = "✅" if _vv.get("done") else "⚡"
                        st.caption(
                            f"{_done_icon} **{_vv['cp_id']}**  \n"
                            f"{_vv['battery_kwh']:.0f} kWh · {_vv['initial_soc']:.0f}% SOC  \n"
                            f"{_vv['max_kw']:.1f} kW · stay {_vv['stay_min']:.0f} min"
                        )
            if st.button("⏹ Stop Simulation", type="secondary", use_container_width=True):
                try:
                    requests.post(f"{GATEWAY_URL}/ocpp/simulate/stop", timeout=5)
                    st.rerun()
                except Exception as _e:
                    st.error(f"Stop failed: {_e}")
        else:
            _lc1, _lc2, _lc3 = st.columns(3)
            with _lc1:
                _l_fleet   = st.slider("Fleet size", 1, 10, 3, key="l_fleet")
                _l_real    = st.checkbox("Realistic profiles", value=True, key="l_real")
            with _lc2:
                _l_speed   = st.select_slider(
                    "Speed factor",
                    options=[1, 5, 10, 30, 60, 120, 300],
                    value=60,
                    key="l_speed",
                    help="1 = real time · 60 = 1 min of simulation per real second",
                )
                _l_interval = st.select_slider(
                    "Meter interval (s)",
                    options=[10, 15, 30, 60],
                    value=30,
                    key="l_interval",
                )
            with _lc3:
                _l_seed    = st.number_input("Seed (blank = random)", value=None, step=1,
                                             key="l_seed", format="%d",
                                             placeholder="random")
                _l_minutes = st.number_input("Session minutes (non-realistic)", value=30,
                                             min_value=1, max_value=480, key="l_minutes",
                                             disabled=bool(_l_real))

            if st.button("▶ Start Fleet Simulation", type="primary", use_container_width=True):
                _payload = {
                    "fleet": _l_fleet,
                    "realistic": _l_real,
                    "speed_factor": float(_l_speed),
                    "meter_interval": _l_interval,
                    "seed": int(_l_seed) if _l_seed else None,
                    "session_minutes": float(_l_minutes),
                }
                try:
                    _sr = requests.post(f"{GATEWAY_URL}/ocpp/simulate/start",
                                        json=_payload, timeout=8).json()
                    if _sr.get("started"):
                        st.success(f"Started {_sr['fleet']} vehicles at {_sr['speed_factor']}x speed!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning(_sr.get("reason", "Could not start"))
                except Exception as _e:
                    st.error(f"Failed to start simulation: {_e}")

    st.divider()

    if not _sessions:
        st.info("No active sessions yet. Launch a simulation above or wait for a chargepoint to connect.")
    else:
        for _s in _sessions:
            _cp_id   = _s["cp_id"]
            _conn_id = _s["connector_id"]
            _tx_id   = _s["transaction_id"]
            _id_tag  = _s["id_tag"]
            _stay    = _s.get("predicted_stay_min", 0.0)
            _en_need = _s.get("energy_needed_kwh", 0.0)
            _m_start = _s.get("meter_start_wh", 0.0)
            _m_last  = _s.get("last_meter_wh", 0.0)
            _sched   = _s.get("schedule_kw", [])
            _cur_kw  = _s.get("current_power_kw", 0.0)
            try:
                _arr_dt  = datetime.fromisoformat(_s["arrival_time"])
                _elapsed = (datetime.now(timezone.utc) - _arr_dt).total_seconds() / 60
            except Exception:
                _elapsed = 0.0
            _del_kwh  = (_m_last - _m_start) / 1000.0
            _prog_pct = min(_del_kwh / _en_need * 100, 100.0) if _en_need > 0 else 0.0
            _rem_min  = max(_stay - _elapsed, 0.0)

            with st.container(border=True):
                _h1, _h2, _h3 = st.columns([3, 1, 1])
                _h1.markdown(f"### ⚡ {_cp_id}  ·  Connector {_conn_id}")
                _h2.markdown(f"**TX #{_tx_id}**")
                _h3.markdown(f"ID Tag: `{_id_tag}`")

                _m1, _m2, _m3, _m4, _m5 = st.columns(5)
                _m1.metric("Elapsed",       f"{_elapsed:.0f} min")
                _m2.metric("Remaining",     f"{_rem_min:.0f} min")
                _m3.metric("Current Power", f"{_cur_kw:.2f} kW")
                _m4.metric("Delivered",     f"{_del_kwh:.3f} kWh")
                _m5.metric("Progress",      f"{_prog_pct:.0f}%")
                st.progress(min(_del_kwh / _en_need, 1.0) if _en_need > 0 else 0.0)

                if _sched:
                    _ts_min      = 15
                    _n           = len(_sched)
                    _cur_slot    = min(int(_elapsed / _ts_min), _n - 1)
                    _slot_labels = [f"+{i * _ts_min}m" for i in range(_n)]
                    _max_p       = max(_sched) if max(_sched) > 0 else 1.0
                    _bar_colors  = []
                    for _i, _p in enumerate(_sched):
                        if _i == _cur_slot:
                            _bar_colors.append("rgba(231,76,60,0.85)")
                        elif _p > 0:
                            _alpha = 0.35 + 0.55 * _p / _max_p
                            _bar_colors.append(f"rgba(39,174,96,{_alpha:.2f})")
                        else:
                            _bar_colors.append("rgba(189,195,199,0.4)")

                    _fig_s = go.Figure()
                    _fig_s.add_trace(go.Bar(
                        x=_slot_labels, y=_sched,
                        marker_color=_bar_colors,
                        text=[f"{_p:.1f}" for _p in _sched],
                        textposition="outside", name="kW",
                    ))
                    if 0 <= _cur_slot < _n:
                        _fig_s.add_vline(
                            x=_cur_slot,
                            line=dict(color="rgba(231,76,60,0.6)", width=2, dash="dot"),
                            annotation_text="▶ now",
                            annotation_position="top right",
                            annotation_font=dict(color="rgba(231,76,60,0.9)", size=11),
                        )
                    _fig_s.update_layout(
                        title=dict(text=f"LP Schedule — {_cp_id} connector {_conn_id}", font=dict(size=13)),
                        xaxis_title="Slot (from arrival)", yaxis_title="Power (kW)",
                        height=220, margin=dict(l=40, r=20, t=40, b=30),
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(_fig_s, use_container_width=True)
                    _nonzero = [(f"+{_i*_ts_min}m", _p) for _i, _p in enumerate(_sched) if _p > 0]
                    _summary = "  ·  ".join(f"`{_t}` → **{_p:.1f} kW**" for _t, _p in _nonzero)
                    st.caption(f"Schedule: {_summary}")
                else:
                    st.caption("Schedule not yet available.")

                st.caption(
                    f"Arrival: {_s['arrival_time']} | "
                    f"Predicted stay: {_stay:.0f} min | "
                    f"Key: `{_s['session_key']}`"
                )

    if _auto_refresh and _gw_up:
        time.sleep(_refresh_s)
        st.rerun()

    st.stop()  # do not render simulation UI when OCPP view is active

# ── MLflow Tracking ───────────────────────────────────────────────────────
if _view == "📈 MLflow Tracking":
    st.header("📈 MLflow Experiment Tracking")
    st.caption("Displays MLflow at http://localhost:5000 — compare runs, view metrics, and inspect model artifacts.")
    _components.iframe("http://localhost:5000", height=900, scrolling=True)
    st.stop()

# ── Grafana Monitoring ────────────────────────────────────────────────────
if _view == "📡 Grafana Monitoring":
    st.header("📡 Grafana Monitoring")
    st.caption("Displays Grafana at http://localhost:3000 — system metrics, charging KPIs, and grid health. Default login: admin / admin.")
    _components.iframe("http://localhost:3000", height=900, scrolling=True)
    st.stop()

# ── Prometheus Metrics ────────────────────────────────────────────────────
if _view == "🔬 Prometheus Metrics":
    st.header("🔬 Prometheus Metrics")
    st.caption("Displays Prometheus at http://localhost:9090 — raw metric queries and target scrape status.")
    _components.iframe("http://localhost:9090", height=900, scrolling=True)
    st.stop()


# ── Optimizer Simulation ─────────────────────────────────────────────────
st.title("EV Charging Optimization Dashboard")

# ── Grid Network ─────────────────────────────────────────────────────────
st.header("⚡ Grid Network")
st.caption(
    "**Single-line diagram** of the distribution network powering the EV charging hub. "
    "Power flows top → bottom: the **HV Grid** (4.16 kV) feeds a step-down "
    "**Transformer** that supplies the **LV Busbar** (480 V). From there, underground "
    "feeders connect to **EV Zone A**, **EV Zone B**, and the **Building**. "
    "Adjust charger counts and feeder length in the sidebar. After running a simulation, "
    "nodes turn green/amber/red based on bus voltage and transformer loading."
)
_topo_live = None
if "result" in st.session_state:
    _topo_live = st.session_state["result"].get("ai_result", {}).get("grid_validation")
_topo_params = {
    "transformer_kw": transformer_kw,
    "ev_chargers_a": ev_chargers_a,
    "ev_chargers_b": ev_chargers_b,
    "feeder_length_m": feeder_length_m,
    "building_load_kw": building_load_kw,
}
st.plotly_chart(_topology_figure(_topo_params, _topo_live), use_container_width=True)
if not _topo_live:
    st.info(
        "💡 Nodes are grey (no data). **Run a simulation** below to overlay live "
        "bus voltages, transformer loading, and network losses on the diagram.",
        icon="ℹ️",
    )
st.divider()

# ── EV Configuration ─────────────────────────────────────────────────────
col_rand, _ = st.columns([1, 4])
with col_rand:
    if st.button("🎲 Randomize EVs"):
        st.session_state["rand_seed"] = random.randint(0, 99999)

# ── Compute arrival offsets ──────────────────────────────────────────────
_arrival_rng = random.Random(st.session_state.get("rand_seed", 0) + 9999)
if arrival_mode == "Staggered (random)":
    _arrival_offsets = sorted([_arrival_rng.uniform(0, spread_hours * 60) for _ in range(num_evs)])
elif arrival_mode == "Staggered (evenly spaced)":
    _arrival_offsets = [i * (spread_hours * 60) / max(1, num_evs - 1) for i in range(num_evs)]
else:
    _arrival_offsets = [0.0] * num_evs

_sim_start = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0)
if _sim_start > datetime.now(timezone.utc):
    _sim_start -= timedelta(days=1)

evs = []
cols = st.columns(min(num_evs, 5))
for i in range(num_evs):
    def_soc, def_target, def_kw, def_cap, def_name = _random_ev_defaults(i)
    arr_time = _sim_start + timedelta(minutes=_arrival_offsets[i])
    arr_label = f"+{_arrival_offsets[i]:.0f}min" if _arrival_offsets[i] > 0 else "start"
    col = cols[i % len(cols)]
    with col:
        with st.expander(f"EV {i+1} — {def_name}  ({arr_label})", expanded=(i < 3)):
            battery = st.slider("Current SoC %", 0, 100, def_soc, key=f"bat_{i}_{st.session_state.get('rand_seed',0)}")
            target = st.slider("Target SoC %", battery, 100, max(battery + 5, def_target), key=f"tar_{i}_{st.session_state.get('rand_seed',0)}")
            pow_opts = [3.6, 7.2, 11, 22, 50]
            pow_idx = pow_opts.index(def_kw) if def_kw in pow_opts else 1
            max_power = st.selectbox("Max Power (kW)", pow_opts, index=pow_idx, key=f"pow_{i}_{st.session_state.get('rand_seed',0)}")
            cap_opts = [33, 40, 50, 58, 60, 72, 75, 77, 80, 95, 100]
            cap_idx = cap_opts.index(def_cap) if def_cap in cap_opts else 4
            capacity = st.selectbox("Battery (kWh)", cap_opts, index=cap_idx, key=f"cap_{i}_{st.session_state.get('rand_seed',0)}")
            st.caption(f"Arrives at {arr_time.strftime('%H:%M')}")
            evs.append({
                "ev_id": f"EV-{i+1:03d}",
                "battery_pct": battery,
                "target_pct": target,
                "max_charge_kw": max_power,
                "battery_capacity_kwh": capacity,
                "arrival_time": arr_time.isoformat(),
            })

# ── Run Simulation ──────────────────────────────────────────────────────
if st.button("Run Simulation", type="primary", use_container_width=True):
    payload = {
        "vehicles": evs,
        "transformer_capacity_kw": transformer_kw,
        "building_base_load_kw": building_load_kw,
        "simulation_duration_hours": duration_hours,
        "time_step_minutes": time_step,
        "compare_baseline": compare_baseline,
        "chargers_section_a": ev_chargers_a,
        "chargers_section_b": ev_chargers_b,
        "feeder_length_m": feeder_length_m,
    }

    with st.spinner("Running simulation..."):
        try:
            resp = requests.post(f"{GATEWAY_URL}/api/v1/simulation/run", json=payload, timeout=120)
            resp.raise_for_status()
            st.session_state["result"] = resp.json()
        except requests.ConnectionError:
            st.error("Cannot connect to gateway. Is the system running?")
        except Exception as e:
            st.error(f"Simulation failed: {e}")

# ── Display Results ─────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    ai = result["ai_result"]
    no_ml = result.get("no_ml_result")
    baseline = result.get("baseline_result")
    ml_met = result.get("ml_metrics")

    # ══════════════════════════════════════════════════════════════════════
    #  ML CONTRIBUTION PANEL
    # ══════════════════════════════════════════════════════════════════════
    if ml_met:
        st.header("🤖 ML Model Contribution")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.metric("Departures Predicted by ML",
                      ml_met["departure_predictions_used"],
                      help="Vehicles whose departure time was predicted by the ML model")
        with mcol2:
            avg_stay = ml_met.get("avg_predicted_stay_min")
            st.metric("Avg ML-Predicted Stay",
                      f"{avg_stay:.0f} min" if avg_stay else "N/A",
                      delta=f"vs {ml_met['default_stay_min']:.0f} min fixed" if avg_stay else None,
                      help="ML-predicted stay vs the fixed 4h assumption used by no-ML baseline")
        with mcol3:
            st.metric("Demand Forecast Windows",
                      ml_met["demand_forecast_windows"],
                      help="Future time windows where ML predicted incoming EV demand")
        with mcol4:
            st.metric("Capacity Reserved by ML",
                      f"{ml_met['capacity_reserved_kwh']:.1f} kWh",
                      help="Energy capacity proactively reserved for predicted future arrivals")

        if ml_met["departure_predictions_used"] > 0 and no_ml:
            improvement = ai["overall_satisfaction_pct"] - no_ml["overall_satisfaction_pct"]
            st.info(f"**ML Impact**: Departure prediction + demand forecasting "
                    f"{'improved' if improvement > 0 else 'changed'} satisfaction by "
                    f"**{improvement:+.1f}%** compared to the same optimizer without ML.")

    # ══════════════════════════════════════════════════════════════════════
    #  KEY PERFORMANCE INDICATORS — 3-way comparison
    # ══════════════════════════════════════════════════════════════════════
    st.header("Key Performance Indicators")

    strategies = [("AI + ML", ai, "blue")]
    if no_ml:
        strategies.append(("AI (no ML)", no_ml, "orange"))
    if baseline:
        strategies.append(("FCFS", baseline, "red"))

    kpi_cols = st.columns(len(strategies))
    for idx, (name, strat, color) in enumerate(strategies):
        with kpi_cols[idx]:
            st.subheader(name)
            delta_vs_fcfs = None
            if baseline and strat != baseline:
                delta_vs_fcfs = strat["overall_satisfaction_pct"] - baseline["overall_satisfaction_pct"]

            st.metric("Satisfaction",
                      f"{strat['overall_satisfaction_pct']:.1f}%",
                      delta=f"{delta_vs_fcfs:+.1f}% vs FCFS" if delta_vs_fcfs is not None else None)
            st.metric("Peak Load", f"{strat['peak_load_kw']:.1f} kW")
            st.metric("Energy Delivered", f"{strat['total_energy_delivered_kwh']:.1f} kWh")
            st.metric("Overload Slots", f"{strat['overload_slots']}")

    # ── Comparison bar chart ──
    st.subheader("Satisfaction Comparison")
    fig_bar = go.Figure()
    colors = {"AI + ML": "#1f77b4", "AI (no ML)": "#ff7f0e", "FCFS": "#d62728"}
    for name, strat, color in strategies:
        fig_bar.add_trace(go.Bar(
            x=[name], y=[strat["overall_satisfaction_pct"]],
            name=name, marker_color=colors.get(name, color),
            text=[f"{strat['overall_satisfaction_pct']:.1f}%"],
            textposition="auto",
        ))
    fig_bar.update_layout(yaxis_title="Satisfaction (%)", height=350, showlegend=False,
                          yaxis=dict(range=[0, 105]))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    #  LOAD PROFILE CHART — all strategies
    # ══════════════════════════════════════════════════════════════════════
    st.header("Load Profile Comparison")
    fig = go.Figure()

    ai_ts = ai["time_series"]
    times = [p["time_minutes"] / 60 for p in ai_ts]

    fig.add_trace(go.Scatter(
        x=times, y=[p["total_load_kw"] for p in ai_ts],
        name="AI+ML Total Load", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=times, y=[p["building_load_kw"] for p in ai_ts],
        name="Building Load", line=dict(color="gray", dash="dot"),
    ))

    if no_ml:
        no_ml_ts = no_ml["time_series"]
        fig.add_trace(go.Scatter(
            x=times, y=[p["total_load_kw"] for p in no_ml_ts],
            name="AI (no ML) Total Load", line=dict(color="#ff7f0e", width=2, dash="dashdot"),
        ))

    if baseline:
        bl_ts = baseline["time_series"]
        fig.add_trace(go.Scatter(
            x=times, y=[p["total_load_kw"] for p in bl_ts],
            name="FCFS Total Load", line=dict(color="#d62728", width=2, dash="dash"),
        ))

    fig.add_hline(y=transformer_kw, line_dash="dash", line_color="darkred",
                  annotation_text=f"Transformer Limit ({transformer_kw} kW)")

    fig.update_layout(
        xaxis_title="Time (hours)", yaxis_title="Load (kW)", height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Transformer Utilization ──
    st.header("Transformer Utilization")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=times, y=[p["transformer_utilization_pct"] for p in ai_ts],
        name="AI+ML", fill="tozeroy", line=dict(color="#1f77b4"),
    ))
    if no_ml:
        fig2.add_trace(go.Scatter(
            x=times, y=[p["transformer_utilization_pct"] for p in no_ml_ts],
            name="AI (no ML)", line=dict(color="#ff7f0e", dash="dashdot"),
        ))
    if baseline:
        fig2.add_trace(go.Scatter(
            x=times, y=[p["transformer_utilization_pct"] for p in bl_ts],
            name="FCFS", line=dict(color="#d62728", dash="dash"),
        ))
    fig2.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Limit")
    fig2.add_hline(y=80, line_dash="dot", line_color="orange", annotation_text="80% Warning")
    fig2.update_layout(xaxis_title="Time (hours)", yaxis_title="Utilization (%)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    #  PER-VEHICLE RESULTS — 3-column comparison
    # ══════════════════════════════════════════════════════════════════════
    st.header("Per-Vehicle Results")
    ev_cols = st.columns(len(strategies))

    for idx, (name, strat, color) in enumerate(strategies):
        with ev_cols[idx]:
            st.subheader(name)
            for ev in strat["ev_results"]:
                pct = ev["satisfaction_pct"]
                clr = "green" if pct >= 90 else "orange" if pct >= 50 else "red"
                # Find this EV's arrival time from the input
                ev_input = next((e for e in evs if e["ev_id"] == ev["ev_id"]), None)
                arr_str = ""
                if ev_input and "arrival_time" in ev_input:
                    try:
                        arr_dt = datetime.fromisoformat(ev_input["arrival_time"])
                        arr_str = f" — arrives {arr_dt.strftime('%H:%M')}"
                    except Exception:
                        pass
                st.markdown(f"**{ev['ev_id']}{arr_str}**: {ev['energy_delivered_kwh']:.1f}/{ev['energy_needed_kwh']:.1f} kWh "
                            f"(:{clr}[{pct:.0f}%])")

    # ══════════════════════════════════════════════════════════════════════
    #  ML MODEL MONITORING PANEL
    # ══════════════════════════════════════════════════════════════════════
    st.header("🔍 ML Model Monitoring")

    try:
        ml_status = requests.get(f"{GATEWAY_URL}/api/v1/ml/metrics", timeout=5).json()

        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.subheader("Model 1: Demand Forecasting")
            if "evaluation" in ml_status:
                ev_metrics = ml_status["evaluation"]
                st.metric("Arrival Count MAE", f"{ev_metrics.get('demand_arrival_count_mae', 0):.3f}")
                st.metric("Arrival Count RMSE", f"{ev_metrics.get('demand_arrival_count_rmse', 0):.3f}")
                st.metric("Total kWh MAE", f"{ev_metrics.get('demand_total_kwh_mae', 0):.2f}")
                st.metric("Total kWh RMSE", f"{ev_metrics.get('demand_total_kwh_rmse', 0):.2f}")

            if "models" in ml_status and "demand_forecast" in ml_status["models"]:
                m = ml_status["models"]["demand_forecast"]
                st.caption(f"**Type**: {m['type']}")
                st.caption(f"**Description**: {m['description']}")
                with st.expander("Input Features"):
                    for feat in m.get("input_features", []):
                        st.code(feat)

        with mcol2:
            st.subheader("Model 2: Departure Prediction")
            if "evaluation" in ml_status:
                st.metric("Stay Duration MAE", f"{ev_metrics.get('departure_mae_min', 0):.1f} min")
                st.metric("Stay Duration RMSE", f"{ev_metrics.get('departure_rmse_min', 0):.1f} min")
                st.metric("Within 15 min", f"{ev_metrics.get('departure_within_15min_pct', 0):.1f}%")
                st.metric("Within 30 min", f"{ev_metrics.get('departure_within_30min_pct', 0):.1f}%")

            if "models" in ml_status and "departure_prediction" in ml_status["models"]:
                m = ml_status["models"]["departure_prediction"]
                st.caption(f"**Type**: {m['type']}")
                st.caption(f"**Description**: {m['description']}")
                with st.expander("Input Features"):
                    for feat in m.get("input_features", []):
                        st.code(feat)

        if "evaluation" in ml_status and "acceptance_checks" in ml_status["evaluation"]:
            st.subheader("Acceptance Tests")
            checks = ml_status["evaluation"]["acceptance_checks"]
            for check, passed in checks.items():
                if passed:
                    st.success(f"✅ {check}")
                else:
                    st.warning(f"⚠️ {check}")

        st.caption(f"Model version: {ml_status.get('model_version', 'unknown')}")

    except Exception as e:
        st.warning(f"ML monitoring unavailable: {e}")

    # ── Grid Validation ──
    if ai.get("grid_validation"):
        st.header("⚡ Grid Validation (OpenDSS AC Power Flow)")
        gv = ai["grid_validation"]

        gc1, gc2, gc3, gc4 = st.columns(4)
        with gc1:
            color = "normal" if gv.get("feasible") else "inverse"
            st.metric("Feasible", "✅ Yes" if gv.get("feasible") else "❌ No")
        with gc2:
            loading = gv.get("max_transformer_loading_pct", 0)
            st.metric(
                "Transformer Loading",
                f"{loading:.1f}%",
                delta=f"{loading - 100:.1f}%" if loading > 100 else None,
                delta_color="inverse" if loading > 100 else "normal",
            )
        with gc3:
            v_min = gv.get("min_bus_voltage_pu", 1.0)
            st.metric(
                "Min Bus Voltage",
                f"{v_min:.4f} p.u.",
                delta=f"{v_min - 0.95:.4f}" if v_min < 0.95 else None,
                delta_color="inverse",
            )
        with gc4:
            losses = gv.get("network_losses_kw", 0.0)
            st.metric("Network Losses", f"{losses:.1f} kW")

        # ── Per-bus voltage bar chart ──
        bus_voltages = gv.get("bus_voltages_pu", {})
        if bus_voltages:
            bus_names = list(bus_voltages.keys())
            bus_vals = list(bus_voltages.values())
            bar_colors = [
                "#d62728" if v < 0.90 else "#ff7f0e" if v < 0.95 else "#2ca02c"
                for v in bus_vals
            ]
            fig_v = go.Figure(
                go.Bar(
                    x=bus_names,
                    y=bus_vals,
                    marker_color=bar_colors,
                    text=[f"{v:.4f}" for v in bus_vals],
                    textposition="outside",
                )
            )
            fig_v.update_layout(
                title="Bus Voltage Profile (p.u.)",
                xaxis_title="Bus",
                yaxis_title="Voltage (p.u.)",
                yaxis=dict(range=[0.88, 1.05]),
                height=280,
                margin=dict(t=40, b=10, l=10, r=10),
                shapes=[
                    dict(
                        type="line", y0=0.95, y1=0.95,
                        x0=-0.5, x1=len(bus_names) - 0.5,
                        line=dict(color="orange", dash="dash", width=1),
                    ),
                    dict(
                        type="line", y0=0.90, y1=0.90,
                        x0=-0.5, x1=len(bus_names) - 0.5,
                        line=dict(color="red", dash="dash", width=1),
                    ),
                ],
            )
            st.plotly_chart(fig_v, use_container_width=True)
            st.caption("🟠 Dashed orange = 0.95 p.u. warning threshold  |  🔴 Dashed red = 0.90 p.u. critical (ANSI C84.1)")

        # ── Overload & warnings ──
        overload = gv.get("overload_minutes", 0)
        if overload > 0:
            st.warning(f"⚠️ Transformer overloaded for **{overload:.0f} minutes** during this session")
        warnings = gv.get("warnings", [])
        if warnings:
            with st.expander(f"Grid warnings ({len(warnings)})"):
                for w in warnings:
                    st.write(f"• {w}")

    # ── Raw JSON ──
    with st.expander("Raw API Response"):
        st.json(result)
