"""
Generates a professional PDF report explaining the EV Charging Optimizer + ML coupling.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import ListFlowable, ListItem
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "EV_Optimizer_ML_Report.pdf")

# ── Colour palette ───────────────────────────────────────────────────────────
BLUE_DARK  = colors.HexColor("#1a3a5c")
BLUE_MID   = colors.HexColor("#2e6da4")
BLUE_LIGHT = colors.HexColor("#d6e8f7")
ACCENT     = colors.HexColor("#e8a020")
GREY_BG    = colors.HexColor("#f4f6f9")
GREY_LINE  = colors.HexColor("#c0c8d4")
WHITE      = colors.white
BLACK      = colors.black


def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#cce0f5"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    styles["h1"] = ParagraphStyle(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=22,
        textColor=BLUE_DARK,
        spaceBefore=18,
        spaceAfter=6,
    )
    styles["h2"] = ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=BLUE_MID,
        spaceBefore=12,
        spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=BLACK,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles["mono"] = ParagraphStyle(
        "mono",
        fontName="Courier",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e1e1e"),
        backColor=colors.HexColor("#eef2f7"),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=6,
        borderPad=4,
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=BLACK,
        leftIndent=16,
        spaceAfter=3,
    )
    styles["label"] = ParagraphStyle(
        "label",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=BLUE_DARK,
    )
    styles["cell"] = ParagraphStyle(
        "cell",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=BLACK,
    )
    return styles


def section_header(title, styles):
    """Returns a visually distinct section header block."""
    return [
        HRFlowable(width="100%", thickness=2, color=BLUE_MID, spaceAfter=4),
        Paragraph(title, styles["h1"]),
    ]


def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=2.2*cm,
        rightMargin=2.2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title="EV Charging: ML + Optimizer Architecture",
        author="EV Smart Charging Project",
    )

    S = build_styles()
    W = A4[0] - 4.4*cm  # usable width

    story = []

    # ── Cover banner ─────────────────────────────────────────────────────────
    cover_data = [[
        Paragraph("EV Smart Charging System", S["cover_title"]),
    ]]
    cover_sub_data = [[
        Paragraph("ML Models + LP Optimizer: Architecture &amp; Coupling", S["cover_sub"]),
    ]]
    cover_meta = [[
        Paragraph("April 2026  |  Project Report", S["cover_sub"]),
    ]]

    def make_banner(data, bg, pad=(14, 8, 14, 8)):
        t = Table(data, colWidths=[W])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("TOPPADDING",    (0, 0), (-1, -1), pad[0]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad[1]),
            ("LEFTPADDING",   (0, 0), (-1, -1), pad[2]),
            ("RIGHTPADDING",  (0, 0), (-1, -1), pad[3]),
            ("ROUNDEDCORNERS", [6]),
        ]))
        return t

    story.append(make_banner(cover_data, BLUE_DARK, (20, 6, 20, 6)))
    story.append(make_banner(cover_sub_data, BLUE_MID, (8, 6, 20, 6)))
    story.append(make_banner(cover_meta, colors.HexColor("#3a7bbf"), (6, 10, 20, 6)))
    story.append(Spacer(1, 0.5*cm))

    # ── 1. Overview ──────────────────────────────────────────────────────────
    story += section_header("1. System Overview", S)
    story.append(Paragraph(
        "The EV Smart Charging system is a microservice platform that schedules "
        "charging power across multiple electric vehicles simultaneously. It is built "
        "around a <b>Predict-Then-Optimize</b> architecture: Machine Learning models "
        "first produce predictions about each vehicle's future behaviour, and a "
        "Linear Programming (LP) optimizer then allocates power using those predictions "
        "as inputs to its constraint set.",
        S["body"]
    ))
    story.append(Paragraph(
        "The system runs <b>three strategies in parallel</b> for every simulation to "
        "quantify ML value: (1) <b>AI + ML</b> — LP optimizer with ML departure "
        "predictions and demand forecasts; (2) <b>AI only</b> — LP optimizer with "
        "historical mean departures, no ML; (3) <b>FCFS</b> — First-Come-First-Served "
        "greedy, no ML, no optimization.",
        S["body"]
    ))

    # Architecture flow table
    flow_data = [
        [Paragraph("Step", S["label"]), Paragraph("Component", S["label"]),
         Paragraph("Action", S["label"]), Paragraph("Output", S["label"])],
        ["1", "ML Service", "Predict departure per vehicle", "Stay duration (q10/q50/q90)"],
        ["2", "ML Service", "Forecast future arrivals (aggregate)", "Count + kWh per future window"],
        ["3", "Gateway Orchestrator", "Build optimizer request", "Per-vehicle slots + future arrivals"],
        ["4", "LP Optimizer", "Solve 2-phase LP", "Power schedule per vehicle per slot"],
        ["5", "Gateway", "Return results + collect feedback", "Satisfaction %, energy delivered"],
    ]
    col_w = [W * f for f in [0.07, 0.20, 0.40, 0.33]]
    flow_table = Table(flow_data, colWidths=col_w, repeatRows=1)
    flow_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(flow_table)
    story.append(Paragraph("Table 1 — End-to-end data flow through the system.", S["caption"]))

    # ── 2. ML Models ────────────────────────────────────────────────────────
    story += section_header("2. Machine Learning Models", S)

    # Model 1
    story.append(Paragraph("2.1  Departure Time Predictor (M1)", S["h2"]))
    story.append(Paragraph(
        "A <b>Quantile Regression</b> model (HistGradientBoostingRegressor) trained on "
        "23,444 real sessions from the ACN-Data Caltech dataset. For each connected "
        "vehicle it outputs three quantile estimates of how long the vehicle will stay:",
        S["body"]
    ))
    q_data = [
        [Paragraph("Quantile", S["label"]), Paragraph("Meaning", S["label"]),
         Paragraph("Use in optimizer", S["label"])],
        ["q10  (10th percentile)", "Earliest realistic departure", "Conservative lower bound"],
        ["q50  (median)",          "Most likely departure time",   "Default departure_slot input"],
        ["q90  (90th percentile)", "Latest realistic departure",   "Optimistic upper bound"],
    ]
    q_table = Table(q_data, colWidths=[W*0.30, W*0.38, W*0.32], repeatRows=1)
    q_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE_MID),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BLUE_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(q_table)
    story.append(Paragraph("Table 2 — Departure quantile outputs and their roles.", S["caption"]))
    story.append(Spacer(1, 0.2*cm))

    inp_data = [
        [Paragraph("Input Feature", S["label"]), Paragraph("Description", S["label"])],
        ["arrival_time",         "Timestamp of vehicle connection (OCPP event)"],
        ["arrival_hour",         "Hour of day (0–23), captures commuter patterns"],
        ["day_of_week",          "0=Monday … 6=Sunday"],
        ["is_weekend",           "Binary flag"],
        ["requested_energy_kwh", "Energy needed = battery_capacity × (target_pct − current_pct)"],
        ["site_id / cluster_id", "Charging site identifier (affects stay distribution)"],
    ]
    inp_table = Table(inp_data, colWidths=[W*0.38, W*0.62], repeatRows=1)
    inp_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(inp_table)
    story.append(Paragraph("Table 3 — Departure model input features.", S["caption"]))

    story.append(Paragraph(
        "Performance: test-set MAE ≈ 112 min vs. 184 min for the ACN per-hour "
        "historical mean baseline — a <b>~39% reduction in departure prediction error</b>.",
        S["body"]
    ))

    # Model 2
    story.append(Paragraph("2.2  Arrivals / Demand Forecaster (M2)", S["h2"]))
    story.append(Paragraph(
        "A time-series model that predicts how many EVs will arrive in each upcoming "
        "30-minute window and how much energy (kWh) they will need in total. "
        "The optimizer uses this to <b>reserve grid capacity</b> for vehicles that "
        "have not yet arrived — preventing over-commitment to current vehicles.",
        S["body"]
    ))

    # ── 3. LP Optimizer ─────────────────────────────────────────────────────
    story += section_header("3. LP Optimizer", S)

    story.append(Paragraph(
        "The optimizer (<tt>lp_scheduler.py</tt>) uses <b>scipy HiGHS interior-point LP</b> "
        "to allocate charging power across all vehicles and time slots simultaneously. "
        "Time is discretized into 15-minute slots. Each decision variable "
        "<i>x\u2085\u1d62\u2096</i> is the power (kW) assigned to vehicle <i>i</i> "
        "in slot <i>k</i>.",
        S["body"]
    ))

    story.append(Paragraph("3.1  Inputs to the Optimizer", S["h2"]))
    opt_inp_data = [
        [Paragraph("Parameter", S["label"]), Paragraph("Source", S["label"]),
         Paragraph("Description", S["label"])],
        ["vehicles[i].departure_slot",        "ML M1 (q50)",            "When vehicle i is expected to leave"],
        ["vehicles[i].energy_needed_kwh",     "Battery state (OCPP)",   "kWh still needed to reach target SoC"],
        ["vehicles[i].arrival_slot",          "OCPP event",             "Slot when vehicle connected"],
        ["vehicles[i].max_charge_kw",         "Hardware spec",          "Maximum charger power limit"],
        ["transformer_capacity_kw",           "Grid config",            "Total site power budget (kW)"],
        ["base_load_per_slot_kw[]",           "Building metering",      "Non-EV background load per slot"],
        ["predicted_future_arrivals[]",       "ML M2 forecast",         "Expected new EVs + kWh per future slot"],
    ]
    opt_inp_table = Table(opt_inp_data,
                          colWidths=[W*0.32, W*0.22, W*0.46], repeatRows=1)
    opt_inp_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(opt_inp_table)
    story.append(Paragraph("Table 4 — Optimizer inputs and their sources.", S["caption"]))

    # ── 4. Constraints ──────────────────────────────────────────────────────
    story += section_header("4. Constraints", S)

    constraints = [
        ("Grid Capacity (per slot)",
         "Sum of power across all vehicles in slot k must not exceed the available "
         "transformer headroom, after subtracting building base load and ML-reserved "
         "capacity for predicted future arrivals:",
         "  ∑ᵢ x_{i,k}  ≤  P_max  −  base_load_k  −  reserved_k     for all k"),
        ("Hardware Limit (per vehicle per slot)",
         "Power assigned to vehicle i in slot k is bounded by the physical charger rating "
         "and zero if the vehicle is not present:",
         "  0  ≤  x_{i,k}  ≤  p_i^max  ×  presence_{i,k}"),
        ("Energy Cap (per vehicle)",
         "The total energy delivered to a vehicle cannot exceed what it needs:",
         "  ∑_k x_{i,k} · Δt  ≤  e_i     for all i"),
        ("Minimum Guarantee (per vehicle)",
         "Every vehicle is guaranteed at least 20% of its requested energy, "
         "capped at the physical maximum achievable within its stay window. "
         "This prevents complete starvation:",
         "  ∑_k x_{i,k} · Δt  ≥  0.20 · e_i     for all i"),
    ]

    for title, desc, formula in constraints:
        story.append(KeepTogether([
            Paragraph(title, S["h2"]),
            Paragraph(desc, S["body"]),
            Paragraph(formula, S["mono"]),
            Spacer(1, 0.15*cm),
        ]))

    # ML reservation note
    rsv_box = Table(
        [[Paragraph(
            "<b>ML-driven capacity reservation:</b>  The demand forecast (M2) outputs "
            "predicted_count and predicted_kwh for each future time slot.  "
            "The optimizer computes reserved_k = predicted_count × avg_charge_kw and "
            "subtracts this from the available headroom.  This ensures that when "
            "predicted vehicles actually arrive, there is grid capacity waiting for them.",
            S["body"]
        )]],
        colWidths=[W]
    )
    rsv_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE_LIGHT),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX",           (0, 0), (-1, -1), 1.2, BLUE_MID),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(rsv_box)
    story.append(Spacer(1, 0.2*cm))

    # ── 5. Objective Function ───────────────────────────────────────────────
    story += section_header("5. Objective Function (2-Phase LP)", S)

    story.append(Paragraph(
        "The optimizer solves two LP problems sequentially. Phase 1 finds the "
        "maximum total energy achievable; Phase 2 uses that as a floor constraint "
        "and optimizes fairness.",
        S["body"]
    ))

    story.append(Paragraph("Phase 1 — Maximize Total Energy Delivered", S["h2"]))
    story.append(Paragraph(
        "Each vehicle × slot variable is weighted by an <b>urgency score</b> "
        "(vehicles with less remaining charging time get higher weight) and a "
        "<b>front-loading decay</b> (earlier slots in a vehicle's window are "
        "preferred over later slots):",
        S["body"]
    ))
    story.append(Paragraph(
        "  maximize  ∑ᵢ ∑_k  wᵢ · decay_{i,k} · x_{i,k} · Δt\n\n"
        "  where  wᵢ = 1.0 + 0.5 × min(urgencyᵢ, 2.0)\n"
        "         decay_{i,k} ∈ [0.3, 1.0]  (1.0 at arrival, 0.3 at departure)\n"
        "         urgencyᵢ = e_i / (available_slots × Δt × p_i^max)",
        S["mono"]
    ))
    story.append(Paragraph(
        "<b>Why front-loading?</b>  ML predicted a vehicle stays 8 hours, but it "
        "might leave after 3. By giving higher weight to early slots, the optimizer "
        "delivers energy quickly — making the schedule <b>robust to departure "
        "uncertainty</b> even when the ML prediction is imperfect.",
        S["body"]
    ))

    story.append(Paragraph("Phase 2 — Max-Min Fairness", S["h2"]))
    story.append(Paragraph(
        "Given the optimal total energy E* from Phase 1, Phase 2 introduces a "
        "scalar variable <i>t</i> representing the minimum satisfaction fraction "
        "across all vehicles, and maximises it:",
        S["body"]
    ))
    story.append(Paragraph(
        "  maximize  t\n\n"
        "  subject to:\n"
        "    satᵢ = ∑_k x_{i,k}·Δt / eᵢ  ≥  t         for all i\n"
        "    ∑_{i,k} x_{i,k}·Δt           ≥  0.95 · E*  (efficiency floor)\n"
        "    t  ≤  1.0",
        S["mono"]
    ))
    story.append(Paragraph(
        "This eliminates the 100%/0% extreme outcomes. A vehicle that physically "
        "cannot reach 10% satisfaction within its time window (e.g. very short stay + "
        "large battery) is excluded from the fairness constraint so it does not "
        "drag the minimum down for the entire fleet.",
        S["body"]
    ))

    # ── 6. Goal Summary ─────────────────────────────────────────────────────
    story += section_header("6. Goal Summary", S)

    goal_box = Table(
        [[Paragraph(
            "<b>Primary Goal:</b>  Maximize the total energy delivered to all "
            "connected EVs within physical grid capacity, while ensuring no vehicle "
            "is starved (20% minimum guarantee), preferring early delivery to handle "
            "departure uncertainty from ML predictions, and reserving capacity for "
            "ML-predicted future arrivals — all subject to transformer, hardware, "
            "and energy constraints.",
            S["body"]
        )]],
        colWidths=[W]
    )
    goal_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#fff8e8")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 1.5, ACCENT),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(goal_box)
    story.append(Spacer(1, 0.3*cm))

    # ── 7. ML Value Quantified ──────────────────────────────────────────────
    story += section_header("7. ML Value — Quantified Comparison", S)

    story.append(Paragraph(
        "The system automatically compares all three strategies on every run "
        "to prove that ML integration improves outcomes over the no-ML baseline:",
        S["body"]
    ))

    cmp_data = [
        [Paragraph("Metric", S["label"]),
         Paragraph("AI + ML", S["label"]),
         Paragraph("AI only (no ML)", S["label"]),
         Paragraph("FCFS (no opt.)", S["label"])],
        ["Departure estimate",    "ML q50 prediction",     "ACN per-hour mean",   "ACN per-hour mean"],
        ["Capacity reservation",  "ML demand forecast",    "None",                "None"],
        ["Departure MAE",         "~112 min",              "~184 min",            "~184 min"],
        ["Scheduling method",     "2-phase LP",            "2-phase LP",          "Greedy FCFS"],
        ["Min. guarantee",        "20% per vehicle",       "20% per vehicle",     "None"],
        ["Fairness objective",    "Max-min satisfaction",  "Max-min satisfaction","First-come only"],
    ]
    cmp_table = Table(cmp_data,
                      colWidths=[W*0.28, W*0.24, W*0.24, W*0.24], repeatRows=1)
    cmp_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREY_BG]),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        # Highlight AI+ML column
        ("BACKGROUND",    (1, 1), (1, -1), colors.HexColor("#e8f4e8")),
        ("FONTNAME",      (1, 1), (1, -1), "Helvetica-Bold"),
    ]))
    story.append(cmp_table)
    story.append(Paragraph("Table 5 — Side-by-side strategy comparison.", S["caption"]))

    story.append(Paragraph(
        "The ML departure predictor achieves ~39% lower MAE than the historical mean "
        "baseline. This translates directly to the optimizer receiving more accurate "
        "departure_slot values → tighter, more efficient power allocation → higher "
        "overall satisfaction percentage and more energy delivered per session.",
        S["body"]
    ))

    # ── 8. Feedback Loop ────────────────────────────────────────────────────
    story += section_header("8. Online Feedback &amp; Retraining Loop", S)

    story.append(Paragraph(
        "After each simulation, the gateway writes two feedback files to disk:",
        S["body"]
    ))

    fb_data = [
        [Paragraph("File", S["label"]), Paragraph("Content", S["label"]),
         Paragraph("Used for", S["label"])],
        ["collected_sessions.csv",
         "Actual session records (arrival, departure, kWh)",
         "Periodic retraining of M1 departure model"],
        ["departure_feedback.csv",
         "Predicted vs. actual stay duration per vehicle",
         "Drift monitoring (drift_monitor.py) + MAE tracking"],
    ]
    fb_table = Table(fb_data, colWidths=[W*0.30, W*0.40, W*0.30], repeatRows=1)
    fb_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE_MID),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BLUE_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, GREY_LINE),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(fb_table)
    story.append(Paragraph("Table 6 — Feedback files written after each simulation run.", S["caption"]))

    story.append(Paragraph(
        "A Kubernetes CronJob (<tt>retrain-cronjob.yaml</tt>) re-runs the training "
        "pipeline weekly using accumulated real-session data, enabling the ML models "
        "to improve over time as more charging behaviour is observed.",
        S["body"]
    ))

    # ── Footer line ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=GREY_LINE))
    story.append(Paragraph(
        "EV Smart Charging Project  ·  April 2026  ·  Confidential / Academic Use",
        S["caption"]
    ))

    doc.build(story)
    print(f"PDF saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
