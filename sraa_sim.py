# -*- coding: utf-8 -*-
"""
Simulador didático do Sistema Renina-Angiotensina-Aldosterona (SRAA)

Execução:
    streamlit run sraa_sim_corrigido.py

IMPORTANTE:
- Este é um modelo didático de índices relativos, não um simulador clínico validado.
- Os ganhos das equações são ilustrativos e não foram ajustados a dados de pacientes.
- Na, K, renina, angiotensinas e receptores são índices normalizados (basal sem fármacos = 1),
  exceto PAM (mmHg) e TFG estimada (mL/min/1,73 m²).
- O regime farmacológico selecionado é considerado já presente antes da perturbação.
"""

import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

st.set_page_config(page_title="Simulador didático do SRAA", layout="wide")

# =====================================================================
# 1. PARÂMETROS E CONVENÇÕES
# =====================================================================
PAM0 = 90.0
GFR0 = 112.0  # referência didática da TFG estimada, em mL/min/1,73 m²
T_PRE = 2.0   # minutos exibidos antes do início da perturbação
DISTURBANCE_TAU = 0.20  # min; subida suave da perturbação após t = 0

# As taxas são por minuto. Para variáveis dinâmicas de primeira ordem,
# k = ln(2)/t½. As escalas são ilustrativas; não representam uma calibração clínica.
PARAMS = {
    # Hormônios / mediadores de resposta rápida
    "k_ren": np.log(2) / 15.0,       # escala de resposta da atividade de renina
    "k_a1": np.log(2) / 3.0,         # Ang I
    "k_a2": np.log(2) / 1.5,         # Ang II (escala rápida ilustrativa)
    "k_ald": np.log(2) / 35.0,       # aldosterona: resposta mais lenta
    "k_adh": np.log(2) / 7.0,        # ADH, escala ilustrativa
    "k_bk": np.log(2) / 1.5,         # bradicinina, escala ilustrativa
    "k_symp": np.log(2) / 0.5,       # resposta simpática simplificada

    # Ocupação normalizada de receptores
    "Kd_AT1": 1.0,
    "Kd_AT2": 2.0,
    "arb_kd_multiplier": 19.0,

    # Ganhos adimensionais: selecionados para comportamento qualitativo estável
    "g_ren_press": 2.0,
    "g_ren_symp": 0.65,
    "g_ren_tgf": 1.2,
    "g_ren_feedback_at1": 1.5,
    "g_ald_k": 0.8,
    "g_adh_press": 1.2,
    "g_adh_vol": 0.5,
    "g_symp_press": 1.8,

    # Hemodinâmica. São ganhos didáticos, não parâmetros clínicos calibrados.
    "g_pam_vol": 1.2,
    "g_pam_at1": 0.28,
    "g_pam_at2": 0.08,
    "g_pam_bk": 0.06,
    "g_pam_symp": 0.08,
    "g_ra_pressure": 0.75,
    "g_re_at1": 0.30,

    # Balanços lentos: tempos característicos em horas, convertidos para minutos
    "k_na": 1.0 / (12.0 * 60.0),       # ~12 h
    "k_k": 1.0 / (5.0 * 60.0),         # ~5 h
    "k_vol": 1.0 / (18.0 * 60.0),      # ~18 h
    "g_na_press": 0.80,
    "g_na_ald": 0.25,
    "g_na_gfr": 0.30,
    "g_na_vol": 0.80,
    "g_k_ald": 0.70,
    "g_k_gfr": 0.45,
    "g_vol_na": 0.45,
    "g_vol_adh": 0.12,
}

STATE_NAMES = ["R", "A1", "A2", "Ald", "ADH", "BK", "Na_body", "K_plasma", "Vol", "S"]
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}


def clamp(value, low, high):
    """Limita um valor escalar ao intervalo especificado."""
    return float(np.clip(value, low, high))


def disturbance_factor(t, drop):
    """Fator de perturbação: igual a 1 antes de t=0 e cai suavemente depois."""
    if drop <= 0.0 or t < 0.0:
        return 1.0
    progress = 1.0 - np.exp(-t / DISTURBANCE_TAU)
    return 1.0 - drop * progress


def receptor_effects(a2, drugs, p=PARAMS):
    """Atividades efetivas dos receptores, normalizadas para 1 no basal sem fármacos.

    O BRA é representado didaticamente como aumento da Kd aparente de AT1.
    Isso é uma aproximação simples de antagonismo competitivo, não uma PK/PD completa.
    """
    a2 = max(float(a2), 1e-8)
    kd_at1_eff = p["Kd_AT1"] * (1.0 + p["arb_kd_multiplier"] * drugs["arb"])

    occ_at1 = a2 / (kd_at1_eff + a2)
    occ_at2 = a2 / (p["Kd_AT2"] + a2)

    occ_at1_baseline = 1.0 / (p["Kd_AT1"] + 1.0)
    occ_at2_baseline = 1.0 / (p["Kd_AT2"] + 1.0)
    eff_at1 = occ_at1 / occ_at1_baseline
    eff_at2 = occ_at2 / occ_at2_baseline
    return float(eff_at1), float(eff_at2)


def hemodynamics(pam, a2, drugs, p=PARAMS):
    """TFG didática estimada a partir da perfusão e de resistências relativas.

    A resistência aferente diminui quando a perfusão cai (dilatação compensatória)
    e aumenta quando a perfusão sobe. A resistência eferente depende do sinal AT1.
    Isso é uma aproximação fenomenológica, não um modelo de néfron completo.
    """
    eff_at1, _ = receptor_effects(a2, drugs, p)
    pressure_ratio = max(float(pam), 1.0) / PAM0

    # Sinal correto da resposta miogênica simplificada: menor pressão -> menor Ra.
    r_aff = clamp(1.0 + p["g_ra_pressure"] * (pressure_ratio - 1.0), 0.60, 1.50)
    r_eff = clamp(1.0 + p["g_re_at1"] * (eff_at1 - 1.0), 0.60, 1.60)

    # Normalizado para GFR0 quando PAM, Ra e Re estão em seus valores basais.
    gfr = GFR0 * pressure_ratio * (2.0 * r_eff / (r_aff + r_eff))
    gfr = clamp(gfr, 5.0, 220.0)

    # Proxy de entrega tubular para o TGF. NÃO é conteúdo corporal de sódio.
    # O modelo não representa transporte segmentar nem NaCl medido na mácula densa.
    distal_delivery_ratio = clamp(gfr / GFR0, 0.20, 2.50)
    return gfr, distal_delivery_ratio, r_aff, r_eff


def model_outputs(t, y, drop, drugs, p=PARAMS):
    """Calcula variáveis algébricas: PAM, TFG, receptor e sinais de feedback."""
    R, A1, A2, Ald, ADH, BK, Na_body, K_plasma, Vol, S = [max(float(v), 1e-6) for v in y]
    D = disturbance_factor(t, drop)
    eff_at1, eff_at2 = receptor_effects(A2, drugs, p)

    # Fator vascular relativo. O basal (AT1=AT2=BK=S=Vol=1) é exatamente 1.
    vascular_factor = (
        1.0
        + p["g_pam_at1"] * (eff_at1 - 1.0)
        - p["g_pam_at2"] * (eff_at2 - 1.0)
        - p["g_pam_bk"] * (BK - 1.0)
        + p["g_pam_symp"] * (S - 1.0)
    )
    vascular_factor = clamp(vascular_factor, 0.35, 1.80)
    pam = PAM0 * (Vol ** p["g_pam_vol"]) * vascular_factor * D
    pam = clamp(pam, 5.0, 220.0)

    gfr, distal_delivery_ratio, r_aff, r_eff = hemodynamics(pam, A2, drugs, p)
    return {
        "PAM": pam,
        "GFR": gfr,
        "D": D,
        "AT1": eff_at1,
        "AT2": eff_at2,
        "distal_delivery_ratio": distal_delivery_ratio,
        "R_aff": r_aff,
        "R_eff": r_eff,
    }


def model_rhs(t, y, drop, drugs, p=PARAMS):
    """Sistema de EDOs do modelo didático.

    Todos os índices têm estado basal 1 sem fármacos/perturbação. O uso de
    (alvo - estado) evita deriva espontânea quando alvo=1.
    """
    R, A1, A2, Ald, ADH, BK, Na_body, K_plasma, Vol, S = [max(float(v), 1e-6) for v in y]
    out = model_outputs(t, y, drop, drugs, p)
    pam = out["PAM"]
    gfr = out["GFR"]
    eff_at1 = out["AT1"]
    eff_at2 = out["AT2"]
    gfr_ratio = gfr / GFR0

    # Feedback tubuloglomerular: maior entrega distal -> menor estímulo de renina.
    # A entrega é um proxy da TFG, não uma medida de NaCl na mácula densa.
    tgf_signal = clamp(np.exp(-p["g_ren_tgf"] * (out["distal_delivery_ratio"] - 1.0)), 0.35, 2.80)

    pressure_drive = clamp(np.exp(p["g_ren_press"] * (PAM0 - pam) / PAM0), 0.20, 3.50)
    sympathetic_drive = clamp(1.0 + p["g_ren_symp"] * (S - 1.0), 0.30, 3.00)

    # Ang II/AT1 fornece feedback negativo; bloqueio de AT1 reduz esse feedback.
    feedback_at1 = (1.0 + p["g_ren_feedback_at1"]) / (1.0 + p["g_ren_feedback_at1"] * eff_at1)
    feedback_at1 = clamp(feedback_at1, 0.30, 3.00)

    renin_drive = clamp(pressure_drive * sympathetic_drive * tgf_signal * feedback_at1, 0.10, 5.00)
    renin_inhibition = 1.0 - 0.90 * drugs["dri"]
    renin_target = clamp(renin_drive * renin_inhibition, 0.05, 5.00)

    # A cascata é normalizada: sem fármacos, R=A1=A2=1 é equilíbrio.
    ace_conversion = 1.0 - 0.85 * drugs["ace"]  # conversão residual por vias não-ACE
    a1_target = R
    a2_target = ace_conversion * A1

    # Aldosterona: estímulo AT1 + estímulo direto pelo K+ (índice relativo).
    k_ald_stimulus = clamp(np.exp(p["g_ald_k"] * (K_plasma - 1.0)), 0.30, 3.00)
    ald_target = clamp(eff_at1 * k_ald_stimulus, 0.25, 3.50)

    # ADH aumenta com hipotensão e com redução do volume relativo.
    adh_target = clamp(
        np.exp(p["g_adh_press"] * (PAM0 - pam) / PAM0 + p["g_adh_vol"] * (1.0 - Vol)),
        0.30, 3.00
    )

    # IECA reduz uma das vias de degradação de bradicinina; resta depuração residual.
    bk_clearance = 0.25 + 0.75 * (1.0 - 0.90 * drugs["ace"])
    bk_target = 1.0 / bk_clearance

    # Índice de retenção corporal de Na+: favorecido por baixa PAM/aldosterona,
    # reduzido por maior TFG relativa e expansão de volume.
    na_target = clamp(
        np.exp(
            p["g_na_press"] * (PAM0 - pam) / PAM0
            + p["g_na_ald"] * (Ald - 1.0)
            - p["g_na_gfr"] * (gfr_ratio - 1.0)
            - p["g_na_vol"] * (Vol - 1.0)
        ),
        0.60, 1.90
    )

    # Índice de K+ plasmático: maior aldosterona e maior TFG favorecem redução;
    # TFG menor favorece elevação. Não equivale a uma concentração em mmol/L.
    k_target = clamp(
        np.exp(-p["g_k_ald"] * (Ald - 1.0) - p["g_k_gfr"] * (gfr_ratio - 1.0)),
        0.60, 1.70
    )

    # Volume relativo responde lentamente aos índices de Na corporal e ADH.
    vol_target = clamp(
        np.exp(p["g_vol_na"] * (Na_body - 1.0) + p["g_vol_adh"] * (ADH - 1.0)),
        0.65, 1.80
    )

    # Barorreflexo simpático simplificado.
    symp_target = clamp(np.exp(p["g_symp_press"] * (PAM0 - pam) / PAM0), 0.40, 3.00)

    dR = p["k_ren"] * (renin_target - R)
    dA1 = p["k_a1"] * (a1_target - A1)
    dA2 = p["k_a2"] * (a2_target - A2)
    dAld = p["k_ald"] * (ald_target - Ald)
    dADH = p["k_adh"] * (adh_target - ADH)
    dBK = p["k_bk"] * (bk_target - BK)
    dNa_body = p["k_na"] * (na_target - Na_body)
    dK_plasma = p["k_k"] * (k_target - K_plasma)
    dVol = p["k_vol"] * (vol_target - Vol)
    dS = p["k_symp"] * (symp_target - S)

    return np.array([dR, dA1, dA2, dAld, dADH, dBK, dNa_body, dK_plasma, dVol, dS], dtype=float)


def find_equilibrium(drugs, p=PARAMS):
    """Encontra estado estacionário para os fármacos selecionados, sem perturbação.

    A busca é em log-estados para manter variáveis positivas. O residual é dividido
    pelas taxas características de cada variável para tornar o ajuste mais equilibrado.
    """
    rate_scale = np.array([
        p["k_ren"], p["k_a1"], p["k_a2"], p["k_ald"], p["k_adh"],
        p["k_bk"], p["k_na"], p["k_k"], p["k_vol"], p["k_symp"]
    ])

    def residual(log_y):
        y = np.exp(log_y)
        return model_rhs(0.0, y, 0.0, drugs, p) / rate_scale

    lower = np.log(np.full(10, 0.05))
    upper = np.log(np.full(10, 5.0))
    result = least_squares(
        residual,
        x0=np.zeros(10),
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=3000,
    )
    y_eq = np.exp(result.x)
    residual_max = float(np.max(np.abs(residual(result.x))))
    converged = bool(result.success and residual_max < 1e-5)
    return y_eq, converged, residual_max, result.message


def make_time_grid(t_end):
    """Cria amostragem densa no início para preservar os processos rápidos e lenta depois."""
    if t_end <= 180.0:
        early = np.linspace(-T_PRE, t_end, 2400)
        return early
    early_end = min(180.0, t_end)
    early = np.linspace(-T_PRE, early_end, 2200)
    late = np.linspace(early_end, t_end, 2800)
    return np.unique(np.concatenate([early, late]))


def simulate(drop=0.20, t_end=60.0, is_chronic=False, drugs=None):
    if drugs is None:
        drugs = {"ace": 0.0, "arb": 0.0, "dri": 0.0}
    p = PARAMS.copy()
    y_eq, eq_ok, eq_residual, eq_message = find_equilibrium(drugs, p)

    t_eval = make_time_grid(float(t_end))
    sol = solve_ivp(
        fun=lambda t, y: model_rhs(t, y, drop, drugs, p),
        t_span=(-T_PRE, float(t_end)),
        y0=y_eq,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
        max_step=0.5,
    )
    if not sol.success:
        raise RuntimeError(f"A integração falhou: {sol.message}")

    R, A1, A2, Ald, ADH, BK, Na_body, K_plasma, Vol, S = sol.y
    n = len(sol.t)
    pam = np.empty(n)
    gfr = np.empty(n)
    at1 = np.empty(n)
    at2 = np.empty(n)
    ra = np.empty(n)
    re = np.empty(n)
    disturbance = np.empty(n)

    for i, ti in enumerate(sol.t):
        out = model_outputs(ti, sol.y[:, i], drop, drugs, p)
        pam[i] = out["PAM"]
        gfr[i] = out["GFR"]
        at1[i] = out["AT1"]
        at2[i] = out["AT2"]
        ra[i] = out["R_aff"]
        re[i] = out["R_eff"]
        disturbance[i] = out["D"]

    round_list = lambda arr: np.round(np.asarray(arr, dtype=float), 6).tolist()
    result = {
        "t": round_list(sol.t),
        "PAM": round_list(pam),
        "GFR": round_list(gfr),
        "R": round_list(R),
        "A1": round_list(A1),
        "A2": round_list(A2),
        "AT1": round_list(at1),
        "AT2": round_list(at2),
        "Ald": round_list(Ald),
        "ADH": round_list(ADH),
        "BK": round_list(BK),
        "Na": round_list(Na_body),
        "K": round_list(K_plasma),
        "V": round_list(Vol),
        "S": round_list(S),
        "R_aff": round_list(ra),
        "R_eff": round_list(re),
        "D": round_list(disturbance),
        "t_end": float(t_end),
        "is_chronic": bool(is_chronic),
        "drop": float(drop),
        "drugs": drugs,
        "equilibrium_ok": eq_ok,
        "equilibrium_residual": eq_residual,
        "equilibrium_message": str(eq_message),
        "baseline": {key: float(val) for key, val in zip(STATE_NAMES, y_eq)},
    }
    return result


# =====================================================================
# 2. INTERFACE STREAMLIT
# =====================================================================
st.title("🫘 Simulador didático do SRAA")
st.caption(
    "Modelo qualitativo de hemodinâmica, cascata hormonal, feedback e efeitos farmacológicos. "
    "Não se destina a diagnóstico ou decisão terapêutica."
)

with st.sidebar:
    st.header("1. Cenário e tempo")
    teste_eq = st.checkbox("🔬 Testar equilíbrio (sem perturbação/fármacos)", value=False)
    fase = st.radio(
        "Escala de tempo",
        ["Fase aguda (minutos)", "Fase crônica (dias)"],
        disabled=teste_eq,
    )
    is_chronic = fase == "Fase crônica (dias)"

    if teste_eq:
        queda = 0
        st.info("O sistema deve permanecer estacionário, com PAM basal de 90 mmHg.")
    else:
        queda = st.slider("Queda sustentada aplicada à pressão (%)", 5, 40, 20, 1)

    if is_chronic:
        dur = st.slider("Duração (dias)", 1, 7, 3, 1)
        t_end = float(dur * 24 * 60)
    else:
        dur = st.slider("Duração (minutos)", 15, 120, 60, 15)
        t_end = float(dur)

    st.header("2. Farmacologia")
    if teste_eq:
        ace, arb, dri = 0.0, 0.0, 0.0
        st.info("Fármacos desativados no teste de equilíbrio.")
    else:
        ace = st.slider("IECA (efeito relativo)", 0.0, 1.0, 0.0, 0.1)
        arb = st.slider("BRA (efeito relativo)", 0.0, 1.0, 0.0, 0.1)
        dri = st.slider("Inibidor da renina (efeito relativo)", 0.0, 1.0, 0.0, 0.1)
        st.caption("Os efeitos escolhidos já estão ativos antes da perturbação; o basal é recalculado para esse regime.")

drugs = {"ace": float(ace), "arb": float(arb), "dri": float(dri)}
try:
    data = simulate(drop=queda / 100.0, t_end=t_end, is_chronic=is_chronic, drugs=drugs)
except Exception as exc:
    st.error(f"Não foi possível concluir a simulação: {exc}")
    st.stop()

if not data["equilibrium_ok"]:
    st.warning(
        "O algoritmo não encontrou um equilíbrio com resíduo suficientemente pequeno. "
        "Interprete esse cenário com cautela; reveja os parâmetros ou a intensidade dos fármacos. "
        f"Resíduo máximo normalizado: {data['equilibrium_residual']:.2e}."
    )

payload = json.dumps(data, separators=(",", ":"))

# =====================================================================
# 3. INTERFACE HTML/SVG/CANVAS
# =====================================================================
TEMPLATE = r"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><style>
:root{--ink:#1b2b4b;--muted:#7a8ba3;--line:#e3e9f2;}
*{box-sizing:border-box;}
body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
.wrap{padding:6px 12px 8px;position:relative;}
.top{display:flex;align-items:center;gap:16px;margin-bottom:2px;}
.pam{font-size:30px;font-weight:800;color:#c0392b;line-height:1;min-width:190px;}
.pam small{display:block;font-size:10.5px;font-weight:600;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;margin-top:3px;}
.gw{flex:1;position:relative;height:16px;}
.gauge{position:absolute;inset:0;background:#eef2f8;border-radius:8px;overflow:hidden;}
.gauge>i{position:absolute;left:0;top:0;bottom:0;width:75%;border-radius:8px;background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71);}
.clock{font-variant-numeric:tabular-nums;font-size:15px;font-weight:700;color:#4a5b78;min-width:120px;text-align:right;}
svg{display:block;width:100%;height:auto;}
.bg{stroke:#dbe4f0;stroke-width:2;}
.ttl{font-size:14px;font-weight:700;fill:#22344f;text-anchor:middle;}
.sub{font-size:10.5px;fill:#7a8ba3;text-anchor:middle;}
.val{font-size:11.5px;font-weight:700;text-anchor:middle;font-variant-numeric:tabular-nums;}
.flow{fill:none;stroke-width:4;stroke-linecap:round;stroke-dasharray:9 11;}
.charts{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap;}
.card{flex:1;min-width:280px;background:#fbfcfe;border:1px solid var(--line);border-radius:12px;padding:6px 8px 4px;}
.card h4{margin:0 0 2px 4px;font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);}
.card .hint{font-size:10px;color:#9aabc2;margin:0 0 1px 4px;}
canvas{display:block;width:100%;height:auto;}
.bar{display:flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap;}
button{border:1px solid #d3dceb;background:#fff;color:#22344f;font-weight:600;font-size:13px;padding:7px 14px;border-radius:9px;cursor:pointer;}
button:hover{background:#f2f6fd;}
button.pri{background:#1d6fb8;border-color:#1d6fb8;color:#fff;}
button.pri:hover{background:#185a95;}
.speed{display:flex;gap:6px;}
.speed button{padding:6px 10px;}
.speed button.on{background:#e8f0fb;border-color:#1d6fb8;color:#1d6fb8;}
.status{font-size:12px;color:#4a5b78;flex:1;text-align:right;font-weight:600;min-width:190px;}
#alertBox{position:absolute;top:70px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:100;pointer-events:none;max-width:350px;}
.alert{background:#ffe5e5;border-left:6px solid #d32f2f;color:#b71c1c;padding:9px 12px;border-radius:6px;font-weight:700;font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.15);display:none;}
.alert.warning{background:#fff3e0;border-color:#f57c00;color:#e65100;}
.note{font-size:11px;color:#7a8ba3;margin:7px 2px 0;line-height:1.4;}
</style></head><body><div class="wrap">
<div id="alertBox">
  <div id="alertCough" class="alert warning">⚠ Índice de bradicinina elevado neste modelo (não é previsão de tosse individual).</div>
  <div id="alertK" class="alert">⚠ Índice de K⁺ elevado (não equivale a diagnóstico de hipercalemia).</div>
</div>
<div class="top">
  <div class="pam"><span id="pamv">90</span> <span style="font-size:14px;color:#7a8ba3">mmHg</span><small>Pressão arterial média estimada</small></div>
  <div class="gw"><div class="gauge"><i id="gfill"></i></div></div>
  <div class="clock" id="clock">t = 0 min</div>
</div>
<svg id="dia" viewBox="0 0 1000 420" preserveAspectRatio="xMidYMid meet">
  <g id="flows">
    <line class="flow" id="f_ren" x1="180" y1="72" x2="240" y2="72" stroke="#7b52ab"/>
    <line class="flow" id="f_a1" x1="380" y1="72" x2="440" y2="72" stroke="#2a9d8f"/>
    <line class="flow" id="f_a2" x1="580" y1="72" x2="640" y2="72" stroke="#e76f51"/>
    <line class="flow" id="f_ald" x1="780" y1="72" x2="840" y2="72" stroke="#1d6fb8"/>
    <line class="flow" id="f_at1" x1="710" y1="110" x2="710" y2="220" stroke="#e76f51"/>
    <line class="flow" id="f_at2" x1="760" y1="110" x2="760" y2="220" stroke="#f39c12"/>
    <line class="flow" id="f_bk" x1="580" y1="150" x2="580" y2="220" stroke="#9b59b6"/>
    <line class="flow" id="f_adh" x1="840" y1="110" x2="840" y2="220" stroke="#2ecc71"/>
  </g>
  <g id="bxR"><rect class="bg" x="20" y="28" width="160" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="100" y="56">Rim</text><text class="sub" x="100" y="73">atividade da renina</text>
    <text class="val" x="100" y="98" id="vR" fill="#7b52ab">renina 1.00</text></g>
  <g id="bxF"><rect class="bg" x="240" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="310" y="56">Angiotensina I</text><text class="sub" x="310" y="73">precursor</text>
    <text class="val" x="310" y="98" id="vA1" fill="#2a9d8f">Ang I 1.00</text></g>
  <g id="bxP"><rect class="bg" x="440" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="510" y="56">Angiotensina II</text><text class="sub" x="510" y="73">formação via ECA</text>
    <text class="val" x="510" y="98" id="vA2" fill="#e76f51">Ang II 1.00</text></g>
  <g id="bxA"><rect class="bg" x="640" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="710" y="56">Adrenal</text><text class="sub" x="710" y="73">aldosterona</text>
    <text class="val" x="710" y="98" id="vAld" fill="#1d6fb8">Ald 1.00</text></g>
  <g id="bxH"><rect class="bg" x="840" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="910" y="56">Neuro-hipófise</text><text class="sub" x="910" y="73">ADH / vasopressina</text>
    <text class="val" x="910" y="98" id="vADH" fill="#2ecc71">ADH 1.00</text></g>
  <g id="bxV"><rect class="bg" x="620" y="240" width="180" height="110" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="710" y="265" font-size="13">Vasos / receptores</text>
    <rect id="lumen" x="645" y="280" width="130" height="34" rx="10" fill="#f7d9d3" stroke="#e76f51" stroke-width="2"/>
    <text class="val" x="660" y="330" id="vAT1" fill="#e76f51">AT1 1.00</text>
    <text class="val" x="760" y="330" id="vAT2" fill="#f39c12">AT2 1.00</text></g>
  <g id="bxT"><rect class="bg" x="820" y="240" width="160" height="110" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="900" y="265" font-size="13">Índices tubulares</text>
    <rect x="835" y="280" width="130" height="26" rx="13" fill="#e6eefb" stroke="#c3d5ee"/>
    <g id="dots"></g>
    <text class="val" x="865" y="330" id="vNa" fill="#1d6fb8">Na 1.00</text>
    <text class="val" x="935" y="330" id="vK" fill="#d32f2f">K 1.00</text></g>
  <g id="bxK"><rect class="bg" x="440" y="150" width="140" height="70" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="510" y="175" font-size="13">Bradicinina</text>
    <text class="val" x="510" y="200" id="vBK" fill="#9b59b6">BK 1.00</text></g>
</svg>
<div class="charts">
  <div class="card"><h4>PAM e TFG estimada</h4><div class="hint">Eixo esquerdo: mmHg · eixo direito: mL/min/1,73 m²</div><canvas id="c1" width="600" height="260"></canvas></div>
  <div class="card"><h4>Cascata hormonal</h4><div class="hint">Índices relativos (basal = 1)</div><canvas id="c2" width="600" height="260"></canvas></div>
</div>
<div class="charts">
  <div class="card"><h4>Sinalização dos receptores</h4><div class="hint">Atividade efetiva relativa (basal = 1)</div><canvas id="c3" width="600" height="260"></canvas></div>
  <div class="card"><h4>Volume relativo e aldosterona</h4><div class="hint">Índices relativos (basal = 1)</div><canvas id="c4" width="600" height="260"></canvas></div>
  <div class="card"><h4>Sódio corporal e K⁺ plasmático</h4><div class="hint">Índices relativos; não são concentrações clínicas</div><canvas id="c5" width="600" height="260"></canvas></div>
</div>
<div class="bar">
  <button id="btnPlay" class="pri">⏸ Pausar</button><button id="btnReset">↺ Reiniciar</button>
  <div class="speed"><button data-s="0.5">0.5×</button><button data-s="1" class="on">1×</button><button data-s="2">2×</button><button data-s="4">4×</button></div>
  <div class="status" id="status"></div>
</div>
<div class="note">Modelo didático ilustrativo. Índices hormonais, de receptores, sódio, potássio e volume são relativos; alertas não são diagnósticos. A TFG é uma estimativa fenomenológica, não um cálculo validado de filtração glomerular.</div>
</div>
<script>
const D = __DATA__;
const T=D.t, PAM=D.PAM, GFR=D.GFR, RR=D.R, A1=D.A1, A2=D.A2, AT1=D.AT1, AT2=D.AT2,
      ALD=D.Ald, ADH=D.ADH, BK=D.BK, NA=D.Na, K=D.K, V=D.V, SS=D.S;
const T0=T[0], T1=T[T.length-1], IS_CHRONIC=D.is_chronic;
const HAS_DIST=D.drop>0;
const DRUGS=D.drugs;
const lerp=(a,b,f)=>a+(b-a)*f;
function sample(arr,t){
  const n=T.length; if(t<=T[0])return arr[0]; if(t>=T[n-1])return arr[n-1];
  let lo=0,hi=n-1; while(hi-lo>1){const m=(lo+hi)>>1; if(T[m]<=t)lo=m; else hi=m;}
  const f=(t-T[lo])/((T[hi]-T[lo])||1); return lerp(arr[lo],arr[hi],f);
}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
function fmtTick(v,step){
  const decimals=step<0.1?2:(step<1?2:(step<10?1:0));
  return Number(v).toFixed(decimals);
}
function getRange(vals){
  let mn=Math.min(...vals), mx=Math.max(...vals), span=mx-mn;
  if(span < 1e-8){const pad=Math.max(Math.abs(mx)*0.05,0.05); mn-=pad; mx+=pad;}
  else {const pad=Math.max(span*0.12, Math.abs(mx)*0.005, 0.005); mn-=pad; mx+=pad;}
  if(mn>0 && Math.min(...vals)>0) mn=Math.max(0,mn);
  return [mn,mx];
}
function makeChart(cv,config){
  const ctx=cv.getContext('2d'), W=cv.width, H=cv.height;
  const L=52, R=config.hasRightAxis?52:12, Tp=28, B=32;
  const leftSeries=config.series.filter(s=>s.axis!=='right');
  const rightSeries=config.series.filter(s=>s.axis==='right');
  const [y1min,y1max]=getRange(leftSeries.flatMap(s=>s.data));
  let y2min=0,y2max=1;
  if(config.hasRightAxis && rightSeries.length){[y2min,y2max]=getRange(rightSeries.flatMap(s=>s.data));}
  const px=t=>L+(t-T0)/(T1-T0)*(W-L-R);
  const pyLeft=v=>Tp+(1-(v-y1min)/(y1max-y1min))*(H-Tp-B);
  const pyRight=v=>Tp+(1-(v-y2min)/(y2max-y2min))*(H-Tp-B);
  const off=document.createElement('canvas'); off.width=W;off.height=H;
  const o=off.getContext('2d');o.fillStyle='#fbfcfe';o.fillRect(0,0,W,H);
  o.font='12px -apple-system,Segoe UI,Roboto,sans-serif';
  for(let i=0;i<=4;i++){
    const f=i/4,v=y1min+(y1max-y1min)*f,y=pyLeft(v);
    o.strokeStyle='#eef2f8';o.lineWidth=1;o.beginPath();o.moveTo(L,y);o.lineTo(W-R,y);o.stroke();
    o.fillStyle=leftSeries[0].color;o.textAlign='right';o.textBaseline='middle';
    o.fillText(fmtTick(v,(y1max-y1min)/4),L-6,y);
    if(config.hasRightAxis && rightSeries.length){
      const vr=y2min+(y2max-y2min)*f;
      o.fillStyle=rightSeries[0].color;o.textAlign='left';
      o.fillText(fmtTick(vr,(y2max-y2min)/4),W-R+6,y);
    }
  }
  for(let i=0;i<=4;i++){
    const t=T0+(T1-T0)*i/4,x=px(t);
    let label;
    if(t<0) label='pré';
    else label=IS_CHRONIC?(t/1440).toFixed(t/1440<10?1:0):t.toFixed(0);
    o.fillStyle='#9aabc2';o.textAlign='center';o.textBaseline='top';o.fillText(label,x,H-B+7);
  }
  if(HAS_DIST){o.strokeStyle='#e8b04b';o.lineWidth=2;o.setLineDash([5,5]);o.beginPath();o.moveTo(px(0),Tp);o.lineTo(px(0),H-B);o.stroke();o.setLineDash([]);}
  config.series.forEach(s=>{
    const py=s.axis==='right'?pyRight:pyLeft;
    o.strokeStyle=s.color;o.lineWidth=2;o.beginPath();
    for(let i=0;i<T.length;i++){const x=px(T[i]),y=py(s.data[i]);i?o.lineTo(x,y):o.moveTo(x,y);}
    o.stroke();
  });
  if(config.legend){
    o.font='11px -apple-system,Segoe UI,Roboto,sans-serif';o.textAlign='left';o.textBaseline='middle';
    let lx=L+4, ly=12;
    config.series.forEach(s=>{
      const width=15+o.measureText(s.name).width+10;
      if(lx+width>W-R){lx=L+4;ly+=13;}
      o.fillStyle=s.color;o.fillRect(lx,ly,12,3);o.fillStyle='#5a6b85';o.fillText(s.name,lx+16,ly+2);
      lx+=width;
    });
  }
  return {ctx,off,px,pyLeft,pyRight,W,H,L,R,Tp,B,config};
}
const ch1=makeChart(document.getElementById('c1'),{hasRightAxis:true,legend:true,series:[
  {name:'PAM',data:PAM,color:'#c0392b',axis:'left'}, {name:'TFG estimada',data:GFR,color:'#2980b9',axis:'right'}]});
const ch2=makeChart(document.getElementById('c2'),{hasRightAxis:false,legend:true,series:[
  {name:'Renina',data:RR,color:'#7b52ab',axis:'left'}, {name:'Ang I',data:A1,color:'#2a9d8f',axis:'left'}, {name:'Ang II',data:A2,color:'#e76f51',axis:'left'}]});
const ch3=makeChart(document.getElementById('c3'),{hasRightAxis:false,legend:true,series:[
  {name:'AT1',data:AT1,color:'#d35400',axis:'left'}, {name:'AT2',data:AT2,color:'#f39c12',axis:'left'}]});
const ch4=makeChart(document.getElementById('c4'),{hasRightAxis:false,legend:true,series:[
  {name:'Aldosterona',data:ALD,color:'#1d6fb8',axis:'left'}, {name:'Volume',data:V,color:'#8e44ad',axis:'left'}]});
const ch5=makeChart(document.getElementById('c5'),{hasRightAxis:false,legend:true,series:[
  {name:'Na corporal',data:NA,color:'#1d6fb8',axis:'left'}, {name:'K plasmático',data:K,color:'#d32f2f',axis:'left'}]});
function drawChart(ch,t){
  const c=ch.ctx;c.clearRect(0,0,ch.W,ch.H);c.drawImage(ch.off,0,0);
  const x=ch.px(t);c.strokeStyle='#22344f';c.lineWidth=1.2;c.beginPath();c.moveTo(x,ch.Tp);c.lineTo(x,ch.H-ch.B);c.stroke();
  ch.config.series.forEach(s=>{const py=s.axis==='right'?ch.pyRight:ch.pyLeft;const y=py(sample(s.data,t));
    c.beginPath();c.arc(x,y,4,0,Math.PI*2);c.fillStyle=s.color;c.fill();c.strokeStyle='#fff';c.lineWidth=1.5;c.stroke();});
}
function deviationScale(arr,v){
  const maxDev=Math.max(...arr.map(x=>Math.abs(x-1)));
  if(maxDev<0.02)return 0.18;
  return clamp(0.18+0.82*Math.abs(v-1)/maxDev,0.18,1);
}
function h2r(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function mix(a,b,f){const A=h2r(a),B=h2r(b);return 'rgb('+Math.round(lerp(A[0],B[0],f))+','+Math.round(lerp(A[1],B[1],f))+','+Math.round(lerp(A[2],B[2],f))+')';}
function setBox(id,n,pale,sat){const rect=document.querySelector('#'+id+' .bg');if(rect)rect.setAttribute('fill',mix(pale,sat,n));}
const dg=document.getElementById('dots'),dots=[];
for(let i=0;i<6;i++){const el=document.createElementNS('http://www.w3.org/2000/svg','circle');el.setAttribute('r',4);el.setAttribute('cy',293);el.setAttribute('fill','#1d6fb8');dg.appendChild(el);dots.push({el,x:835+i*22});}
const off={f_ren:0,f_a1:0,f_a2:0,f_ald:0,f_at1:0,f_at2:0,f_bk:0,f_adh:0};
let simT=T0,playing=true,speed=1,lastTs=null,coughShown=false,kShown=false;
function render(dt){
  const pam=sample(PAM,simT),r=sample(RR,simT),a1=sample(A1,simT),a2=sample(A2,simT),at1=sample(AT1,simT),at2=sample(AT2,simT),
        al=sample(ALD,simT),adh=sample(ADH,simT),bk=sample(BK,simT),na=sample(NA,simT),k=sample(K,simT),vol=sample(V,simT);
  document.getElementById('pamv').textContent=pam.toFixed(1);
  document.getElementById('gfill').style.width=clamp(pam/120*100,0,100)+'%';
  let timeStr=IS_CHRONIC?(Math.max(simT,0)/1440).toFixed(2)+' dias':Math.max(simT,0).toFixed(1)+' min';
  if(simT<0)timeStr='pré-perturbação';
  document.getElementById('clock').textContent='t = '+timeStr;
  setBox('bxR',deviationScale(RR,r),'#ece4f7','#7b52ab');
  setBox('bxF',deviationScale(A1,a1),'#dff3f0','#2a9d8f');
  setBox('bxP',deviationScale(A2,a2),'#fbe3dc','#e76f51');
  setBox('bxA',deviationScale(ALD,al),'#dce9f8','#1d6fb8');
  setBox('bxH',deviationScale(ADH,adh),'#e0f5e6','#2ecc71');
  setBox('bxK',deviationScale(BK,bk),'#f4e6f7','#9b59b6');
  setBox('bxV',Math.max(deviationScale(AT1,at1),deviationScale(AT2,at2)),'#fbe3dc','#e76f51');
  setBox('bxT',deviationScale(NA,na),'#e6eefb','#1d6fb8');
  document.getElementById('vR').textContent='renina '+r.toFixed(2);
  document.getElementById('vA1').textContent='Ang I '+a1.toFixed(2);
  document.getElementById('vA2').textContent='Ang II '+a2.toFixed(2);
  document.getElementById('vAT1').textContent='AT1 '+at1.toFixed(2);
  document.getElementById('vAT2').textContent='AT2 '+at2.toFixed(2);
  document.getElementById('vAld').textContent='Ald '+al.toFixed(2);
  document.getElementById('vADH').textContent='ADH '+adh.toFixed(2);
  document.getElementById('vBK').textContent='BK '+bk.toFixed(2);
  document.getElementById('vNa').textContent='Na '+na.toFixed(2);
  document.getElementById('vK').textContent='K '+k.toFixed(2);
  const arrows=[['f_ren',r,RR],['f_a1',a1,A1],['f_a2',a2,A2],['f_ald',al,ALD],['f_at1',at1,AT1],['f_at2',at2,AT2],['f_bk',bk,BK],['f_adh',adh,ADH]];
  arrows.forEach(([id,v,arr])=>{const n=deviationScale(arr,v);off[id]-=(15+100*n)*dt;const el=document.getElementById(id);el.setAttribute('stroke-dashoffset',off[id]);el.setAttribute('opacity',0.2+0.8*n);});
  // Lúmen: AT1 contrai; AT2 dilata. Altura basal é exatamente 34 px.
  const h=clamp(34-8*(at1-1)+4*(at2-1),12,50);
  const lum=document.getElementById('lumen');lum.setAttribute('height',h);lum.setAttribute('y',280+(34-h)/2);
  dots.forEach(d=>{d.x+=(8+12*clamp(na,0.3,2.0))*dt;if(d.x>965)d.x=835;d.el.setAttribute('cx',d.x);d.el.setAttribute('opacity',clamp(0.5+0.25*(na-1),0.2,1));});
  drawChart(ch1,simT);drawChart(ch2,simT);drawChart(ch3,simT);drawChart(ch4,simT);drawChart(ch5,simT);
  // Avisos deliberadamente rotulados como índices didáticos, não diagnósticos.
  if(DRUGS.ace>0.05 && bk>1.50 && !coughShown && simT>0){document.getElementById('alertCough').style.display='block';coughShown=true;}
  if(k>1.15 && !kShown && simT>0){document.getElementById('alertK').style.display='block';kShown=true;}
  const status=document.getElementById('status');
  if(simT<0)status.textContent='Regime basal antes da perturbação';
  else if(k>1.15)status.textContent='Índice relativo de K⁺ elevado';
  else if(DRUGS.arb>0.05)status.textContent='BRA: sinalização AT1 reduzida no modelo';
  else if(DRUGS.ace>0.05)status.textContent='IECA: conversão Ang I → Ang II reduzida';
  else if(pam<85)status.textContent='PAM baixa no modelo; compensações em curso';
  else if(!HAS_DIST)status.textContent='Regime estacionário (sem perturbação)';
  else status.textContent='Resposta do modelo à perturbação';
}
function loop(ts){
  if(lastTs===null)lastTs=ts;
  let dt=(ts-lastTs)/1000;lastTs=ts;dt=Math.min(dt,0.05);
  if(playing){simT+=dt*(IS_CHRONIC?1440.0/15.0:1.0)*speed;if(simT>=T1){simT=T1;playing=false;btnPlay.textContent='▶ Reproduzir';}}
  render(playing?dt*speed:0);requestAnimationFrame(loop);
}
const btnPlay=document.getElementById('btnPlay');
btnPlay.onclick=()=>{if(!playing&&simT>=T1)simT=T0;playing=!playing;btnPlay.textContent=playing?'⏸ Pausar':'▶ Reproduzir';};
document.getElementById('btnReset').onclick=()=>{simT=T0;playing=true;btnPlay.textContent='⏸ Pausar';for(const key in off)off[key]=0;coughShown=false;kShown=false;document.getElementById('alertCough').style.display='none';document.getElementById('alertK').style.display='none';};
document.querySelectorAll('.speed button').forEach(b=>{b.onclick=()=>{document.querySelectorAll('.speed button').forEach(x=>x.classList.remove('on'));b.classList.add('on');speed=parseFloat(b.dataset.s);};});
requestAnimationFrame(loop);
</script></body></html>
"""

html = TEMPLATE.replace("__DATA__", payload)
components.html(html, height=1470, scrolling=True)

# =====================================================================
# 4. DIAGNÓSTICOS MATEMÁTICOS E LIMITAÇÕES
# =====================================================================
with st.expander("Verificar o equilíbrio numérico", expanded=False):
    st.write(f"Convergência do cálculo do equilíbrio: **{data['equilibrium_ok']}**")
    st.write(f"Maior resíduo normalizado: **{data['equilibrium_residual']:.3e}**")
    st.write("Estado estacionário usado antes da perturbação, já considerando os fármacos selecionados:")
    baseline_rows = [{"Variável": name, "Índice basal calculado": round(value, 4)} for name, value in data["baseline"].items()]
    st.dataframe(baseline_rows, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown(
    "**Como interpretar o simulador**\n\n"
    "- O teste de equilíbrio deve manter PAM em 90 mmHg e os índices em 1, sem deriva.\n"
    "- Os índices de renina, angiotensinas, receptores, aldosterona, ADH, bradicinina, sódio, potássio e volume são relativos; não representam concentrações clínicas.\n"
    "- Na fase crônica, os balanços de sódio, potássio e volume usam constantes de tempo lentas, definidas para fins didáticos.\n"
    "- A TFG é uma estimativa fenomenológica simples. O modelo não inclui a mecânica completa do néfron, o transporte segmentar, a osmolaridade plasmática, a farmacocinética nem variabilidade individual.\n"
    "- Os avisos são indicadores pedagógicos e não devem ser usados para diagnosticar ou orientar tratamentos."
)
