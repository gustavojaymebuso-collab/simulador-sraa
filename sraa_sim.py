# -*- coding: utf-8 -*-
"""
Simulador didático do Sistema Renina-Angiotensina-Aldosterona (SRAA) — v11: expressão de canais e transporte celular

Execução:
    streamlit run sraa_sim_v11.py

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
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

st.set_page_config(page_title="imulador didático do SRAA", layout="wide")

# =====================================================================
# 1. PARÂMETROS E CONVENÇÕES
# =====================================================================
PAM0 = 90.0
GFR0 = 112.0  # referência didática da TFG estimada, em mL/min/1,73 m²
T_PRE = 2.0   # minutos exibidos antes do início da perturbação

# As taxas são por minuto. Para variáveis dinâmicas de primeira ordem,
# k = ln(2)/t½. As escalas são ilustrativas; não representam uma calibração clínica.
PARAMS = {
    # Hormônios / mediadores de resposta rápida
    "k_ren": np.log(2) / 15.0,       # escala de resposta da atividade de renina
    "k_a1": np.log(2) / 3.0,         # Ang I
    "k_a2": np.log(2) / 1.5,         # Ang II (escala rápida ilustrativa)
    "k_ald": np.log(2) / 35.0,       # aldosterona: resposta mais lenta
    # Expressão de canais induzida por aldosterona; atraso didático de 2 h.
    "k_channel_expression": 1.0 / 120.0,
    "k_adh": np.log(2) / 7.0,        # ADH, escala ilustrativa
    "k_bk": np.log(2) / 1.5,         # bradicinina, escala ilustrativa
    "k_symp": np.log(2) / 0.5,       # resposta simpática simplificada

    # Ocupação normalizada de receptores
    "Kd_AT1": 1.0,
    "Kd_AT2": 2.0,
    # Eficácia máxima didática do BRA: 19/21 corresponde a ~90,5% de
    # redução da ocupação relativa quando Ang II = Kd = 1.
    "arb_max_block": 19.0 / 21.0,

    # Intensidades máximas didáticas dos fármacos (não são potências clínicas).
    "max_dri_renin_inhibition": 0.90,
    "max_ace_conversion_inhibition": 0.85,
    "max_ace_bk_degradation_inhibition": 0.90,
    "bk_residual_clearance_fraction": 0.25,
    "bk_ace_sensitive_clearance_fraction": 0.75,

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
    "g_pam_at1": 0.12,
    "g_pam_at2": 0.08,
    "g_pam_bk": 0.06,
    "g_pam_symp": 0.08,
    "g_ra_pressure": 0.75,
    "g_re_at1": 0.65,

    # Balanços lentos: tempos característicos em horas, convertidos para minutos
    "k_na": 1.0 / (12.0 * 60.0),       # ~12 h
    "k_k": 1.0 / (5.0 * 60.0),         # ~5 h
    "k_vol": 1.0 / (18.0 * 60.0),      # ~18 h
    "g_na_press": 0.80,
    "g_na_ald": 0.25,
    "g_na_gfr": 0.30,
    "g_na_vol": 0.80,
    "g_k_ald": 0.25,
    "g_k_gfr": 0.15,
    "g_vol_na": 0.45,
    "g_vol_adh": 0.12,
}

STATE_NAMES = ["R", "A1", "A2", "Ald", "E_channels", "ADH", "BK", "Na_body", "K_plasma", "Vol", "S"]
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}


def clamp(value, low, high):
    """Limita um valor escalar ao intervalo especificado."""
    return float(np.clip(value, low, high))


def receptor_effects(a2, drugs, p=PARAMS):
    """Atividades efetivas relativas de AT1 e AT2.

    O slider do BRA representa intensidade de bloqueio funcional. Para Ang II = 1
    e Kd_AT1 = 1, cada incremento do slider produz uma redução proporcional da
    ocupação relativa no ponto de referência. Isso é uma simplificação didática
    de antagonismo competitivo, não uma curva clínica de dose-resposta.
    """
    a2 = max(float(a2), 1e-8)
    arb_level = clamp(drugs.get("arb", 0.0), 0.0, 1.0)
    block_fraction = p["arb_max_block"] * arb_level
    # Derivado para que occ_AT1/occ_AT1_basal = 1 - block_fraction em A2=Kd=1.
    kd_multiplier = (1.0 + block_fraction) / (1.0 - block_fraction)
    kd_at1_eff = p["Kd_AT1"] * kd_multiplier

    occ_at1 = a2 / (kd_at1_eff + a2)
    occ_at2 = a2 / (p["Kd_AT2"] + a2)

    occ_at1_baseline = 1.0 / (p["Kd_AT1"] + 1.0)
    occ_at2_baseline = 1.0 / (p["Kd_AT2"] + 1.0)
    eff_at1 = occ_at1 / occ_at1_baseline
    eff_at2 = occ_at2 / occ_at2_baseline
    return float(eff_at1), float(eff_at2)


def hemodynamics(pam, a2, drugs, p=PARAMS):
    """Estimativa fenomenológica da TFG, com suporte eferente mais relevante sob baixa perfusão.

    Em normoperfusão, a fórmula é calibrada para GFR0. Quando a PAM cai, a resistência
    aferente diminui (resposta miogênica simplificada). O efeito AT1 sobre a resistência
    eferente ganha peso conforme aumenta a hipoperfusão, permitindo ilustrar por que o
    bloqueio do SRAA pode reduzir a TFG em situações vulneráveis. Não é um modelo completo
    de néfron nem uma estimativa clínica validada.
    """
    eff_at1, _ = receptor_effects(a2, drugs, p)
    pressure_ratio = max(float(pam), 1.0) / PAM0

    # Menor pressão -> menor resistência aferente (dilatação compensatória simplificada).
    r_aff = clamp(1.0 + p["g_ra_pressure"] * (pressure_ratio - 1.0), 0.60, 1.50)

    # O suporte eferente mediado por AT1 importa mais quando a perfusão está reduzida.
    hypoperfusion = clamp((1.0 - pressure_ratio) / 0.25, 0.0, 1.0)
    r_eff = clamp(
        1.0 + p["g_re_at1"] * hypoperfusion * (eff_at1 - 1.0),
        0.45,
        1.60,
    )

    # Basal sem fármacos: PAM=90, Ra=Re=1 -> TFG=GFR0.
    gfr = GFR0 * pressure_ratio * (2.0 * r_eff / (r_aff + r_eff))
    gfr = clamp(gfr, 5.0, 220.0)

    # Proxy de entrega tubular para TGF. Não equivale a NaCl medido na mácula densa.
    distal_delivery_ratio = clamp(gfr / GFR0, 0.20, 2.50)
    return gfr, distal_delivery_ratio, r_aff, r_eff


def model_outputs(t, y, drugs, p=PARAMS):
    """Calcula PAM, TFG, receptores e variáveis algébricas associadas."""
    R, A1, A2, Ald, E_channels, ADH, BK, Na_body, K_plasma, Vol, S = [max(float(v), 1e-6) for v in y]
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
    pam = PAM0 * (Vol ** p["g_pam_vol"]) * vascular_factor
    pam = clamp(pam, 5.0, 220.0)

    gfr, distal_delivery_ratio, r_aff, r_eff = hemodynamics(pam, A2, drugs, p)
    return {
        "PAM": pam,
        "GFR": gfr,
        "AT1": eff_at1,
        "AT2": eff_at2,
        "distal_delivery_ratio": distal_delivery_ratio,
        "R_aff": r_aff,
        "R_eff": r_eff,
    }


def model_rhs(t, y, drugs, p=PARAMS):
    """Sistema de EDOs do modelo didático.

    Todos os índices têm estado basal 1 sem fármacos/perturbação. O uso de
    (alvo - estado) evita deriva espontânea quando alvo=1.
    """
    R, A1, A2, Ald, E_channels, ADH, BK, Na_body, K_plasma, Vol, S = [max(float(v), 1e-6) for v in y]
    out = model_outputs(t, y, drugs, p)
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
    renin_inhibition = 1.0 - p["max_dri_renin_inhibition"] * drugs["dri"]
    renin_target = clamp(renin_drive * renin_inhibition, 0.05, 5.00)

    # A cascata é normalizada: sem fármacos, R=A1=A2=1 é equilíbrio.
    ace_conversion = 1.0 - p["max_ace_conversion_inhibition"] * drugs["ace"]  # conversão residual por vias não-ACE
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
    bk_clearance = p["bk_residual_clearance_fraction"] + p["bk_ace_sensitive_clearance_fraction"] * (
        1.0 - p["max_ace_bk_degradation_inhibition"] * drugs["ace"]
    )
    bk_target = 1.0 / bk_clearance

    # Índice de retenção corporal de Na+: favorecido por baixa PAM/aldosterona,
    # reduzido por maior TFG relativa e expansão de volume.
    na_target = clamp(
        np.exp(
            p["g_na_press"] * (PAM0 - pam) / PAM0
            + p["g_na_ald"] * (E_channels - 1.0)
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
    # Estado dinâmico de expressão/abundância funcional dos canais epiteliais.
    # E acompanha Ald com atraso; este é um proxy didático, não expressão molecular medida.
    dE_channels = p["k_channel_expression"] * (Ald - E_channels)
    dADH = p["k_adh"] * (adh_target - ADH)
    dBK = p["k_bk"] * (bk_target - BK)
    dNa_body = p["k_na"] * (na_target - Na_body)
    dK_plasma = p["k_k"] * (k_target - K_plasma)
    dVol = p["k_vol"] * (vol_target - Vol)
    dS = p["k_symp"] * (symp_target - S)

    return np.array([dR, dA1, dA2, dAld, dE_channels, dADH, dBK, dNa_body, dK_plasma, dVol, dS], dtype=float)


def find_equilibrium(drugs, p=PARAMS):
    """Encontra estado estacionário para os fármacos selecionados, sem perturbação.

    A busca é em log-estados para manter variáveis positivas. O residual é dividido
    pelas taxas características de cada variável para tornar o ajuste mais equilibrado.
    """
    rate_scale = np.array([
        p["k_ren"], p["k_a1"], p["k_a2"], p["k_ald"], p["k_channel_expression"],
        p["k_adh"], p["k_bk"], p["k_na"], p["k_k"], p["k_vol"], p["k_symp"]
    ])

    def residual(log_y):
        y = np.exp(log_y)
        return model_rhs(0.0, y, drugs, p) / rate_scale

    lower = np.log(np.full(11, 0.05))
    upper = np.log(np.full(11, 5.0))
    result = least_squares(
        residual,
        x0=np.zeros(11),
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
    """Amostragem pós-perturbação em minutos; prioriza a resposta rápida no começo."""
    t_end = float(t_end)
    if t_end <= 180.0:
        return np.linspace(0.0, t_end, 2400)
    early_end = min(180.0, t_end)
    early = np.linspace(0.0, early_end, 2200)
    late = np.linspace(early_end, t_end, 2800)
    return np.unique(np.concatenate([early, late]))


@st.cache_data(show_spinner=False, max_entries=32)
def simulate(drop=0.20, t_end=60.0, is_chronic=False, drugs=None, params=None):
    """Simula um desafio hemodinâmico reversível por alteração inicial do volume relativo.

    O cenário farmacológico já está presente antes de t=0; por isso, o equilíbrio
    é recalculado com os fármacos ativos. No instante t=0, reduz-se Vol o suficiente
    para produzir a queda percentual inicial escolhida na PAM. Depois disso, não há
    multiplicador externo permanente de pressão: os mecanismos de feedback podem
    compensar a perturbação. Isso é um desafio didático, não simulação de hemorragia real.
    """
    # params é argumento explícito da função cacheada: alterações nos
    # parâmetros invalidam corretamente a chave de cache.
    p = dict(PARAMS if params is None else params)
    drop = float(np.clip(drop, 0.0, 0.40))
    if drugs is None:
        drugs = {"ace": 0.0, "arb": 0.0, "dri": 0.0}
    drugs = {
        "ace": float(np.clip(drugs.get("ace", 0.0), 0.0, 1.0)),
        "arb": float(np.clip(drugs.get("arb", 0.0), 0.0, 1.0)),
        "dri": float(np.clip(drugs.get("dri", 0.0), 0.0, 1.0)),
    }

    y_eq, eq_ok, eq_residual, eq_message = find_equilibrium(drugs, p)

    # Período pré-evento estacionário, para visualizar um verdadeiro basal.
    t_pre = np.linspace(-T_PRE, -1e-4, 80)
    y_pre = np.repeat(y_eq[:, None], len(t_pre), axis=1)

    # O choque é imposto uma única vez ao estado de volume efetivo. Como PAM ~ Vol^g,
    # essa transformação produz a queda inicial solicitada mantendo os outros estados.
    y0_post = y_eq.copy()
    if drop > 0.0:
        i_vol = STATE_INDEX["Vol"]
        y0_post[i_vol] *= (1.0 - drop) ** (1.0 / p["g_pam_vol"])

    t_post = make_time_grid(float(t_end))
    rhs = lambda t, y: model_rhs(t, y, drugs, p)
    early_end = min(30.0, float(t_end))

    # Integração em dois segmentos:
    # 1) 0–30 min: passo máximo curto para resolver os transientes rápidos;
    # 2) após 30 min: LSODA escolhe o passo adaptativamente, sem max_step fixo.
    # t_eval controla apenas os instantes de saída; não força o passo interno do solver.
    if float(t_end) <= 30.0:
        sol_early = solve_ivp(
            fun=rhs,
            t_span=(0.0, float(t_end)),
            y0=y0_post,
            t_eval=t_post,
            method="LSODA",
            rtol=1e-7,
            atol=1e-9,
            max_step=0.25,
        )
        if not sol_early.success:
            raise RuntimeError(f"A integração inicial falhou: {sol_early.message}")
        t_post_sim = sol_early.t
        y_post_sim = sol_early.y
    else:
        t_eval_early = np.unique(
            np.concatenate(([0.0], t_post[(t_post > 0.0) & (t_post < early_end)], [early_end]))
        )
        sol_early = solve_ivp(
            fun=rhs,
            t_span=(0.0, early_end),
            y0=y0_post,
            t_eval=t_eval_early,
            method="LSODA",
            rtol=1e-7,
            atol=1e-9,
            max_step=0.25,
        )
        if not sol_early.success:
            raise RuntimeError(f"A integração inicial falhou: {sol_early.message}")

        t_eval_late = np.unique(
            np.concatenate(([early_end], t_post[t_post > early_end]))
        )
        sol_late = solve_ivp(
            fun=rhs,
            t_span=(early_end, float(t_end)),
            y0=sol_early.y[:, -1],
            t_eval=t_eval_late,
            method="LSODA",
            rtol=1e-7,
            atol=1e-9,
            # Intencionalmente sem max_step nesta etapa: passo adaptativo livre.
        )
        if not sol_late.success:
            raise RuntimeError(f"A integração tardia falhou: {sol_late.message}")

        # O instante de fronteira (early_end) está nos dois segmentos; inclui-se uma vez.
        t_post_sim = np.concatenate([sol_early.t, sol_late.t[1:]])
        y_post_sim = np.concatenate([sol_early.y, sol_late.y[:, 1:]], axis=1)

    t_all = np.concatenate([t_pre, t_post_sim])
    y_all = np.concatenate([y_pre, y_post_sim], axis=1)
    R, A1, A2, Ald, E_channels, ADH, BK, Na_body, K_plasma, Vol, S = y_all
    n = len(t_all)
    pam = np.empty(n)
    gfr = np.empty(n)
    at1 = np.empty(n)
    at2 = np.empty(n)
    ra = np.empty(n)
    re = np.empty(n)

    for i, ti in enumerate(t_all):
        out = model_outputs(ti, y_all[:, i], drugs, p)
        pam[i] = out["PAM"]
        gfr[i] = out["GFR"]
        at1[i] = out["AT1"]
        at2[i] = out["AT2"]
        ra[i] = out["R_aff"]
        re[i] = out["R_eff"]

    baseline_outputs = model_outputs(0.0, y_eq, drugs, p)
    round_list = lambda arr: np.round(np.asarray(arr, dtype=float), 6).tolist()
    result = {
        "t": round_list(t_all),
        "PAM": round_list(pam),
        "GFR": round_list(gfr),
        "R": round_list(R),
        "A1": round_list(A1),
        "A2": round_list(A2),
        "AT1": round_list(at1),
        "AT2": round_list(at2),
        "Ald": round_list(Ald),
        "E": round_list(E_channels),
        "ADH": round_list(ADH),
        "BK": round_list(BK),
        "Na": round_list(Na_body),
        "K": round_list(K_plasma),
        "V": round_list(Vol),
        "S": round_list(S),
        "R_aff": round_list(ra),
        "R_eff": round_list(re),
        "t_end": float(t_end),
        "is_chronic": bool(is_chronic),
        "drop": drop,
        "drugs": drugs,
        "equilibrium_ok": eq_ok,
        "equilibrium_residual": eq_residual,
        "equilibrium_message": str(eq_message),
        "baseline": {key: float(val) for key, val in zip(STATE_NAMES, y_eq)},
        "baseline_outputs": {key: float(val) for key, val in baseline_outputs.items()},
    }
    return result


# =====================================================================
# 2. INTERFACE STREAMLIT
# =====================================================================
st.title("Simulador didático do SRAA")
st.caption(
    "Modelo qualitativo de hemodinâmica, cascata hormonal, feedback e efeitos farmacológicos. "
    "Não se destina a diagnóstico ou decisão terapêutica."
)

PRESETS = {
    "Sem fármaco": {"ace": 0.0, "arb": 0.0, "dri": 0.0},
    "IECA máximo": {"ace": 1.0, "arb": 0.0, "dri": 0.0},
    "BRA máximo": {"ace": 0.0, "arb": 1.0, "dri": 0.0},
    "Inibidor da renina máximo": {"ace": 0.0, "arb": 0.0, "dri": 1.0},
    "IECA + BRA máximos": {"ace": 1.0, "arb": 1.0, "dri": 0.0},
    "Triplo bloqueio máximo": {"ace": 1.0, "arb": 1.0, "dri": 1.0},
    "Personalizado": None,
}


COMPARISON_QUICK_PRESETS = {
    "Personalizado (manter os controles)": None,
    "IECA máximo × BRA máximo": (
        {"ace": 1.0, "arb": 0.0, "dri": 0.0},
        {"ace": 0.0, "arb": 1.0, "dri": 0.0},
        "BRA máximo",
    ),
    "Sem fármaco × IECA + BRA máximos": (
        {"ace": 0.0, "arb": 0.0, "dri": 0.0},
        {"ace": 1.0, "arb": 1.0, "dri": 0.0},
        "IECA + BRA máximos",
    ),
}

def apply_comparison_quick_preset():
    """Preenche A/B antes da criação dos sliders; os modelos não são modificados."""
    selected = st.session_state.get("comparison_quick_preset", "Personalizado (manter os controles)")
    pair = COMPARISON_QUICK_PRESETS.get(selected)
    if pair is None:
        return
    regime_a, regime_b, preset_b = pair
    for name in ("ace", "arb", "dri"):
        st.session_state[f"drug_a_{name}"] = float(regime_a[name])
        st.session_state[f"drug_b_{name}"] = float(regime_b[name])
    st.session_state["preset_b"] = preset_b


def regime_label(regime):
    """Cria um rótulo compacto para a legenda da comparação."""
    labels = []
    for key, name in (("ace", "IECA"), ("arb", "BRA"), ("dri", "Inibidor da renina")):
        value = float(regime.get(key, 0.0))
        if value > 0.0001:
            labels.append(f"{name} {value:.1f}".replace(".", ","))
    return "Sem fármaco" if not labels else " + ".join(labels)


def basal_value(sim_data, key):
    """Obtém o basal correspondente a cada série, específico do regime."""
    state_map = {
        "R": "R", "A1": "A1", "A2": "A2", "Ald": "Ald", "ADH": "ADH",
        "BK": "BK", "Na": "Na_body", "K": "K_plasma", "V": "Vol",
    }
    if key in ("PAM", "GFR", "AT1", "AT2"):
        return float(sim_data["baseline_outputs"][key])
    return float(sim_data["baseline"][state_map[key]])


def fmt_value(value, key):
    if key in ("PAM", "GFR"):
        return f"{float(value):.1f}"
    return f"{float(value):.2f}"


def deviation_pct(value, baseline):
    if abs(baseline) < 1e-12:
        return 0.0
    return 100.0 * (float(value) / float(baseline) - 1.0)


def sequence_summary(sim_data, key):
    """Pré → ponto de maior desvio percentual absoluto → final."""
    times = sim_data["t"]
    vals = sim_data[key]
    baseline = basal_value(sim_data, key)
    pre_indices = [i for i, t in enumerate(times) if t < 0]
    post_indices = [i for i, t in enumerate(times) if t >= 0]
    pre_idx = pre_indices[-1] if pre_indices else 0
    final_idx = len(vals) - 1
    if post_indices:
        extreme_idx = max(
            post_indices,
            key=lambda i: abs(deviation_pct(vals[i], baseline)),
        )
    else:
        extreme_idx = pre_idx
    pre = fmt_value(vals[pre_idx], key)
    extreme = fmt_value(vals[extreme_idx], key)
    final = fmt_value(vals[final_idx], key)
    ext_pct = deviation_pct(vals[extreme_idx], baseline)
    fin_pct = deviation_pct(vals[final_idx], baseline)
    return (
        f"{pre} → {extreme} ({ext_pct:+.1f}%) → "
        f"{final} ({fin_pct:+.1f}%)"
    )


def final_delta_text(sim_a, sim_b, key):
    delta = float(sim_b[key][-1]) - float(sim_a[key][-1])
    if key in ("PAM", "GFR"):
        return f"{delta:+.1f}"
    return f"{delta:+.2f}"


def pam_recovery_time(sim_data, fraction=0.95):
    """Primeiro instante em que a PAM atinge fraction do seu próprio basal."""
    baseline = float(sim_data["baseline_outputs"]["PAM"])
    threshold = fraction * baseline
    times = sim_data["t"]
    values = sim_data["PAM"]
    for i, time_value in enumerate(times):
        if time_value >= 0 and values[i] >= threshold:
            return float(time_value)
    return None


def format_recovery(time_value, chronic, horizon_minutes):
    if time_value is None:
        if chronic:
            days = horizon_minutes / 1440.0
            return f"> {days:.0f} dia(s)"
        return f"> {horizon_minutes:.0f} min"
    if time_value <= 1e-9:
        return "0 min (já ≥95%)"
    if chronic:
        if time_value < 60:
            return f"{time_value:.1f} min"
        if time_value < 1440:
            return f"{time_value / 60:.2f} h"
        return f"{time_value / 1440:.2f} dias"
    return f"{time_value:.1f} min"


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
        st.info("Sem perturbação e sem fármacos, o sistema deve permanecer estacionário em PAM = 90 mmHg e índices = 1.")
    else:
        queda = st.slider("Queda inicial da PAM equivalente (%)", 5, 40, 20, 1)
        st.caption("Aplicada como uma redução inicial do volume relativo; não é um multiplicador permanente da pressão.")

    if is_chronic:
        dur = st.slider("Duração (dias)", 1, 7, 3, 1)
        t_end = float(dur * 24 * 60)
    else:
        dur = st.slider("Duração (minutos)", 15, 120, 60, 15)
        t_end = float(dur)

    compare_enabled = st.checkbox("📈 Comparar dois regimes lado a lado", value=False, disabled=teste_eq)
    if compare_enabled and not teste_eq:
        st.selectbox(
            "Comparação pronta",
            list(COMPARISON_QUICK_PRESETS.keys()),
            key="comparison_quick_preset",
            on_change=apply_comparison_quick_preset,
        )

    st.header("2. Farmacologia — Regime A")
    if teste_eq:
        ace, arb, dri = 0.0, 0.0, 0.0
        st.info("Fármacos desativados no teste de equilíbrio.")
    else:
        for key, default in (("drug_a_ace", 0.0), ("drug_a_arb", 0.0), ("drug_a_dri", 0.0)):
            st.session_state.setdefault(key, default)
        ace = st.slider("IECA — intensidade sobre a conversão via ECA", 0.0, 1.0, step=0.1, key="drug_a_ace")
        arb = st.slider("BRA — intensidade do bloqueio funcional de AT1", 0.0, 1.0, step=0.1, key="drug_a_arb")
        dri = st.slider("Inibidor da renina — intensidade de inibição", 0.0, 1.0, step=0.1, key="drug_a_dri")
        st.caption("São intensidades didáticas, não doses. Os fármacos já estão ativos antes da perturbação; o equilíbrio é recalculado para cada regime.")

    drugs = {"ace": float(ace), "arb": float(arb), "dri": float(dri)}
    drugs_b = None
    label_b = ""
    diagram_regime = "A"

    if compare_enabled and not teste_eq:
        st.header("3. Farmacologia — Regime B")
        st.session_state.setdefault("preset_b", "BRA máximo")
        preset_b = st.selectbox(
            "Preset do Regime B",
            list(PRESETS.keys()),
            key="preset_b",
        )
        if PRESETS[preset_b] is None:
            for key, default in (("drug_b_ace", 0.0), ("drug_b_arb", 1.0), ("drug_b_dri", 0.0)):
                st.session_state.setdefault(key, default)
            b_ace = st.slider("IECA — Regime B", 0.0, 1.0, step=0.1, key="drug_b_ace")
            b_arb = st.slider("BRA — Regime B", 0.0, 1.0, step=0.1, key="drug_b_arb")
            b_dri = st.slider("Inibidor da renina — Regime B", 0.0, 1.0, step=0.1, key="drug_b_dri")
            drugs_b = {"ace": float(b_ace), "arb": float(b_arb), "dri": float(b_dri)}
        else:
            drugs_b = dict(PRESETS[preset_b])
        label_b = regime_label(drugs_b)
        diagram_choice = st.selectbox("Diagrama animado mostra", ["Regime A", "Regime B"], index=0)
        diagram_regime = "B" if diagram_choice == "Regime B" else "A"
        st.caption("Curva contínua = A; curva tracejada = B. Cada regime mantém seu próprio basal.")

label_a = regime_label(drugs)
try:
    data = simulate(drop=queda / 100.0, t_end=t_end, is_chronic=is_chronic, drugs=drugs, params=PARAMS)
except Exception as exc:
    st.error(f"Não foi possível concluir a simulação do Regime A: {exc}")
    st.stop()

if not data["equilibrium_ok"]:
    st.warning(
        "O algoritmo não encontrou um equilíbrio com resíduo suficientemente pequeno no Regime A. "
        "Interprete este cenário com cautela; reveja os parâmetros ou a intensidade dos fármacos. "
        f"Resíduo máximo normalizado: {data['equilibrium_residual']:.2e}."
    )

data_b = None
comparison_payload = None
if compare_enabled and drugs_b is not None:
    try:
        data_b = simulate(drop=queda / 100.0, t_end=t_end, is_chronic=is_chronic, drugs=drugs_b, params=PARAMS)
    except Exception as exc:
        st.error(f"Não foi possível concluir a simulação do Regime B: {exc}")
        st.stop()
    # Não interpolar: os dois regimes precisam ter exatamente a mesma grade temporal.
    if data["t"] != data_b["t"]:
        st.error("As simulações A e B não compartilham exatamente o mesmo vetor de tempo; a comparação foi interrompida para evitar interpolação silenciosa.")
        st.stop()
    comparison_keys = ("PAM", "GFR", "R", "A1", "A2", "AT1", "AT2", "Ald", "E", "ADH", "BK", "Na", "K", "V")
    comparison_payload = {key: data_b[key] for key in comparison_keys}
    comparison_payload.update({
        "drugs": data_b["drugs"],
        "baseline": data_b["baseline"],
        "baseline_outputs": data_b["baseline_outputs"],
        "equilibrium_ok": data_b["equilibrium_ok"],
        "equilibrium_residual": data_b["equilibrium_residual"],
    })
    if not data_b["equilibrium_ok"]:
        st.warning(
            "O algoritmo não encontrou um equilíbrio com resíduo suficientemente pequeno no Regime B. "
            f"Resíduo máximo normalizado: {data_b['equilibrium_residual']:.2e}."
        )

if compare_enabled and data_b is not None:
    same_regime = all(abs(float(drugs.get(k, 0.0)) - float(drugs_b.get(k, 0.0))) < 1e-9 for k in ("ace", "arb", "dri"))
    if same_regime:
        st.info("Regimes idênticos: as curvas A e B devem se sobrepor (salvo diferenças numéricas de arredondamento).")

payload_data = dict(data)
payload_data.update({
    "cmp": comparison_payload,
    "compare_enabled": bool(compare_enabled and comparison_payload is not None),
    "diagram_regime": diagram_regime,
    "labelA": label_a,
    "labelB": label_b,
})

if compare_enabled and data_b is not None:
    if drugs.get("ace", 0) > 0 and drugs.get("arb", 0) == 0 and drugs.get("dri", 0) == 0 \
       and drugs_b.get("arb", 0) > 0 and drugs_b.get("ace", 0) == 0 and drugs_b.get("dri", 0) == 0:
        compare_hint = (
            "Observe: com IECA, a Ang II tende a cair e a bradicinina aumenta; com BRA, a Ang II pode aumentar "
            "e a sinalização via AT2 sobe, enquanto a bradicinina permanece no basal do modelo. "
            "A TFG é estimada e a variação é calculada em relação ao basal próprio de cada regime."
        )
    elif drugs.get("arb", 0) > 0 and drugs.get("ace", 0) == 0 and drugs.get("dri", 0) == 0 \
         and drugs_b.get("ace", 0) > 0 and drugs_b.get("arb", 0) == 0 and drugs_b.get("dri", 0) == 0:
        compare_hint = (
            "Observe: com BRA, a Ang II pode aumentar e a sinalização via AT2 sobe; com IECA, a Ang II tende a cair "
            "e a bradicinina aumenta. A TFG é estimada e a variação é calculada em relação ao basal próprio de cada regime."
        )
    elif all(abs(float(drugs.get(k, 0.0))) < 1e-9 for k in ("ace", "arb", "dri")) and \
         drugs_b.get("ace", 0.0) > 0 and drugs_b.get("arb", 0.0) > 0 and drugs_b.get("dri", 0.0) == 0:
        compare_hint = (
            "Observe a diferença entre o regime sem fármacos e o bloqueio combinado: acompanhe Ang II, AT1, aldosterona, "
            "K⁺ e TFG estimada. O bloqueio combinado é apenas um cenário didático, não uma recomendação terapêutica."
        )
    else:
        compare_hint = (
            "Compare a direção e a magnitude das curvas. Cada regime começa em seu próprio estado estacionário; "
            "as porcentagens da tabela usam esse basal, não um único basal comum."
        )
else:
    compare_hint = ""
payload_data["comparison_hint"] = compare_hint
payload = json.dumps(payload_data, separators=(",", ":"))

# =====================================================================
# 3. INTERFACE HTML/SVG/CANVAS
# =====================================================================
TEMPLATE = r"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><style>
:root{--ink:#172554;--muted:#526581;--line:#d5deea;}
*{box-sizing:border-box;}
body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
.wrap{padding:6px 12px 8px;position:relative;}
.top{display:flex;align-items:center;gap:16px;margin-bottom:2px;}
.pam{font-size:32px;font-weight:800;color:#c0392b;line-height:1;min-width:190px;}
.pam small{display:block;font-size:12px;font-weight:600;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;margin-top:3px;}
.gw{flex:1;position:relative;height:16px;}
.gauge{position:absolute;inset:0;background:linear-gradient(90deg,#e74c3c 0%,#f1c40f 50%,#16a34a 100%);border-radius:8px;overflow:hidden;}
.gauge>i{position:absolute;right:0;top:0;bottom:0;width:25%;border-radius:0 8px 8px 0;background:#eef2f8;transition:width .08s linear;}
.clock{font-variant-numeric:tabular-nums;font-size:16px;font-weight:700;color:#4a5b78;min-width:120px;text-align:right;}
svg{display:block;width:100%;height:auto;}
.bg{stroke:#dbe4f0;stroke-width:2;}
.ttl{font-size:15px;font-weight:700;fill:#22344f;text-anchor:middle;}
.sub{font-size:12px;fill:#7a8ba3;text-anchor:middle;}
.val{font-size:13px;font-weight:700;text-anchor:middle;font-variant-numeric:tabular-nums;}
.flow{fill:none;stroke-width:4;stroke-linecap:round;stroke-dasharray:9 11;}
.charts{display:flex;flex-direction:column;gap:16px;margin-top:16px;width:100%;}
.card{flex:none;width:100%;min-width:0;background:#fbfcfe;border:1px solid var(--line);border-radius:14px;padding:12px 14px 10px;}
.card h4{margin:0 0 5px 4px;font-size:16px;letter-spacing:.025em;text-transform:uppercase;color:var(--muted);}
.card .hint{font-size:14px;color:#71839d;margin:0 0 5px 4px;line-height:1.4;}
canvas{display:block;width:100%;height:auto;}
.bar{display:flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap;}
button{border:1px solid #cbd5e1;background:#fff;color:#172554;font-weight:650;font-size:14px;padding:7px 14px;border-radius:9px;cursor:pointer;}
button:hover{background:#f2f6fd;}
button.pri{background:#0077ff;border-color:#0077ff;color:#fff;}
button.pri:hover{background:#185a95;}
.speed{display:flex;gap:6px;}
.speed button{padding:6px 10px;}
.speed button.on{background:#e8f0fb;border-color:#0077ff;color:#0077ff;}
.status{font-size:13px;color:#4a5b78;flex:1;text-align:right;font-weight:600;min-width:190px;}
#alertBox{position:absolute;top:70px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:100;pointer-events:none;max-width:350px;}
.alert{background:#ffe5e5;border-left:6px solid #dc2626;color:#b71c1c;padding:10px 13px;border-radius:6px;font-weight:700;font-size:13px;box-shadow:0 4px 12px rgba(0,0,0,0.15);display:none;}
.alert.warning{background:#fff3e0;border-color:#f57c00;color:#e65100;}
.note{font-size:12px;color:#7a8ba3;margin:7px 2px 0;line-height:1.4;}
.comparison-legend{border:1px solid #dbe4f0;background:#f8fafc;border-radius:10px;padding:8px 12px;margin-top:10px;display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;}
.legend-items{display:flex;gap:18px;align-items:center;flex-wrap:wrap;}
.legend-item{display:inline-flex;gap:7px;align-items:center;font-size:13px;color:#334155;}
.line-sample{display:inline-block;width:34px;height:0;border-top:3px solid #334155;}
.line-sample.dashed{border-top-style:dashed;}
.comparison-hint{flex-basis:100%;font-size:12px;color:#64748b;line-height:1.4;}
</style></head><body><div class="wrap">
<div id="alertBox">
  <div id="alertCough" class="alert warning">⚠ Índice de bradicinina elevado neste modelo (não é previsão de tosse individual).</div>
  <div id="alertK" class="alert">⚠ Índice de K⁺ elevado (não equivale a diagnóstico de hipercalemia).</div>
</div>
<div class="top">
  <div class="pam"><span id="pamv">90</span> <span style="font-size:14px;color:#7a8ba3">mmHg</span><small id="pamLabel">Pressão arterial média estimada</small></div>
  <div class="gw"><div class="gauge"><i id="gfill"></i></div></div>
  <div class="clock" id="clock">t = 0 min</div>
</div>
<svg id="dia" viewBox="0 0 1000 420" preserveAspectRatio="xMidYMid meet">
  <defs>
    <marker id="flowArrow" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L6,3 L0,6 z" fill="#64748b"/>
    </marker>
  </defs>
  <g id="flows">
    <line class="flow" id="f_ren" x1="180" y1="72" x2="240" y2="72" stroke="#7c3aed" marker-end="url(#flowArrow)"/>
    <line class="flow" id="f_a1" x1="380" y1="72" x2="440" y2="72" stroke="#009e73" marker-end="url(#flowArrow)"/>
    <line class="flow" id="f_a2" x1="580" y1="72" x2="640" y2="72" stroke="#f04e37" marker-end="url(#flowArrow)"/>
    <line class="flow" id="f_ald" x1="755" y1="115" x2="850" y2="239" stroke="#d81b60" marker-end="url(#flowArrow)"/>
    <polyline class="flow" id="f_at1" points="570,116 600,130 610,190 660,239" stroke="#f04e37" marker-end="url(#flowArrow)"/>
    <line class="flow" id="f_at2" x1="535" y1="116" x2="760" y2="239" stroke="#f59e0b" marker-end="url(#flowArrow)"/>
    <line class="flow" id="f_bk" x1="580" y1="185" x2="619" y2="278" stroke="#9333ea" marker-end="url(#flowArrow)"/>
    <line class="flow" id="f_adh" x1="880" y1="116" x2="920" y2="239" stroke="#16a34a" marker-end="url(#flowArrow)"/>
  </g>
  <g id="bxR"><rect class="bg" x="20" y="28" width="160" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="100" y="56">Rim</text><text class="sub" x="100" y="73">atividade da renina</text>
    <text class="val" x="100" y="98" id="vR" fill="#7c3aed">renina 1.00</text></g>
  <g id="bxF"><rect class="bg" x="240" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="310" y="56">Angiotensina I</text><text class="sub" x="310" y="73">precursor</text>
    <text class="val" x="310" y="98" id="vA1" fill="#009e73">Ang I 1.00</text></g>
  <g id="bxP"><rect class="bg" x="440" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="510" y="56">Angiotensina II</text><text class="sub" x="510" y="73">formação via ECA</text>
    <text class="val" x="510" y="98" id="vA2" fill="#f04e37">Ang II 1.00</text></g>
  <g id="bxA"><rect class="bg" x="640" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="710" y="56">Adrenal</text><text class="sub" x="710" y="73">aldosterona</text>
    <text class="val" x="710" y="98" id="vAld" fill="#d81b60">Ald 1.00</text></g>
  <g id="bxH"><rect class="bg" x="840" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="910" y="56">Neuro-hipófise</text><text class="sub" x="910" y="73">ADH / vasopressina</text>
    <text class="val" x="910" y="98" id="vADH" fill="#16a34a">ADH 1.00</text></g>
  <g id="bxV"><rect class="bg" x="620" y="240" width="180" height="110" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="710" y="265" font-size="13">Vasos / receptores</text>
    <rect id="lumen" x="645" y="280" width="130" height="34" rx="10" fill="#f7d9d3" stroke="#f04e37" stroke-width="2"/>
    <text class="val" x="660" y="330" id="vAT1" fill="#f04e37">AT1 1.00</text>
    <text class="val" x="760" y="330" id="vAT2" fill="#f59e0b">AT2 1.00</text></g>
  <g id="bxT"><rect class="bg" x="820" y="240" width="160" height="110" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="900" y="261" font-size="12">Balanço hidroeletrolítico</text>
    <text class="sub" x="900" y="276">Na corporal / K⁺ plasmático</text>
    <rect x="835" y="280" width="130" height="26" rx="13" fill="#e6eefb" stroke="#c3d5ee"/>
    <g id="dots"></g>
    <text class="val" x="865" y="330" id="vNa" fill="#006dff">Na 1.00</text>
    <text class="val" x="935" y="330" id="vK" fill="#dc2626">K 1.00</text></g>
  <g id="bxK"><rect class="bg" x="440" y="150" width="140" height="70" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="510" y="175" font-size="13">Bradicinina</text>
    <text class="val" x="510" y="200" id="vBK" fill="#9333ea">BK 1.00</text></g>
</svg>
<div class="comparison-legend" id="comparisonLegend" style="display:none">
  <div class="legend-items">
    <span class="legend-item"><i class="line-sample solid"></i><strong id="labelA">Regime A</strong></span>
    <span class="legend-item"><i class="line-sample dashed"></i><strong id="labelB">Regime B</strong></span>
  </div>
  <div class="comparison-hint" id="comparisonHint"></div>
</div>
<div class="charts" aria-label="Gráficos do simulador">
  <div class="card"><h4>PAM e TFG estimada</h4><div class="hint">Eixo esquerdo: mmHg · eixo direito: mL/min/1,73 m²</div><canvas id="c1" width="1200" height="440"></canvas></div>
  <div class="card"><h4>Cascata hormonal</h4><div class="hint">Índices relativos (referência sem fármacos = 1)</div><canvas id="c2" width="1200" height="440"></canvas></div>
  <div class="card"><h4>Sinalização dos receptores</h4><div class="hint">Atividade efetiva relativa (referência sem fármacos = 1)</div><canvas id="c3" width="1200" height="440"></canvas></div>
  <div class="card"><h4>Aldosterona, volume e bradicinina</h4><div class="hint">Eixo esquerdo: aldosterona/volume · eixo direito: bradicinina (índices relativos)</div><canvas id="c4" width="1200" height="440"></canvas></div>
  <div class="card"><h4>Sódio corporal e K⁺ plasmático</h4><div class="hint">Referência sem fármacos = 1; não são concentrações clínicas</div><canvas id="c5" width="1200" height="440"></canvas></div>
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
const CMP = D.cmp || null;
const IS_COMPARE = Boolean(D.compare_enabled && CMP);
const T = D.t;
// O diagrama animado pode mostrar A ou B, enquanto todos os gráficos comparam A/B.
const DIAG = (D.diagram_regime === 'B' && CMP) ? CMP : D;
const T0=T[0], T1=T[T.length-1], IS_CHRONIC=D.is_chronic;
const HAS_DIST=D.drop>0;
const DRUGS=DIAG.drugs;
const BASE=DIAG.baseline;
const BASE_OUT=DIAG.baseline_outputs;
const PAM=DIAG.PAM, GFR=DIAG.GFR, RR=DIAG.R, A1=DIAG.A1, A2=DIAG.A2,
      AT1=DIAG.AT1, AT2=DIAG.AT2, ALD=DIAG.Ald, ADH=DIAG.ADH, BK=DIAG.BK,
      NA=DIAG.Na, K=DIAG.K, V=DIAG.V, SS=DIAG.S;
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
// Percorre arrays explicitamente; evita passar dezenas de milhares de argumentos
// para Math.min/Math.max no modo crônico com duas séries sobrepostas.
function getRange(series){
  let mn=Infinity, mx=-Infinity, rawMin=Infinity, rawMax=-Infinity;
  for(const s of series){
    const arrays=[s.data,s.dataB];
    for(const arr of arrays){
      if(!arr)continue;
      for(let i=0;i<arr.length;i++){
        const v=arr[i];
        if(!Number.isFinite(v))continue;
        if(v<mn)mn=v;if(v>mx)mx=v;
        if(v<rawMin)rawMin=v;if(v>rawMax)rawMax=v;
      }
    }
  }
  if(!Number.isFinite(mn)||!Number.isFinite(mx))return [0,1];
  const span=mx-mn;
  if(span<1e-8){const pad=Math.max(Math.abs(mx)*0.05,0.05);mn-=pad;mx+=pad;}
  else {const pad=Math.max(span*0.12,Math.abs(mx)*0.005,0.005);mn-=pad;mx+=pad;}
  // Só fixa zero como limite se a margem gerou um limite negativo para séries positivas.
  if(rawMin>0 && mn<0)mn=0;
  return [mn,mx];
}
function setPath(ctx,times,arr,px,py,color,dashed){
  ctx.strokeStyle=color;ctx.lineWidth=3.0;ctx.setLineDash(dashed?[8,5]:[]);ctx.beginPath();
  for(let i=0;i<times.length;i++){
    const x=px(times[i]),y=py(arr[i]);
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.stroke();ctx.setLineDash([]);
}
function makeChart(cv,config){
  const ctx=cv.getContext('2d'), W=cv.width, H=cv.height;
  const L=72, R=config.hasRightAxis?72:24, Tp=38, B=44;
  const leftSeries=config.series.filter(s=>s.axis!=='right');
  const rightSeries=config.series.filter(s=>s.axis==='right');
  const [y1min,y1max]=getRange(leftSeries);
  let y2min=0,y2max=1;
  if(config.hasRightAxis && rightSeries.length){[y2min,y2max]=getRange(rightSeries);}
  const px=t=>L+(t-T0)/(T1-T0)*(W-L-R);
  const pyLeft=v=>Tp+(1-(v-y1min)/(y1max-y1min))*(H-Tp-B);
  const pyRight=v=>Tp+(1-(v-y2min)/(y2max-y2min))*(H-Tp-B);
  const off=document.createElement('canvas'); off.width=W;off.height=H;
  const o=off.getContext('2d');o.fillStyle='#fbfcfe';o.fillRect(0,0,W,H);
  o.font='15px -apple-system,Segoe UI,Roboto,sans-serif';
  for(let i=0;i<=4;i++){
    const f=i/4,v=y1min+(y1max-y1min)*f,y=pyLeft(v);
    o.strokeStyle='#eef2f8';o.lineWidth=1;o.beginPath();o.moveTo(L,y);o.lineTo(W-R,y);o.stroke();
    o.fillStyle=leftSeries.length?leftSeries[0].color:'#9aabc2';o.textAlign='right';o.textBaseline='middle';
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
    if(t<0)label='pré';
    else label=IS_CHRONIC?(t/1440).toFixed(t/1440<10?1:0):t.toFixed(0);
    o.fillStyle='#9aabc2';o.textAlign='center';o.textBaseline='top';o.fillText(label,x,H-B+7);
  }
  if(HAS_DIST){o.strokeStyle='#e8b04b';o.lineWidth=2;o.setLineDash([5,5]);o.beginPath();o.moveTo(px(0),Tp);o.lineTo(px(0),H-B);o.stroke();o.setLineDash([]);}
  config.series.forEach(s=>{
    const py=s.axis==='right'?pyRight:pyLeft;
    setPath(o,T,s.data,px,py,s.color,false);
    if(IS_COMPARE && s.dataB)setPath(o,T,s.dataB,px,py,s.color,true);
  });
  if(config.legend){
    o.font='14px -apple-system,Segoe UI,Roboto,sans-serif';o.textAlign='left';o.textBaseline='middle';
    let lx=L+4, ly=12;
    config.series.forEach(s=>{
      const width=15+o.measureText(s.name).width+10;
      if(lx+width>W-R){lx=L+4;ly+=18;}
      o.setLineDash([]);o.strokeStyle=s.color;o.lineWidth=3;o.beginPath();o.moveTo(lx,ly);o.lineTo(lx+15,ly);o.stroke();
      o.fillStyle='#5a6b85';o.fillText(s.name,lx+19,ly+2);lx+=width;
    });
  }
  return {ctx,off,px,pyLeft,pyRight,W,H,L,R,Tp,B,config};
}
function chartSeries(key,name,color,axis='left'){
  return {name,data:D[key],dataB:(IS_COMPARE?CMP[key]:null),color,axis};
}
const ch1=makeChart(document.getElementById('c1'),{hasRightAxis:true,legend:true,series:[
  chartSeries('PAM','PAM','#c0392b','left'),chartSeries('GFR','TFG estimada','#2980b9','right')]});
const cascadeSeries=IS_COMPARE
  ? [chartSeries('R','Renina','#7c3aed'),chartSeries('A2','Ang II','#f04e37')]
  : [chartSeries('R','Renina','#7c3aed'),chartSeries('A1','Ang I','#009e73'),chartSeries('A2','Ang II','#f04e37')];
const ch2=makeChart(document.getElementById('c2'),{hasRightAxis:false,legend:true,series:cascadeSeries});
const ch3=makeChart(document.getElementById('c3'),{hasRightAxis:false,legend:true,series:[
  chartSeries('AT1','AT1','#d35400'),chartSeries('AT2','AT2','#f59e0b')]});
const ch4=makeChart(document.getElementById('c4'),{hasRightAxis:true,legend:true,series:[
  chartSeries('Ald','Aldosterona','#d81b60'),chartSeries('V','Volume','#8e44ad'),chartSeries('BK','Bradicinina','#9333ea','right')]});
const ch5=makeChart(document.getElementById('c5'),{hasRightAxis:false,legend:true,series:[
  chartSeries('Na','Na corporal','#006dff'),chartSeries('K','K plasmático','#dc2626')]});
function drawMarker(ctx,x,y,color,hollow){
  ctx.beginPath();ctx.arc(x,y,hollow?4.5:4,0,Math.PI*2);
  ctx.fillStyle=hollow?'#ffffff':color;ctx.fill();ctx.strokeStyle=hollow?color:'#ffffff';ctx.lineWidth=hollow?2:1.5;ctx.stroke();
}
function drawChart(ch,t){
  const c=ch.ctx;c.clearRect(0,0,ch.W,ch.H);c.drawImage(ch.off,0,0);
  const x=ch.px(t);c.strokeStyle='#22344f';c.lineWidth=1.6;c.beginPath();c.moveTo(x,ch.Tp);c.lineTo(x,ch.H-ch.B);c.stroke();
  ch.config.series.forEach(s=>{
    const py=s.axis==='right'?ch.pyRight:ch.pyLeft;
    drawMarker(c,x,py(sample(s.data,t)),s.color,false);
    if(IS_COMPARE && s.dataB)drawMarker(c,x,py(sample(s.dataB,t)),s.color,true);
  });
}
const legendBox=document.getElementById('comparisonLegend');
if(IS_COMPARE){
  legendBox.style.display='flex';
  document.getElementById('labelA').textContent='A — '+D.labelA+' (linha contínua)';
  document.getElementById('labelB').textContent='B — '+D.labelB+' (linha tracejada)';
  document.getElementById('comparisonHint').textContent=D.comparison_hint||'Cada regime tem seu próprio basal.';
}
function deviationScale(v,baseline){
  // Escala fixa entre cenários: 50% de mudança relativa ao basal satura em 1.
  // Assim, uma perturbação pequena não vira automaticamente intensidade máxima.
  const ref=Math.max(Math.abs(baseline),0.1);
  return clamp(Math.abs(v-baseline)/(0.5*ref),0,1);
}
function h2r(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function mix(a,b,f){const A=h2r(a),B=h2r(b);return 'rgb('+Math.round(lerp(A[0],B[0],f))+','+Math.round(lerp(A[1],B[1],f))+','+Math.round(lerp(A[2],B[2],f))+')';}
function setBox(id,n,pale,sat){const rect=document.querySelector('#'+id+' .bg');if(rect)rect.setAttribute('fill',mix(pale,sat,n));}
const dg=document.getElementById('dots'),dots=[];
for(let i=0;i<6;i++){const el=document.createElementNS('http://www.w3.org/2000/svg','circle');el.setAttribute('r',4);el.setAttribute('cy',293);el.setAttribute('fill','#006dff');dg.appendChild(el);dots.push({el,x:835+i*22});}
const off={f_ren:0,f_a1:0,f_a2:0,f_ald:0,f_at1:0,f_at2:0,f_bk:0,f_adh:0};
let simT=T0,playing=true,speed=1,lastTs=null,coughShown=false,kShown=false;
function render(dt){
  const pam=sample(PAM,simT),r=sample(RR,simT),a1=sample(A1,simT),a2=sample(A2,simT),at1=sample(AT1,simT),at2=sample(AT2,simT),
        al=sample(ALD,simT),adh=sample(ADH,simT),bk=sample(BK,simT),na=sample(NA,simT),k=sample(K,simT),vol=sample(V,simT);
  document.getElementById('pamv').textContent=pam.toFixed(1);
  document.getElementById('pamLabel').textContent='PAM estimada · '+(D.diagram_regime==='B'&&CMP?'Regime B':'Regime A');
  document.getElementById('gfill').style.width=(100-clamp(pam/120*100,0,100))+'%';
  let timeStr=IS_CHRONIC?(Math.max(simT,0)/1440).toFixed(2)+' dias':Math.max(simT,0).toFixed(1)+' min';
  if(simT<0)timeStr='pré-perturbação';
  document.getElementById('clock').textContent='t = '+timeStr;
  setBox('bxR',deviationScale(r,BASE.R),'#ece4f7','#7c3aed');
  setBox('bxF',deviationScale(a1,BASE.A1),'#dff3f0','#009e73');
  setBox('bxP',deviationScale(a2,BASE.A2),'#fbe3dc','#f04e37');
  setBox('bxA',deviationScale(al,BASE.Ald),'#dce9f8','#d81b60');
  setBox('bxH',deviationScale(adh,BASE.ADH),'#e0f5e6','#16a34a');
  setBox('bxK',deviationScale(bk,BASE.BK),'#f4e6f7','#9333ea');
  setBox('bxV',Math.max(deviationScale(at1,BASE_OUT.AT1),deviationScale(at2,BASE_OUT.AT2)),'#fbe3dc','#f04e37');
  setBox('bxT',Math.max(deviationScale(na,BASE.Na_body),deviationScale(k,BASE.K_plasma)),'#e6eefb','#006dff');
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
  const arrows=[['f_ren',r,BASE.R],['f_a1',a1,BASE.A1],['f_a2',a2,BASE.A2],['f_ald',al,BASE.Ald],['f_at1',at1,BASE_OUT.AT1],['f_at2',at2,BASE_OUT.AT2],['f_bk',bk,BASE.BK],['f_adh',adh,BASE.ADH]];
  arrows.forEach(([id,v,b0])=>{const n=deviationScale(v,b0);off[id]-=(5+75*n)*dt;const el=document.getElementById(id);el.setAttribute('stroke-dashoffset',off[id]);el.setAttribute('opacity',0.15+0.85*n);});
  // Lúmen: AT1 contrai; AT2 dilata. Altura basal é exatamente 34 px.
  const h=clamp(34-8*(at1-1)+4*(at2-1),12,50);
  const lum=document.getElementById('lumen');lum.setAttribute('height',h);lum.setAttribute('y',280+(34-h)/2);
  dots.forEach(d=>{d.x+=(8+12*clamp(na,0.3,2.0))*dt;if(d.x>965)d.x=835;d.el.setAttribute('cx',d.x);d.el.setAttribute('opacity',clamp(0.5+0.25*(na-1),0.2,1));});
  drawChart(ch1,simT);drawChart(ch2,simT);drawChart(ch3,simT);drawChart(ch4,simT);drawChart(ch5,simT);
  // Avisos deliberadamente rotulados como índices didáticos, não diagnósticos.
  if(DRUGS.ace>0.05 && bk>1.50 && !coughShown && simT>0){document.getElementById('alertCough').style.display='block';coughShown=true;}
  if(k>1.15 && !kShown){document.getElementById('alertK').style.display='block';kShown=true;}
  const status=document.getElementById('status');
  if(simT<0)status.textContent='Regime basal antes da perturbação';
  else if(pam < BASE_OUT.PAM - 5.0)status.textContent='PAM caiu mais de 5 mmHg em relação ao basal do cenário; compensações em curso';
  else if(k>1.15)status.textContent='Índice relativo de K⁺ elevado no modelo';
  else if(DRUGS.ace>0.05 && bk>1.50)status.textContent='IECA: bradicinina elevada no modelo';
  else if(DRUGS.arb>0.05)status.textContent='BRA: sinalização AT1 reduzida no modelo';
  else if(DRUGS.ace>0.05)status.textContent='IECA: conversão Ang I → Ang II reduzida';
  else if(DRUGS.dri>0.05)status.textContent='Inibidor da renina ativo no modelo';
  else if(!HAS_DIST)status.textContent='Regime estacionário (sem perturbação)';
  else status.textContent='Resposta do modelo à perturbação';
}
let modelVisible=false,modelFrameQueued=false;
function scheduleModelFrame(){if(modelVisible&&!modelFrameQueued){modelFrameQueued=true;requestAnimationFrame(loop);}}
function loop(ts){
  modelFrameQueued=false;if(!modelVisible)return;
  if(lastTs===null)lastTs=ts;
  let dt=(ts-lastTs)/1000;lastTs=ts;dt=Math.min(dt,0.05);
  if(playing){simT+=dt*(IS_CHRONIC?1440.0/15.0:1.0)*speed;if(simT>=T1){simT=T1;playing=false;btnPlay.textContent='▶ Reproduzir';}}
  render(playing?dt*speed:0);scheduleModelFrame();
}
const modelObserver=new IntersectionObserver(entries=>{const visible=Boolean(entries[0]&&entries[0].isIntersecting);if(visible!==modelVisible){modelVisible=visible;lastTs=null;}if(modelVisible)scheduleModelFrame();});
modelObserver.observe(document.body);
const btnPlay=document.getElementById('btnPlay');
btnPlay.onclick=()=>{if(!playing&&simT>=T1)simT=T0;playing=!playing;btnPlay.textContent=playing?'⏸ Pausar':'▶ Reproduzir';};
document.getElementById('btnReset').onclick=()=>{simT=T0;playing=true;btnPlay.textContent='⏸ Pausar';for(const key in off)off[key]=0;coughShown=false;kShown=false;document.getElementById('alertCough').style.display='none';document.getElementById('alertK').style.display='none';};
document.querySelectorAll('.speed button').forEach(b=>{b.onclick=()=>{document.querySelectorAll('.speed button').forEach(x=>x.classList.remove('on'));b.classList.add('on');speed=parseFloat(b.dataset.s);};});
</script></body></html>
"""

CELL_TEMPLATE = r"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--ink:#172554;--muted:#526581;--line:#cbd5e1;--blue:#0077ff;--yellow:#ffb703;--red:#f94144;--purple:#7b2cbf;--green:#00a878;--ald:#f72585;}
*{box-sizing:border-box} body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{padding:12px 14px 16px}.header{display:flex;align-items:flex-start;gap:12px;justify-content:space-between;flex-wrap:wrap}
h2{margin:0 0 5px;font-size:24px;line-height:1.2}.sub{font-size:14px;color:var(--muted);line-height:1.5;margin:0 0 12px}
.pill{border:1px solid #bfdbfe;background:#eff6ff;border-radius:999px;padding:6px 11px;font-size:13px;color:#1e40af;white-space:nowrap}
.controls{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:7px 0}
button{border:1px solid #cbd5e1;background:#fff;color:#172554;border-radius:8px;padding:8px 12px;font-size:14px;font-weight:650;cursor:pointer}
button:hover{background:#eef6ff}button.primary,.stage.on{background:#0077ff;color:#fff;border-color:#0077ff}
button:focus-visible,input:focus-visible{outline:3px solid #93c5fd;outline-offset:2px}
#stageButtons{display:flex;gap:7px;flex-wrap:wrap}.stage{min-width:42px}
.statusbox{border:1px solid var(--line);border-left:5px solid var(--ald);background:#fff7fb;border-radius:10px;padding:11px 14px;margin:10px 0;font-size:14px;line-height:1.5;min-height:58px}
.timebar{display:flex;gap:11px;align-items:center;flex-wrap:wrap;font-size:14px;color:var(--muted);margin:10px 0}
input[type=range]{flex:1;min-width:180px;accent-color:#0077ff;height:22px}
svg{width:100%;height:auto;display:block;background:#fff;border:1px solid var(--line);border-radius:12px}
.regionLabel{font-size:17px;font-weight:750;fill:#172554;text-anchor:middle}.smallLabel{font-size:14px;fill:#334155;text-anchor:middle}.tinyLabel{font-size:12px;font-weight:550;fill:#475569;text-anchor:middle}
.guide{fill:none;stroke:#94a3b8;stroke-width:2;stroke-dasharray:5 6;opacity:.6}.particle{pointer-events:none;filter:drop-shadow(0 0 1px #ffffff)}
#cellBody{fill:#e0f7ff;stroke:#0891b2;stroke-width:3.5} .membrane{fill:#67e8f9;stroke:#0e7490;stroke-width:1.6}
.protein{stroke:#075985;stroke-width:2.2;fill:#38bdf8}.pump{stroke:#075985;stroke-width:2.2;fill:#22d3ee}
#nucleus{fill:#ede9fe;stroke:#7c3aed;stroke-width:2.4}#dna{fill:none;stroke:#6d28d9;stroke-width:2.8;stroke-linecap:round}
#receptor{fill:#e9d5ff;stroke:#7b2cbf;stroke-width:2.5}#receptorHalo{fill:#fce7f3;stroke:#f72585;stroke-width:2.5;opacity:0}
#gateA,#gateB{stroke:#075985;stroke-width:4.5;stroke-linecap:round;transition:transform .35s ease}
#newChannel,#newPump{opacity:.22;transition:opacity .45s ease,filter .45s ease}#newChannel.visible,#newPump.visible{opacity:1;filter:drop-shadow(0 0 5px #0891b2)}
#nucleus.glow{filter:drop-shadow(0 0 9px #a78bfa)}#gateA,#gateB{transform-box:fill-box;transform-origin:center}#atpFlash{opacity:0;fill:#ffb703;stroke:#c2410c;stroke-width:2.3}#atpFlash.on{opacity:1}
#naFlux,#kFlux{opacity:.9;transition:opacity .25s ease,stroke-width .25s ease}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:#334155;margin:10px 2px}
.key{display:inline-flex;align-items:center;gap:7px}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;box-shadow:0 0 0 1px rgba(15,23,42,.14)}
.foot{font-size:12px;color:#526581;line-height:1.45;margin-top:8px}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}button{scroll-behavior:auto}}
</style></head><body><div class="wrap">
<div class="header"><div><h2>Como a aldosterona altera o transporte tubular</h2><p class="sub">Célula epitelial do túbulo distal — esquema didático. Aldosterona, E (proxy de canais) e K⁺ vêm das EDOs; receptor, transcrição e inserção de proteínas são representações simplificadas.</p></div><span class="pill" id="regimeLabel">Regime selecionado</span></div>
<div class="controls"><button id="modeStep" class="primary">Passo a passo</button><button id="modeContinuous">Simulação contínua</button><button id="playCell">▶ Reproduzir tempo</button><button id="resetCell">↺ Reiniciar partículas</button></div>
<div id="stageButtons"><button class="stage on" data-stage="1">1 · Ligação</button><button class="stage" data-stage="2">2 · Transcrição</button><button class="stage" data-stage="3">3 · Síntese</button><button class="stage" data-stage="4">4 · Modulação</button><button class="stage" data-stage="5">5 · Transporte</button></div>
<div class="timebar"><label for="timeSeek">Tempo simulado:</label><input id="timeSeek" type="range" min="0" max="1" step="any" value="0"><strong id="timeReadout">basal</strong><span id="valuesReadout">Ald = 1,00 · K⁺ = 1,00</span></div>
<div class="statusbox" id="stageText">1. A aldosterona, um esteroide lipossolúvel, difunde-se a partir do sangue e liga-se a um receptor intracelular. Não atravessa um canal iônico.</div>
<svg id="cellSvg" viewBox="0 0 1100 560" role="img" aria-label="Esquema animado de uma célula tubular com lúmen à esquerda, sangue à direita, receptor intracelular, núcleo, canais apicais e bomba sódio-potássio basolateral">
 <defs>
   <marker id="arrBlue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#0077ff"/></marker>
   <marker id="arrRed" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#f94144"/></marker>
   <marker id="arrPurple" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#7b2cbf"/></marker>
 </defs>
 <!-- Compartimentos -->
 <rect x="18" y="55" width="225" height="450" rx="10" fill="#fff9c9" stroke="#eadf8a"/>
 <rect id="cellBody" x="280" y="55" width="480" height="450" rx="28"/>
 <rect x="760" y="55" width="125" height="450" fill="#f1e9fb" stroke="#c4b5d9"/>
 <rect x="885" y="55" width="195" height="450" rx="10" fill="#fbe6e2" stroke="#e4b4ac"/>
 <text class="regionLabel" x="130" y="34">Lúmen tubular</text><text class="regionLabel" x="520" y="34">Célula epitelial</text><text class="smallLabel" x="822" y="34">Interstício</text><text class="regionLabel" x="980" y="34">Sangue</text>
 <!-- Membranas -->
 <rect class="membrane" x="257" y="65" width="13" height="430" rx="6"/><rect class="membrane" x="768" y="65" width="13" height="430" rx="6"/>
 <text class="tinyLabel" x="263" y="522">Membrana apical</text><text class="tinyLabel" x="775" y="522">Membrana basolateral</text>
 <!-- Núcleo e DNA -->
 <ellipse id="nucleus" cx="505" cy="158" rx="91" ry="57"/><text class="smallLabel" x="505" y="125">Núcleo</text>
 <path id="dna" d="M450 151 C465 127 480 175 495 151 S525 127 540 151 S570 175 585 151 M450 164 C465 188 480 140 495 164 S525 188 540 164 S570 140 585 164"/>
 <text class="tinyLabel" x="505" y="201">DNA / transcrição</text>
 <!-- Receptor e complexo -->
 <circle id="receptorHalo" cx="663" cy="258" r="26"/><path id="receptor" d="M650 246 L663 258 L650 270 L638 258 Z"/><circle id="boundLigand" cx="663" cy="258" r="5.5" fill="#f72585" stroke="#ad1457" stroke-width="1.5" opacity="0"/>
 <text class="tinyLabel" x="672" y="292">Receptor mineralocorticoide</text>
 <!-- Ribossomo e mRNA -->
 <path id="mrnaLine" d="M506 216 C495 242 510 255 499 279" fill="none" stroke="#00a878" stroke-width="3" stroke-dasharray="4 4" opacity=".3"/>
 <ellipse id="ribosome" cx="500" cy="302" rx="30" ry="18" fill="#f0c987" stroke="#bd8433" stroke-width="2"/><path d="M475 301 Q500 284 525 301 Q500 319 475 301" fill="none" stroke="#bd8433" stroke-width="2"/>
 <text class="tinyLabel" x="500" y="333">Ribossomo</text>
 <!-- Canais apicais: um já existente e um novo -->
 <rect x="251" y="350" width="25" height="54" rx="8" class="protein"/><path id="gateA" d="M251 369 L276 369"/><path id="gateB" d="M251 383 L276 383"/>
 <text class="tinyLabel" x="315" y="365" text-anchor="start">Canal de Na⁺</text><text class="tinyLabel" x="315" y="380" text-anchor="start">(ENaC esquemático)</text>
 <g id="newChannel"><rect x="251" y="416" width="25" height="48" rx="8" class="protein"/><path d="M251 433 L276 433 M251 447 L276 447" stroke="#075985" stroke-width="3"/><text class="tinyLabel" x="315" y="435" text-anchor="start">ENaC adicional</text></g>
 <!-- Bomba Na/K basolateral -->
 <g id="existingPump"><rect x="744" y="345" width="35" height="75" rx="12" class="pump"/><path d="M752 364 L771 364 M752 378 L771 378 M752 392 L771 392" stroke="#075985" stroke-width="2.4"/><text class="tinyLabel" x="739" y="438" text-anchor="end">Na⁺/K⁺-ATPase</text></g>
 <g id="newPump"><rect x="744" y="430" width="35" height="53" rx="12" class="pump"/><path d="M752 445 L771 445 M752 456 L771 456" stroke="#075985" stroke-width="2.4"/><text class="tinyLabel" x="711" y="496">Na⁺/K⁺-ATPase</text></g>
 <circle id="atpFlash" cx="807" cy="373" r="18"/><text x="807" y="377" class="smallLabel" id="atpLabel">ATP → ADP</text>
 <!-- Fluxos visíveis de referência -->
 <path id="naFlux" d="M70 378 C125 378 194 378 291 378" fill="none" stroke="#0077ff" stroke-width="3" marker-end="url(#arrBlue)"/>
 <text class="smallLabel" x="140" y="356" fill="#0077ff">Na⁺ entra na célula</text>
 <path id="naCytosolFlow" d="M280 388 C380 388 500 388 742 388" fill="none" stroke="#0077ff" stroke-width="3" stroke-dasharray="5 6" opacity=".58"/>
 <path d="M690 350 C750 320 830 318 920 318" fill="none" stroke="#0077ff" stroke-width="3" marker-end="url(#arrBlue)"/>
 <text class="smallLabel" x="850" y="300" fill="#0077ff">3 Na⁺ → sangue</text>
 <path d="M940 415 C875 415 830 405 785 392" fill="none" stroke="#7b2cbf" stroke-width="3" marker-end="url(#arrPurple)"/>
 <text class="smallLabel" x="900" y="440" fill="#7b2cbf">2 K⁺ entram</text>
 <path id="kFlux" d="M320 466 C270 466 218 460 168 460" fill="none" stroke="#f94144" stroke-width="3" marker-end="url(#arrRed)"/>
 <text class="smallLabel" x="135" y="442" fill="#f94144">K⁺ sai para o lúmen</text>
 <!-- Caminhos nomeados usados por getPointAtLength -->
 <g id="motionPaths" fill="none" stroke="transparent" stroke-width="1">
   <path id="pathBloodToMR" d="M980 155 C920 155 850 165 790 198 C745 222 700 240 666 257"/>
   <path id="pathMRToNucleus" d="M650 258 C610 248 600 218 574 194 C557 178 543 170 534 163"/>
   <path id="pathMRNucleusToRibosome" d="M505 211 C500 235 498 259 500 283"/>
   <path id="pathVesicleApical" d="M470 305 C423 323 370 344 278 436"/>
   <path id="pathVesicleBasolateral" d="M525 307 C590 326 672 368 758 456"/>
   <path id="pathNaApical" d="M78 378 C135 378 195 378 272 378"/>
   <path id="pathNaCytosol" d="M272 378 C360 378 450 386 535 386 S675 383 744 382"/>
   <path id="pathNaPumpOut" d="M778 367 C820 358 858 328 963 318"/>
   <path id="pathKPumpIn" d="M968 414 C900 414 840 404 778 389"/>
   <path id="pathKCytosol" d="M778 389 C765 395 752 400 744 405 C675 424 590 444 500 450 S350 458 278 460"/>
   <path id="pathKApical" d="M278 460 C235 460 194 460 155 460"/>
 </g>
 <g id="particles"></g>
 <text class="tinyLabel" x="550" y="545">A bomba Na⁺/K⁺-ATPase: 3 Na⁺ para fora + 2 K⁺ para dentro por ATP.</text>
</svg>
<div class="legend"><span class="key"><i class="dot" style="background:#f72585"></i>Aldosterona (esteroide)</span><span class="key"><i class="dot" style="background:#0077ff"></i>Na⁺</span><span class="key"><i class="dot" style="background:#7b2cbf"></i>K⁺ entrando pela bomba</span><span class="key"><i class="dot" style="background:#f94144"></i>K⁺ secretado ao lúmen</span><span class="key"><i class="dot" style="background:#00a878"></i>RNAm / síntese</span></div>
<div class="foot">Esquema qualitativo: as partículas mostram o sentido dos fluxos, não quantidades moleculares reais.</div>
</div>
<script>
const D = __CELL_DATA__;
const T=D.t, ALD=D.Ald, E=D.E, K=D.K;
const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const layer=document.getElementById('particles');
const parts=[]; const MAX_PARTICLES=150;
const stageTexts={
  1:'1. A aldosterona é lipossolúvel: atravessa a membrana basolateral e liga-se ao receptor mineralocorticoide dentro da célula. Não entra por um canal iônico.',
  2:'2. O complexo aldosterona–receptor atua no núcleo e regula a transcrição de genes que participam do transporte de Na⁺ e K⁺.',
  3:'3. O RNAm é traduzido nos ribossomos. Proteínas recém-sintetizadas são direcionadas à membrana apical (canais) e à basolateral (bombas).',
  4:'4. A aldosterona pode aumentar a disponibilidade e a atividade de proteínas de transporte. A abertura do canal e a inserção de proteínas são ilustrativas; apenas E representa um estado dinâmico próprio no modelo.',
  5:'5. O Na⁺ entra do lúmen por canais apicais e sai para o sangue pela Na⁺/K⁺-ATPase. Por ciclo, a bomba envia 3 Na⁺ para fora e traz 2 K⁺; parte do K⁺ pode seguir pelo citosol e sair para o lúmen.'
};
let simT=T[0], lastFrame=null, lastSimT=T[0], playing=false, mode='step', currentStage=1, channels=Math.max(0.05,D.baseline.E_channels||1);
let naPool=9, naInTransit=0; const NA_POOL_SETPOINT=9; const MAX_NA_POOL=60;
const tMin=T[0],tMax=T[T.length-1];
// E é estado dinâmico das EDOs; não há um segundo filtro apenas na animação.
function sampleChannelDensity(t){return sample(E,t);}
const timeSeek=document.getElementById('timeSeek'); timeSeek.min=String(tMin);timeSeek.max=String(tMax);timeSeek.value=String(simT);
document.getElementById('regimeLabel').textContent=D.label+' · animação vinculada ao regime selecionado';
function clamp(v,a,b){return Math.min(b,Math.max(a,v));}
function sample(arr,t){if(t<=T[0])return arr[0];if(t>=T[T.length-1])return arr[arr.length-1];let lo=0,hi=T.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(T[m]<=t)lo=m;else hi=m;}const f=(t-T[lo])/((T[hi]-T[lo])||1);return arr[lo]+(arr[hi]-arr[lo])*f;}
function stageMessage(n){const box=document.getElementById('stageText');box.textContent=stageTexts[n];const stageColors=['#f72585','#7b2cbf','#00a878','#0077ff','#f94144'];box.style.borderLeftColor=stageColors[n-1]||'#0077ff';document.querySelectorAll('.stage').forEach(b=>b.classList.toggle('on',Number(b.dataset.stage)===n));}
function spawn(pathId,speed,color,onEnd=null,r=4){if(reducedMotion)return false;if(parts.length>=MAX_PARTICLES)return false;const path=document.getElementById(pathId);if(!path)return false;let len;try{len=path.getTotalLength();}catch(e){return false;}const el=document.createElementNS('http://www.w3.org/2000/svg','circle');el.setAttribute('r',r);el.setAttribute('fill',color);el.setAttribute('class','particle');el.setAttribute('stroke','#fff');el.setAttribute('stroke-width','1.2');layer.appendChild(el);const point=path.getPointAtLength(0);el.setAttribute('cx',point.x);el.setAttribute('cy',point.y);parts.push({el,path,len,s:0,speed,onEnd});return true;}
function stepParticles(dt){for(let i=parts.length-1;i>=0;i--){const p=parts[i];p.s+=p.speed*dt;if(p.s>=p.len){p.el.remove();parts.splice(i,1);if(p.onEnd)p.onEnd();continue;}const pt=p.path.getPointAtLength(p.s);p.el.setAttribute('cx',pt.x);p.el.setAttribute('cy',pt.y);}}
function clearParticles(){while(parts.length){parts.pop().el.remove();}naInTransit=0;naPool=NA_POOL_SETPOINT;}
function highlightBinding(){document.getElementById('receptorHalo').style.opacity='1';document.getElementById('boundLigand').style.opacity='1';}
function startReceptorToNucleus(continueCascade=true){document.getElementById('nucleus').classList.add('glow');spawn('pathMRToNucleus',110,'#f97316',()=>{document.getElementById('nucleus').classList.remove('glow');if(continueCascade)spawn('pathMRNucleusToRibosome',105,'#00a878',()=>{addVesicles();},4);},4);}
function addVesicles(){spawn('pathVesicleApical',105,'#0891b2',()=>document.getElementById('newChannel').classList.add('visible'),7);spawn('pathVesicleBasolateral',105,'#22d3ee',()=>document.getElementById('newPump').classList.add('visible'),7);}
function flashATP(){const el=document.getElementById('atpFlash');el.classList.add('on');window.setTimeout(()=>el.classList.remove('on'),500);}
function spawnNaEntry(speed=105){
  // Reserva o Na⁺ em trânsito para que o limite do pool não descarte íons já entrando.
  if(naPool+naInTransit>=MAX_NA_POOL)return false;
  const accepted=spawn('pathNaApical',speed,'#0077ff',()=>{
    const continued=spawn('pathNaCytosol',speed,'#0077ff',()=>{
      naInTransit=Math.max(0,naInTransit-1);
      naPool+=1;
    },3.5);
    if(!continued)naInTransit=Math.max(0,naInTransit-1);
  },3.5);
  if(accepted)naInTransit+=1;
  return accepted;
}
function spawnKCycleChain(speed=105){
  // Cada K⁺ segue sangue → bomba → citosol → lúmen; não há secreção independente.
  return spawn('pathKPumpIn',speed,'#7b2cbf',()=>{
    spawn('pathKCytosol',speed,'#7b2cbf',()=>{
      spawn('pathKApical',speed,'#f94144',null,3.8);
    },4.0);
  },4.2);
}
function pumpCycle(){
  // Um ciclo é atômico: 3 Na⁺ consumidos e 2 K⁺ transportados por ATP.
  if(naPool<3 || parts.length+5>MAX_PARTICLES)return false;
  naPool-=3;flashATP();
  for(let i=0;i<3;i++){if(!spawn('pathNaPumpOut',105,'#0077ff',null,3.8))return false;}
  for(let i=0;i<2;i++){if(!spawnKCycleChain(105))return false;}
  return true;
}
function stageAction(n){mode='step';playing=false;clearParticles();currentStage=n;stageMessage(n);document.getElementById('modeStep').classList.add('primary');document.getElementById('modeContinuous').classList.remove('primary');document.getElementById('playCell').textContent='▶ Reproduzir tempo';
  document.getElementById('newChannel').classList.toggle('visible',n>=4);document.getElementById('newPump').classList.toggle('visible',n>=4);
  document.getElementById('gateA').style.transform=n>=4?'translateY(-5px)':'';document.getElementById('gateB').style.transform=n>=4?'translateY(5px)':'';
  document.getElementById('nucleus').classList.remove('glow');
  if(n>=2){highlightBinding();}else{document.getElementById('receptorHalo').style.opacity='0';document.getElementById('boundLigand').style.opacity='0';}
  if(reducedMotion){
    document.getElementById('newChannel').classList.toggle('visible',n>=3);
    document.getElementById('newPump').classList.toggle('visible',n>=3);
    document.getElementById('nucleus').classList.toggle('glow',n===2||n===3);
    return;
  }
  if(n===1){spawn('pathBloodToMR',120,'#f72585',highlightBinding,5);}
  if(n===2){startReceptorToNucleus(false);}
  if(n===3){document.getElementById('newChannel').classList.remove('visible');document.getElementById('newPump').classList.remove('visible');document.getElementById('nucleus').classList.add('glow');spawn('pathMRNucleusToRibosome',105,'#00a878',()=>{document.getElementById('nucleus').classList.remove('glow');addVesicles();},4);}
  if(n===5){naPool=NA_POOL_SETPOINT;for(let i=0;i<4;i++)spawnNaEntry(100);pumpCycle();}
}
function startContinuous(){mode='continuous';playing=true;clearParticles();lastSimT=simT;document.getElementById('modeStep').classList.remove('primary');document.getElementById('modeContinuous').classList.add('primary');document.getElementById('playCell').textContent='⏸ Pausar tempo';stageMessage(1);if(reducedMotion){playing=false;document.getElementById('playCell').textContent='▶ Reproduzir tempo';document.getElementById('stageText').textContent='Movimento contínuo pausado porque o dispositivo indica preferência por movimento reduzido. Use os botões 1–5 para percorrer as etapas.';}}
function poisson(rate,dt){return Math.random() < 1-Math.exp(-Math.max(0,rate)*Math.max(0,dt));}
function continuousEvents(dtReal,dtModel,ald,k){channels=clamp(sampleChannelDensity(simT),0.05,4.0);const baselineE=Math.max(0.05,D.baseline.E_channels||1);const mod=clamp(channels/baselineE-1,-1,1);document.getElementById('gateA').style.transform='translateY('+(-4*mod)+'px)';document.getElementById('gateB').style.transform='translateY('+(4*mod)+'px)';const flowWidth=(2.4+Math.min(channels,3.8)).toFixed(1);const flowOpacity=clamp(0.38+0.16*channels,0.38,1).toFixed(2);for(const flowId of ['naFlux','naCytosolFlow']){const flowEl=document.getElementById(flowId);if(flowEl){flowEl.style.strokeWidth=flowWidth;flowEl.style.opacity=flowOpacity;}}if(reducedMotion)return;
  if(poisson(0.34*clamp(ald,0.05,3.0),dtReal)){spawn('pathBloodToMR',115,'#f72585',()=>{highlightBinding();startReceptorToNucleus(true);window.setTimeout(()=>{document.getElementById('receptorHalo').style.opacity='0.2';document.getElementById('boundLigand').style.opacity='0';},500);},4);}
  // Canal apical controla entradas. Cada partícula segue para o citosol e incrementa o pool ao chegar.
  if(naPool<MAX_NA_POOL && poisson(0.55*channels,dtReal))spawnNaEntry(105);
  // A bomba só pode ciclar se houver 3 Na⁺ disponíveis. A taxa depende do pool intracelular.
  const poolFactor=clamp(naPool/NA_POOL_SETPOINT,0,1.5);
  const pumpRate=(0.55*channels/3.0)*poolFactor*clamp(k,0.05,3.0);
  if(naPool>=3 && poisson(pumpRate,dtReal))pumpCycle();
}
function updateReadouts(){const ald=sample(ALD,simT),k=sample(K,simT);document.getElementById('timeReadout').textContent=simT<0?'basal':(D.is_chronic?(simT/1440).toFixed(2)+' dias':simT.toFixed(1)+' min');document.getElementById('valuesReadout').textContent='Ald = '+ald.toFixed(2)+' · E canais = '+channels.toFixed(2)+' · K⁺ = '+k.toFixed(2)+' · Na⁺ disponível à bomba (unid. didáticas) = '+naPool;timeSeek.value=String(simT);}
document.querySelectorAll('.stage').forEach(b=>b.addEventListener('click',()=>stageAction(Number(b.dataset.stage))));
document.getElementById('modeStep').addEventListener('click',()=>{mode='step';playing=false;clearParticles();document.getElementById('modeStep').classList.add('primary');document.getElementById('modeContinuous').classList.remove('primary');document.getElementById('playCell').textContent='▶ Reproduzir tempo';stageMessage(currentStage);});
document.getElementById('modeContinuous').addEventListener('click',startContinuous);
document.getElementById('playCell').addEventListener('click',()=>{if(mode!=='continuous')startContinuous();else{playing=!playing;document.getElementById('playCell').textContent=playing?'⏸ Pausar tempo':'▶ Reproduzir tempo';lastSimT=simT;}});
document.getElementById('resetCell').addEventListener('click',()=>{clearParticles();naPool=NA_POOL_SETPOINT;naInTransit=0;document.getElementById('receptorHalo').style.opacity='0';document.getElementById('boundLigand').style.opacity='0';document.getElementById('newChannel').classList.remove('visible');document.getElementById('newPump').classList.remove('visible');document.getElementById('gateA').style.transform='';document.getElementById('gateB').style.transform='';document.getElementById('nucleus').classList.remove('glow');channels=Math.max(0.05,D.baseline.E_channels||1);stageAction(1);});
timeSeek.addEventListener('input',()=>{simT=Number(timeSeek.value);lastSimT=simT;channels=clamp(sampleChannelDensity(simT),0.05,4.0);updateReadouts();});
let cellVisible=false,cellFrameQueued=false;
function scheduleCellFrame(){if(cellVisible&&!cellFrameQueued){cellFrameQueued=true;requestAnimationFrame(frame);}}
function frame(ts){cellFrameQueued=false;if(!cellVisible)return;if(lastFrame===null)lastFrame=ts;let dtReal=Math.min((ts-lastFrame)/1000,0.05);lastFrame=ts;let dtModel=0;if(playing){const factor=D.is_chronic?96.0:1.0;simT+=dtReal*factor;if(simT>=tMax){simT=tMax;playing=false;document.getElementById('playCell').textContent='▶ Reproduzir tempo';}dtModel=Math.max(0,simT-lastSimT);const ald=sample(ALD,simT),k=sample(K,simT);continuousEvents(dtReal,dtModel,ald,k);}lastSimT=simT;stepParticles(dtReal);updateReadouts();scheduleCellFrame();}
const cellObserver=new IntersectionObserver(entries=>{const visible=Boolean(entries[0]&&entries[0].isIntersecting);if(visible!==cellVisible){cellVisible=visible;lastFrame=null;}if(cellVisible)scheduleCellFrame();});
cellObserver.observe(document.body);
if(reducedMotion){document.getElementById('stageText').textContent='Preferência por movimento reduzido detectada. Use os botões 1–5 para explorar as etapas; a simulação contínua começa pausada.';stageMessage(1);}
else{stageAction(1);}
</script></body></html>
"""

cell_source = data_b if (diagram_regime == "B" and data_b is not None) else data
cell_data = {
    "t": cell_source["t"],
    "Ald": cell_source["Ald"],
    "E": cell_source["E"],
    "K": cell_source["K"],
    "baseline": cell_source["baseline"],
    "label": label_b if (diagram_regime == "B" and data_b is not None) else label_a,
    "is_chronic": bool(is_chronic),
}
cell_payload = json.dumps(cell_data, separators=(",", ":"))
html = TEMPLATE.replace("__DATA__", payload)
cell_html = CELL_TEMPLATE.replace("__CELL_DATA__", cell_payload)
model_tab, aldosterone_tab = st.tabs(["📈 SRAA e comparação de regimes", "🧫 Efeitos celulares da aldosterona"])
with model_tab:
    st.iframe(html, width="stretch", height=3750 if compare_enabled else 3650, alt="Diagrama e gráficos animados do simulador didático do SRAA")
with aldosterone_tab:
    st.markdown(
        "A animação abaixo reutiliza as séries de **aldosterona (Ald)** e **potássio (K⁺)** do regime selecionado. "
        "O mecanismo molecular é uma ilustração; a expressão funcional E tem agora um atraso explícito nas EDOs, com constante de tempo didática de 120 min."
    )
    st.iframe(cell_html, width="stretch", height=940, alt="Animação didática da ação da aldosterona numa célula epitelial do néfron distal")

if compare_enabled and data_b is not None:
    st.subheader("Resumo quantitativo da comparação")
    st.caption(
        "Cada sequência mostra pré-perturbação → ponto com maior desvio percentual absoluto do próprio basal → final. "
        "As porcentagens entre parênteses são relativas ao basal específico de A ou B. Δ final = B − A nos valores exibidos. "
        "O tempo de recuperação é o primeiro instante em que a PAM alcança 95% do basal próprio; é um critério descritivo, não clínico."
    )
    summary_variables = [
        ("PAM", "PAM", "mmHg"),
        ("GFR", "TFG estimada", "mL/min/1,73 m²"),
        ("R", "Renina", "índice"),
        ("A2", "Angiotensina II", "índice"),
        ("AT1", "Atividade efetiva AT1", "índice"),
        ("AT2", "Atividade efetiva AT2", "índice"),
        ("Ald", "Aldosterona", "índice"),
        ("BK", "Bradicinina", "índice"),
        ("V", "Volume relativo", "índice"),
        ("Na", "Sódio corporal", "índice"),
        ("K", "Potássio plasmático", "índice"),
    ]
    rows = []
    for key, label, unit in summary_variables:
        rows.append({
            "Variável": label,
            "Unidade": unit,
            f"A — {label_a}": sequence_summary(data, key),
            f"B — {label_b}": sequence_summary(data_b, key),
            "Δ final (B − A)": final_delta_text(data, data_b, key),
        })
    st.dataframe(rows, hide_index=True, width="stretch")

    rec_a = format_recovery(pam_recovery_time(data), is_chronic, t_end)
    rec_b = format_recovery(pam_recovery_time(data_b), is_chronic, t_end)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Tempo de A até 95% da própria PAM basal", rec_a)
    with col_b:
        st.metric("Tempo de B até 95% da própria PAM basal", rec_b)

    with st.expander("Teste automático das assinaturas farmacológicas IECA × BRA", expanded=False):
        st.write(
            "Executa um cenário padronizado de 20% de redução inicial do volume relativo por 60 minutos. "
            "O teste verifica as assinaturas didáticas: IECA → Ang II < 1 e bradicinina > 1; "
            "BRA → Ang II > 1, bradicinina ≈ 1 e AT2 > 1."
        )
        if st.button("Executar teste IECA × BRA", key="run_signature_test"):
            with st.spinner("Executando os dois cenários de referência..."):
                test_ieca = simulate(
                    drop=0.20, t_end=60.0, is_chronic=False,
                    drugs={"ace": 1.0, "arb": 0.0, "dri": 0.0}, params=PARAMS,
                )
                test_bra = simulate(
                    drop=0.20, t_end=60.0, is_chronic=False,
                    drugs={"ace": 0.0, "arb": 1.0, "dri": 0.0}, params=PARAMS,
                )
            same_grid = test_ieca["t"] == test_bra["t"]
            checks = {
                "Vetor de tempo idêntico": same_grid,
                "IECA: Ang II < 1 aos 60 min": test_ieca["A2"][-1] < 1.0,
                "IECA: bradicinina > 1 aos 60 min": test_ieca["BK"][-1] > 1.0,
                "BRA: Ang II > 1 aos 60 min": test_bra["A2"][-1] > 1.0,
                "BRA: bradicinina ≈ 1 aos 60 min": abs(test_bra["BK"][-1] - 1.0) < 1e-3,
                "BRA: AT2 > 1 aos 60 min": test_bra["AT2"][-1] > 1.0,
            }
            st.dataframe(
                [{"Verificação": name, "Resultado": "OK" if passed else "FALHOU"} for name, passed in checks.items()],
                hide_index=True, width="stretch",
            )
            if all(checks.values()):
                st.success("Todas as assinaturas didáticas verificadas neste cenário passaram.")
            else:
                st.warning("Uma ou mais assinaturas não passaram. Reveja os parâmetros antes de interpretar o contraste.")

SHOW_INTERNAL_DIAGNOSTICS = False  # Mude para True para exibir controles internos de validação.
if SHOW_INTERNAL_DIAGNOSTICS:
    if st.button("Executar suíte ampliada de autoverificação", key="run_full_validation"):
        with st.spinner("Executando testes farmacológicos, monotonicidade e recuperação crônica..."):
            none_60 = simulate(drop=0.20, t_end=60.0, is_chronic=False, drugs={"ace":0.0,"arb":0.0,"dri":0.0}, params=PARAMS)
            dri_max = simulate(drop=0.20, t_end=60.0, is_chronic=False, drugs={"ace":0.0,"arb":0.0,"dri":1.0}, params=PARAMS)
            ace_levels = [simulate(drop=0.20,t_end=60.0,is_chronic=False,drugs={"ace":v,"arb":0.0,"dri":0.0},params=PARAMS) for v in (0.0,0.5,1.0)]
            bra_levels = [simulate(drop=0.20,t_end=60.0,is_chronic=False,drugs={"ace":0.0,"arb":v,"dri":0.0},params=PARAMS) for v in (0.0,0.5,1.0)]
            dri_levels = [simulate(drop=0.20,t_end=60.0,is_chronic=False,drugs={"ace":0.0,"arb":0.0,"dri":v},params=PARAMS) for v in (0.0,0.5,1.0)]
            recovery_3d = simulate(drop=0.20,t_end=3*1440.0,is_chronic=True,drugs={"ace":0.0,"arb":0.0,"dri":0.0},params=PARAMS)
        checks_full = {
            "Grade temporal idêntica entre os níveis de dose": all(q["t"] == ace_levels[0]["t"] for q in ace_levels) and all(q["t"] == bra_levels[0]["t"] for q in bra_levels) and all(q["t"] == dri_levels[0]["t"] for q in dri_levels),
            "Inibidor da renina reduz a atividade de renina aos 60 min": dri_max["R"][-1] < none_60["R"][-1],
            "IECA: mais intensidade não eleva Ang II": ace_levels[0]["A2"][-1] >= ace_levels[1]["A2"][-1] >= ace_levels[2]["A2"][-1],
            "IECA: mais intensidade não reduz bradicinina": ace_levels[0]["BK"][-1] <= ace_levels[1]["BK"][-1] <= ace_levels[2]["BK"][-1],
            "BRA: mais intensidade não eleva AT1 efetivo": bra_levels[0]["AT1"][-1] >= bra_levels[1]["AT1"][-1] >= bra_levels[2]["AT1"][-1],
            "Inibidor da renina: mais intensidade não eleva renina": dri_levels[0]["R"][-1] >= dri_levels[1]["R"][-1] >= dri_levels[2]["R"][-1],
            "Resposta de expressão E atrasa a aldosterona (não é instantânea)": np.max(np.abs(np.asarray(ace_levels[2]["E"]) - np.asarray(ace_levels[2]["Ald"]))) > 1e-3,
            "Recuperação de PAM ≥95% do basal em até 3 dias (sem fármaco, queda 20%)": pam_recovery_time(recovery_3d) is not None,
        }
        st.dataframe([{"Verificação": name, "Resultado": "OK" if ok else "FALHOU"} for name,ok in checks_full.items()], hide_index=True, width="stretch")
        if all(checks_full.values()):
            st.success("Todas as verificações da suíte ampliada passaram para estes cenários didáticos.")
        else:
            st.warning("Há verificações que não passaram. Isso não deve ser ocultado: reveja os mecanismos/parâmetros e interprete as curvas com cautela.")

# =====================================================================
# 4. DIAGNÓSTICOS MATEMÁTICOS E LIMITAÇÕES
# =====================================================================
if SHOW_INTERNAL_DIAGNOSTICS:
    with st.expander("Verificar o equilíbrio numérico", expanded=False):
        st.write(f"Convergência do cálculo do equilíbrio: **{data['equilibrium_ok']}**")
        st.write(f"Maior resíduo normalizado: **{data['equilibrium_residual']:.3e}**")
        st.write("Estado estacionário usado antes da perturbação, já considerando os fármacos selecionados:")
        baseline_rows = [{"Variável": name, "Índice basal calculado": round(value, 4)} for name, value in data["baseline"].items()]
        st.dataframe(baseline_rows, hide_index=True, width="stretch")

st.markdown("---")
st.markdown(
    "**Como interpretar o simulador**\n\n"
    "- O teste de equilíbrio sem fármacos deve manter PAM em 90 mmHg e os índices em 1, sem deriva.\n"
    "- Os índices de renina, angiotensinas, receptores, aldosterona, ADH, bradicinina, sódio, potássio e volume são relativos; não representam concentrações clínicas.\n"
    "- A perturbação é aplicada uma vez como redução inicial de volume relativo; o sistema pode compensá-la ao longo do tempo. Isso não equivale a uma hemorragia ou choque clínico real.\n- Na fase crônica, os balanços de sódio, potássio e volume usam constantes de tempo lentas, definidas para fins didáticos.\n"
    "- A TFG é uma estimativa fenomenológica simples. O modelo não inclui a mecânica completa do néfron, o transporte segmentar, a osmolaridade plasmática, a farmacocinética nem variabilidade individual.\n"
    "- Os avisos são indicadores pedagógicos e não devem ser usados para diagnosticar ou orientar tratamentos.\n"
)
