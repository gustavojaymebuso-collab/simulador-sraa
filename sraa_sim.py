# -*- coding: utf-8 -*-
"""
Simulador Fisiológico Imersivo do SRAA (Versão Clínica Avançada).
Inclui: TFG corrigida, AT2, Potássio, Escalas Temporais (Aguda/Crônica) e Alertas Clínicos.
Execute com:  streamlit run sraa_sim.py
"""
import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Simulador SRAA Clínico", layout="wide")

# =====================================================================
# 1. MODELO MATEMÁTICO AVANÇADO
# =====================================================================
PAM0 = 90.0      
T_PRE = 1.0      

PARAMS = dict(
    # SRAA (Escala Rápida)
    k_ren=0.25, k_ren_deg=0.20, k_a1=0.80, k_a1_deg=0.60, 
    k_a2=0.65, k_a2_deg=0.50, k_at1=0.90, k_at1_deg=0.40, 
    k_at2=0.60, k_at2_deg=0.40,
    k_bk=0.50, k_bk_deg=0.50,
    k_adh=0.10, k_adh_deg=0.10,
    # Escala Lenta (Aldosterona, Sódio, Volume, Potássio)
    k_ald=0.08, k_ald_deg=0.08, # Meia-vida de horas
    k_na=0.02, k_k=0.05, k_vol=0.01,
    k_symp=0.50,
    # Ganhos
    g_ren_press=2.5, g_ren_symp=1.5, g_ren_na=1.0, g_ren_a2=2.0,
    g_ald_at1=1.0, g_adh_at1=0.5, g_na_ald=1.0, g_na_press=1.5,
    g_pam_vol=1.5, g_pam_at1=0.6, g_pam_at2=0.4, g_pam_bk=0.4
)

def simular(drop=0.20, t_end=2.0, is_chronic=False, drugs=None):
    p = PARAMS.copy()
    if drugs is None: drugs = {"ace": 0, "arb": 0, "dri": 0}
    
    # Ajustar parâmetros farmacológicos
    p['k_a2'] *= (1 - drugs['ace'] * 0.95)  
    p['k_at1'] *= (1 - drugs['arb'] * 0.95) 
    p['k_ren'] *= (1 - drugs['dri'] * 0.95) 

    def _disturbio(t, d, width=0.10):
        return 1.0 - d / (1.0 + np.exp(-t / width))

    def rhs(t, y):
        R, A1, A2, AT1, AT2, Ald, ADH, BK, Na, K, V, S = y
        
        V = max(V, 0.3); A2 = max(A2, 0.05); Na = max(Na, 0.3); K = max(K, 0.2)
        D = _disturbio(t, drop)
        
        # Hemodinâmica: AT1 contrai, AT2 e BK dilatam
        pam = PAM0 * (V ** p['g_pam_vol']) * (AT1 ** p['g_pam_at1']) * (AT2 ** (-p['g_pam_at2'])) * (BK ** (-p['g_pam_bk'])) * D
        
        # Correção da TFG: Ang II contrai a arteríola EFERENTE, preservando a TFG
        gfr = pam * (1.0 + 0.25 * (A2 - 1.0))
        
        s_alvo = np.clip(1.0 + 3.0 * (PAM0 - pam) / PAM0, 0.3, 4.0)
        drive_ren = np.clip((1.0 + p['g_ren_press']*(PAM0-pam)/PAM0 + p['g_ren_symp']*(S-1.0) + p['g_ren_na']*(1.0-Na)) / (1.0 + p['g_ren_a2']*A2), 0.01, 5.0)
        
        # EDOs
        dR   = p['k_ren'] * drive_ren - p['k_ren_deg'] * R
        dA1  = p['k_a1'] * R - p['k_a1_deg'] * A1
        dA2  = p['k_a2'] * A1 - p['k_a2_deg'] * A2
        dAT1 = p['k_at1'] * A2 - p['k_at1_deg'] * AT1
        dAT2 = p['k_at2'] * A2 - p['k_at2_deg'] * AT2 # Sobra Ang II ativa AT2
        dAld = p['k_ald'] * AT1 - p['k_ald_deg'] * Ald # Escala lenta
        dADH = p['k_adh'] * (1.0 + p['g_adh_at1']*(AT1-1.0)) - p['k_adh_deg'] * ADH
        
        d_BK_deg = p['k_bk_deg'] * (1 - drugs['ace'] * 0.90)
        dBK  = p['k_bk'] - d_BK_deg * BK
        
        # Sódio, Potássio e Volume (Escala lenta)
        # Aldosterona retém Na+ e excreta K+
        dNa  = p['k_na'] * (1.0 - p['g_na_press']*(pam/PAM0 - 1.0) + p['g_na_ald']*(Ald-1.0)) - 0.01 * (Na - 1.0)
        dK   = p['k_k'] * (1.0 - Ald) - 0.02 * (K - 1.0) - 0.01 * (pam/PAM0 - 1.0) # Ald ↑ excreta K ↓
        dV   = p['k_vol'] * (ADH - 1.0) + 0.005 * (Na - 1.0) - 0.01 * (pam/PAM0 - 1.0)
        
        dS   = p['k_symp'] * (s_alvo - S)
        return [dR, dA1, dA2, dAT1, dAT2, dAld, dADH, dBK, dNa, dK, dV, dS]

    # Se for crônico, integramos por dias. Se agudo, por horas.
    tev = np.linspace(-T_PRE, t_end, 1500)
    
    # Condições iniciais
    sol = solve_ivp(rhs, (-T_PRE, t_end), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    t_eval=tev, method="LSODA", rtol=1e-5, atol=1e-7)

    R, A1, A2, AT1, AT2, Ald, ADH, BK, Na, K, V, S = sol.y
    D = np.array([_disturbio(t, drop) for t in sol.t])
    pam = PAM0 * (V ** PARAMS['g_pam_vol']) * (AT1 ** PARAMS['g_pam_at1']) * (AT2 ** (-PARAMS['g_pam_at2'])) * (BK ** (-PARAMS['g_pam_bk'])) * D
    gfr = pam * (1.0 + 0.25 * (A2 - 1.0))

    r = lambda a: np.round(a, 4).tolist()
    return {"t": r(sol.t), "PAM": r(pam), "GFR": r(gfr), "R": r(R), "A2": r(A2), 
            "AT1": r(AT1), "AT2": r(AT2), "Ald": r(Ald), "ADH": r(ADH), "BK": r(BK), 
            "Na": r(Na), "K": r(K), "V": r(V), "t_end": float(t_end), "is_chronic": is_chronic}

# =====================================================================
# 2. INTERFACE STREAMLIT
# =====================================================================
st.title("🫘 Simulador Clínico do SRAA")
st.caption("Análise hemodinâmica, farmacológica e eletrolítica avançada.")

with st.sidebar:
    st.header("1. Cenário e Tempo")
    fase = st.radio("Escala de Tempo", ["Fase Aguda (Minutos)", "Fase Crônica (Dias)"])
    is_chronic = (fase == "Fase Crônica (Dias)")
    
    queda = st.slider("Queda inicial da pressão (%)", 5, 40, 20, 1)
    if is_chronic:
        dur = st.slider("Duração da simulação (dias)", 1, 7, 3, 1)
        t_end = dur * 24.0
    else:
        dur = st.slider("Duração da simulação (minutos)", 15, 120, 60, 15)
        t_end = dur / 60.0
        
    st.header("2. Farmacologia")
    ace = st.slider("IECA (Captopril)", 0.0, 1.0, 0.0, 0.1)
    arb = st.slider("BRA (Losartana)", 0.0, 1.0, 0.0, 0.1)
    dri = st.slider("Inibidor da Renina", 0.0, 1.0, 0.0, 0.1)

drugs = {"ace": ace, "arb": arb, "dri": dri}
dados = simular(drop=queda / 100.0, t_end=t_end, is_chronic=is_chronic, drugs=drugs)
payload = json.dumps(dados)

# =====================================================================
# 3. TEMPLATE HTML/SVG/JS
# =====================================================================
TEMPLATE = r"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><style>
:root{--ink:#1b2b4b;--muted:#7a8ba3;--line:#e3e9f2;}
*{box-sizing:border-box;}
body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
.wrap{padding:6px 12px 4px;position:relative;}
.top{display:flex;align-items:center;gap:16px;margin-bottom:2px;}
.pam{font-size:30px;font-weight:800;color:#c0392b;line-height:1;min-width:170px;}
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
.charts{display:flex;gap:10px;margin-top:10px;}
.card{flex:1;background:#fbfcfe;border:1px solid var(--line);border-radius:12px;padding:6px 8px 2px;min-width:0;}
.card h4{margin:0 0 2px 4px;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);}
canvas{display:block;width:100%;height:auto;}
.bar{display:flex;align-items:center;gap:10px;margin-top:8px;}
button{border:1px solid #d3dceb;background:#fff;color:#22344f;font-weight:600;font-size:13px;padding:7px 14px;border-radius:9px;cursor:pointer;}
button:hover{background:#f2f6fd;}
button.pri{background:#1d6fb8;border-color:#1d6fb8;color:#fff;}
button.pri:hover{background:#185a95;}
.speed{display:flex;gap:6px;}
.speed button{padding:6px 10px;}
.speed button.on{background:#e8f0fb;border-color:#1d6fb8;color:#1d6fb8;}
.status{font-size:13px;color:#4a5b78;flex:1;text-align:right;font-weight:600;}

/* Pop-up Alertas */
#alertBox{position:absolute;top:70px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:100;pointer-events:none;}
.alert{background:#ffe5e5;border-left:6px solid #d32f2f;color:#b71c1c;padding:10px 16px;border-radius:6px;font-weight:700;font-size:14px;box-shadow:0 4px 12px rgba(0,0,0,0.15);animation:slideIn 0.3s ease-out;display:none;}
.alert.warning{background:#fff3e0;border-color:#f57c00;color:#e65100;}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
</style></head><body><div class="wrap">

<div id="alertBox">
  <div id="alertCough" class="alert warning">⚠️ Tosse seca: Bradicinina acumulada (efeito IECA)</div>
  <div id="alertK" class="alert">🚨 Hipercalemia detectada: Risco de arritmia!</div>
</div>

<div class="top">
  <div class="pam"><span id="pamv">90</span> <span style="font-size:14px;color:#7a8ba3">mmHg</span>
    <small>Pressão arterial média</small></div>
  <div class="gw"><div class="gauge"><i id="gfill"></i></div></div>
  <div class="clock" id="clock">t = 0.0 min</div>
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
    <text class="ttl" x="100" y="56">Rim</text><text class="sub" x="100" y="73">cél. justaglomerulares</text>
    <text class="val" x="100" y="98" id="vR" fill="#7b52ab">renina 1.00</text></g>
  <g id="bxF"><rect class="bg" x="240" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="310" y="56">Fígado</text><text class="sub" x="310" y="73">angiotensinogênio</text>
    <text class="val" x="310" y="98" id="vA1" fill="#2a9d8f">Ang I 1.00</text></g>
  <g id="bxP"><rect class="bg" x="440" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="510" y="56">Pulmão</text><text class="sub" x="510" y="73">ECA</text>
    <text class="val" x="510" y="98" id="vA2" fill="#e76f51">Ang II 1.00</text></g>
  <g id="bxA"><rect class="bg" x="640" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="710" y="56">Adrenal</text><text class="sub" x="710" y="73">zona glomerulosa</text>
    <text class="val" x="710" y="98" id="vAld" fill="#1d6fb8">aldosterona 1.00</text></g>
  <g id="bxH"><rect class="bg" x="840" y="28" width="140" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="910" y="56">Hipófise</text><text class="sub" x="910" y="73">neuro-hipófise</text>
    <text class="val" x="910" y="98" id="vADH" fill="#2ecc71">ADH 1.00</text></g>

  <g id="bxV"><rect class="bg" x="620" y="240" width="180" height="110" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="710" y="265" font-size="13">Vasos (Receptores)</text>
    <rect id="lumen" x="645" y="280" width="130" height="34" rx="10" fill="#f7d9d3" stroke="#e76f51" stroke-width="2"/>
    <text class="val" x="660" y="330" id="vAT1" fill="#e76f51">AT1 1.0</text>
    <text class="val" x="760" y="330" id="vAT2" fill="#f39c12">AT2 1.0</text></g>

  <g id="bxT"><rect class="bg" x="820" y="240" width="160" height="110" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="900" y="265" font-size="13">Túbulos (Na+/K+)</text>
    <rect x="835" y="280" width="130" height="26" rx="13" fill="#e6eefb" stroke="#c3d5ee"/>
    <g id="dots"></g>
    <text class="val" x="865" y="330" id="vNa" fill="#1d6fb8">Na+ 1.00</text>
    <text class="val" x="935" y="330" id="vK" fill="#d32f2f">K+ 1.00</text></g>
    
  <g id="bxK"><rect class="bg" x="440" y="150" width="140" height="70" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="510" y="175" font-size="13">Bradicinina</text>
    <text class="val" x="510" y="200" id="vBK" fill="#9b59b6">BK 1.00</text></g>
</svg>

<div class="charts">
  <div class="card"><h4>PAM & TFG (Filtração)</h4><canvas id="c1" width="600" height="260"></canvas></div>
  <div class="card"><h4>SRAA (Ang II, AT1, AT2)</h4><canvas id="c2" width="600" height="260"></canvas></div>
</div>
<div class="charts">
  <div class="card"><h4>Volume & Aldosterona</h4><canvas id="c3" width="600" height="260"></canvas></div>
  <div class="card"><h4>Eletrólitos (Bradicinina & K+)</h4><canvas id="c4" width="600" height="260"></canvas></div>
</div>

<div class="bar">
  <button id="btnPlay" class="pri">⏸ Pausar</button>
  <button id="btnReset">↺ Reiniciar</button>
  <div class="speed">
    <button data-s="0.5">0.5×</button><button data-s="1" class="on">1×</button>
    <button data-s="2">2×</button><button data-s="4">4×</button>
  </div>
  <div class="status" id="status"></div>
</div>
</div>

<script>
const D = __DATA__;
const T=D.t, PAM=D.PAM, GFR=D.GFR, RR=D.R, A2=D.A2, AT1=D.AT1, AT2=D.AT2, ALD=D.Ald, BK=D.BK, NA=D.Na, K=D.K, V=D.V;
const T0=T[0], T1=T[T.length-1];
const IS_CHRONIC = D.is_chronic;
const MIN_PER_SEC = IS_CHRONIC ? (24.0/10.0) : (1.0); // Crônico: 1s = 2.4h. Agudo: 1s = 1min

const lerp=(a,b,f)=>a+(b-a)*f;
function sample(arr,t){
  const n=T.length; if(t<=T[0])return arr[0]; if(t>=T[n-1])return arr[n-1];
  let lo=0,hi=n-1; while(hi-lo>1){const m=(lo+hi)>>1; if(T[m]<=t)lo=m; else hi=m;}
  const f=(t-T[lo])/((T[hi]-T[lo])||1); return lerp(arr[lo],arr[hi],f);
}

const mx={A2:Math.max(...A2),AT1:Math.max(...AT1),AT2:Math.max(...AT2),Ald:Math.max(...ALD),BK:Math.max(...BK),Na:Math.max(...NA),K:Math.max(...K)};
function nrm(v,m){ if(m<=1.02) return 0.30; return Math.min(1,Math.max(0,0.25+0.75*(v-1)/(m-1))); }
function h2r(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function mix(a,b,f){const A=h2r(a),B=h2r(b); return 'rgb('+Math.round(lerp(A[0],B[0],f))+','+Math.round(lerp(A[1],B[1],f))+','+Math.round(lerp(A[2],B[2],f))+')';}
function setBox(id,n,pale,sat){document.querySelector('#'+id+' .bg').setAttribute('fill',mix(pale,sat,n));}

function makeChart(cv,series,ymin,ymax,legend){
  const ctx=cv.getContext('2d'), W=cv.width, H=cv.height, L=50,R=15,Tp=25,B=30;
  const px=t=>L+(t-T0)/(T1-T0)*(W-L-R);
  const py=v=>Tp+(1-(v-ymin)/(ymax-ymin))*(H-Tp-B);
  const off=document.createElement('canvas'); off.width=W; off.height=H;
  const o=off.getContext('2d'); o.fillStyle='#fbfcfe'; o.fillRect(0,0,W,H);
  o.font='16px -apple-system,Segoe UI,Roboto,sans-serif';
  
  for(let i=0;i<=4;i++){
    const v=ymin+(ymax-ymin)*i/4, y=py(v);
    o.strokeStyle='#eef2f8'; o.lineWidth=1; o.beginPath();o.moveTo(L,y);o.lineTo(W-R,y);o.stroke();
    o.fillStyle='#9aabc2';o.textAlign='right';o.textBaseline='middle';o.fillText(v.toFixed(1),L-8,y);
  }
  for(let i=0;i<=4;i++){
    const t=T0+(T1-T0)*i/4,x=px(t);
    let lbl = IS_CHRONIC ? (t/24).toFixed(1) : (t*60).toFixed(0);
    o.fillStyle='#9aabc2';o.textAlign='center';o.textBaseline='top';o.fillText(lbl,x,H-B+8);
  }
  
  o.strokeStyle='#e8b04b';o.lineWidth=2;o.setLineDash([5,5]);o.beginPath();o.moveTo(px(0),Tp);o.lineTo(px(0),H-B);o.stroke();o.setLineDash([]);
  
  series.forEach(s=>{
    o.strokeStyle=s.color;o.lineWidth=2.5;o.beginPath();
    for(let i=0;i<T.length;i++){const x=px(T[i]),y=py(s.data[i]); i?o.lineTo(x,y):o.moveTo(x,y);}
    o.stroke();
  });
  if(legend){
    o.font='14px -apple-system,Segoe UI,Roboto,sans-serif';o.textAlign='left';o.textBaseline='middle';
    let lx=L+10;
    series.forEach(s=>{
      o.fillStyle=s.color;o.fillRect(lx,Tp-15,15,3);
      o.fillStyle='#5a6b85';o.fillText(s.name,lx+18,Tp-13); lx+=18+o.measureText(s.name).width+10;
    });
  }
  return {ctx,off,px,py,W,H,L,R,Tp,B,series};
}

const ch1=makeChart(document.getElementById('c1'),[{name:'PAM',data:PAM,color:'#c0392b'},{name:'TFG',data:GFR,color:'#2980b9'}],40,110,true);
const ch2=makeChart(document.getElementById('c2'),[{name:'Ang II',data:A2,color:'#e76f51'},{name:'AT1',data:AT1,color:'#d35400'},{name:'AT2',data:AT2,color:'#f39c12'}],0.4,Math.max(2.5,mx.AT1*1.1),true);
const ch3=makeChart(document.getElementById('c3'),[{name:'Aldosterona',data:ALD,color:'#1d6fb8'},{name:'Volume',data:V,color:'#8e44ad'}],0.4,Math.max(1.8,mx.Ald*1.1),true);
const ch4=makeChart(document.getElementById('c4'),[{name:'Bradicinina',data:BK,color:'#9b59b6'},{name:'Potássio',data:K,color:'#d32f2f'}],0.4,Math.max(1.8,mx.BK*1.1,mx.K*1.1),true);

function drawChart(ch,t){
  const c=ch.ctx; c.clearRect(0,0,ch.W,ch.H); c.drawImage(ch.off,0,0);
  const x=ch.px(t); c.strokeStyle='#22344f';c.lineWidth=1.5;c.beginPath();c.moveTo(x,ch.Tp);c.lineTo(x,ch.H-ch.B);c.stroke();
  ch.series.forEach(s=>{
    const v=sample(s.data,t), y=ch.py(v);
    c.beginPath();c.arc(x,y,5,0,Math.PI*2);c.fillStyle=s.color;c.fill();c.strokeStyle='#fff';c.lineWidth=2;c.stroke();
  });
}

const dg=document.getElementById('dots'), dots=[];
for(let i=0;i<6;i++){
  const el=document.createElementNS('http://www.w3.org/2000/svg','circle');
  el.setAttribute('r',4); el.setAttribute('cy',293); el.setAttribute('fill','#1d6fb8');
  dg.appendChild(el); dots.push({el:el,x:835+i*22});
}

const off={f_ren:0,f_a1:0,f_a2:0,f_ald:0,f_at1:0,f_at2:0,f_bk:0,f_adh:0};
let simT=T0, playing=true, speed=1, lastTs=null;
let coughShown=false, kShown=false;

function render(dt){
  const pam=sample(PAM,simT), a2=sample(A2,simT), at1=sample(AT1,simT), at2=sample(AT2,simT),
        al=sample(ALD,simT), adh=sample(D.ADH,simT), bk=sample(BK,simT), na=sample(NA,simT), k=sample(K,simT);
  
  const nA2=nrm(a2,mx.A2), nAT1=nrm(at1,mx.AT1), nAT2=nrm(at2,mx.AT2), nAld=nrm(al,mx.Ald),
        nADH=nrm(adh,Math.max(...D.ADH)), nBK=nrm(bk,mx.BK), nNA=nrm(na,mx.Na), nK=nrm(k,mx.K);

  document.getElementById('pamv').textContent=pam.toFixed(0);
  document.getElementById('gfill').style.width=Math.min(100,pam/120*100)+'%';
  
  let timeStr = IS_CHRONIC ? (simT/24).toFixed(1)+' dias' : (simT*60).toFixed(0)+' min';
  if(simT<0) timeStr = IS_CHRONIC ? 'Basal (h)' : 'Basal (min)';
  document.getElementById('clock').textContent = 't = ' + timeStr;

  setBox('bxR',nA2,'#ece4f7','#7b52ab'); setBox('bxP',nA2,'#fbe3dc','#e76f51');
  setBox('bxA',nAld,'#dce9f8','#1d6fb8'); setBox('bxV',nAT1,'#fbe3dc','#e76f51');
  setBox('bxH',nADH,'#e0f5e6','#2ecc71'); setBox('bxK',nBK,'#f4e6f7','#9b59b6');
  setBox('bxT',nNA,'#e6eefb','#1d6fb8');

  document.getElementById('vA2').textContent ='Ang II '+a2.toFixed(2);
  document.getElementById('vAT1').textContent='AT1 '+at1.toFixed(2);
  document.getElementById('vAT2').textContent='AT2 '+at2.toFixed(2);
  document.getElementById('vAld').textContent='aldost. '+al.toFixed(2);
  document.getElementById('vADH').textContent='ADH '+adh.toFixed(2);
  document.getElementById('vBK').textContent ='BK '+bk.toFixed(2);
  document.getElementById('vNa').textContent ='Na+ '+na.toFixed(2);
  document.getElementById('vK').textContent  ='K+ '+k.toFixed(2);

  const arrows=[['f_ren',nA2],['f_a1',nA2],['f_a2',nA2],['f_ald',nAld],['f_at1',nAT1],['f_at2',nAT2],['f_bk',nBK],['f_adh',nADH]];
  arrows.forEach(([id,n])=>{
    off[id]-= (15+150*n)*dt; const el=document.getElementById(id);
    el.setAttribute('stroke-dashoffset',off[id]); el.setAttribute('opacity',0.20+0.80*n);
  });

  const lum=document.getElementById('lumen'); 
  const h=34-18*nAT1 + 12*nAT2; // AT1 contrai, AT2 dilata
  lum.setAttribute('height',Math.max(6,h)); 
  lum.setAttribute('y',280+(34-Math.max(6,h))/2);

  dots.forEach(d=>{
    d.x += (5+80*(nNA*0.5+nAld*0.5))*dt;
    if(d.x>965) d.x=835;
    d.el.setAttribute('cx',d.x); d.el.setAttribute('opacity',0.3+0.7*nNA);
  });

  drawChart(ch1,simT); drawChart(ch2,simT); drawChart(ch3,simT); drawChart(ch4,simT);

  // ALERTAS CLÍNICOS
  if(nBK>0.6 && !coughShown && simT>0) { document.getElementById('alertCough').style.display='block'; coughShown=true; }
  if(k>1.15 && !kShown && simT>0) { document.getElementById('alertK').style.display='block'; kShown=true; }

  const s=document.getElementById('status');
  if(simT<0) s.textContent='Homeostase';
  else if(k>1.15) s.textContent='ALERTA: Hipercalemia por queda de aldosterona';
  else if(nBK>0.6) s.textContent='ALERTA: Tosse por acúmulo de Bradicinina';
  else if(nAT2>0.6) s.textContent='BRA ativo: Vasodilatação via AT2';
  else if(pam<85) s.textContent='Hipotensão: Compensação ativa';
  else s.textContent='Pressão estabilizada';
}

function loop(ts){
  if(lastTs===null) lastTs=ts;
  let dt=(ts-lastTs)/1000; lastTs=ts; dt=Math.min(dt,0.05);
  const adt = playing ? dt : 0;
  if(playing){ 
      simT += dt * (IS_CHRONIC ? (24.0/15.0) : 1.0) * speed; 
      if(simT>=T1){simT=T1;playing=false;btnPlay.textContent='▶ Reproduzir';} 
  }
  render(adt); requestAnimationFrame(loop);
}

const btnPlay=document.getElementById('btnPlay');
btnPlay.onclick=()=>{ if(!playing && simT>=T1){simT=T0;} playing=!playing; btnPlay.textContent = playing ? '⏸ Pausar' : '▶ Reproduzir'; };
document.getElementById('btnReset').onclick=()=>{ 
    simT=T0; playing=true; btnPlay.textContent='⏸ Pausar'; 
    for(const k in off) off[k]=0; 
    coughShown=false; kShown=false;
    document.getElementById('alertCough').style.display='none';
    document.getElementById('alertK').style.display='none';
};
document.querySelectorAll('.speed button').forEach(b=>{
  b.onclick=()=>{ document.querySelectorAll('.speed button').forEach(x=>x.classList.remove('on')); b.classList.add('on'); speed=parseFloat(b.dataset.s); };
});
requestAnimationFrame(loop);
</script></body></html>
"""

html = TEMPLATE.replace("__DATA__", payload)
components.html(html, height=1100, scrolling=False)

# =====================================================================
# 4. NOTAS CLÍNICAS
# =====================================================================
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🔬 Hemodinâmica:**\n"
                "- A TFG agora é corrigida. Note como, ao cair a pressão, a Ang II contrai a arteríola **eferente** e mantém a TFG estável (linha azul no primeiro gráfico) até que a pressão caia demais.")
with c2:
    st.markdown("**⏱️ Escalas Temporais:**\n"
                "- Na **Fase Aguda (minutos)**, a aldosterona e o volume quase não mudam. A compensação é toda via vasoconstrição (AT1) e barorreflexo.\n"
                "- Na **Fase Crônica (dias)**, a aldosterona sobe lentamente, retendo sódio e água, o que restaura a volemia e a pressão a longo prazo.")
with c3:
    st.markdown("**💊 Efeitos Clínicos (Teste os fármacos):**\n"
                "- **IECA (Captopril):** Aumenta a Bradicinina (dispara o alerta de tosse seca) e, por inibir a aldosterona, eleva o K+ (alerta de hipercalemia).\n"
                "- **BRA (Losartana):** Bloqueia AT1. Como sobra Ang II, ela ativa o **AT2** (laranja), que causa vasodilatação e reduz a pressão.")
