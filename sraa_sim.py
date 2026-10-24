# -*- coding: utf-8 -*-
"""
Simulador didático do Sistema Renina-Angiotensina-Aldosterona (SRAA).
Cenário: queda súbita da pressão arterial -> resposta hormonal ao longo do tempo.
Execute com:  streamlit run sraa_sim.py
"""
import json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Simulador SRAA", layout="wide")

# =====================================================================
# 1. MODELO MATEMÁTICO
# =====================================================================
PAM0 = 90.0      # pressão arterial média basal (mmHg)
T_PRE = 2.0      # minutos de linha de base antes do distúrbio

PARAMS = dict(
    k1=0.15,  k1d=0.15,   # renina        (secreção / degradação)
    k2=0.60,  k2d=0.60,   # angiotensina I
    k3=0.45,  k3d=0.45,   # angiotensina II
    k4=0.06,  k4d=0.06,   # aldosterona    (lenta)
    kv=0.020, kn=0.25,    # volume plasmático / natriurese de pressão
    ks=0.50,  bs=3.00,    # barorreflexo (tônus simpático)
    gS=0.80,  gP=1.50,    # ganhos da renina (simpático / pressão renal)
    c1=0.10,  c2=1.00,    # efeito Ang II (RVP) e volume sobre a PAM
)


def _disturbio(t, drop, width=0.10):
    """Degrau suave: 1.0 antes de t=0 -> (1-drop) depois."""
    return 1.0 - drop / (1.0 + np.exp(-t / width))


def simular(drop=0.20, t_end=30.0, n=900):
    p = PARAMS

    def rhs(t, y):
        R, A1, A2, Ald, V, S = y
        V = max(V, 0.30); A2 = max(A2, 0.05)
        D = _disturbio(t, drop)
        rvp = max(1.0 + p["c1"] * (A2 - 1.0), 0.40)          # resistência periférica
        pam = PAM0 * rvp * (V ** p["c2"]) * D

        s_alvo = np.clip(1.0 + p["bs"] * (PAM0 - pam) / PAM0, 0.30, 4.0)
        drive = max(1.0 + p["gS"] * (S - 1.0)
                    + p["gP"] * (PAM0 - pam) / PAM0, 0.05)

        return [
            p["k1"] * drive - p["k1d"] * R,          # renina
            p["k2"] * R - p["k2d"] * A1,             # Ang I
            p["k3"] * A1 - p["k3d"] * A2,            # Ang II
            p["k4"] * A2 - p["k4d"] * Ald,           # aldosterona
            p["kv"] * (Ald - 1.0) - p["kn"] * (pam / PAM0 - 1.0),  # volume
            p["ks"] * (s_alvo - S),                  # simpático
        ]

    tev = np.linspace(-T_PRE, t_end, n)
    sol = solve_ivp(rhs, (-T_PRE, t_end), [1, 1, 1, 1, 1, 1],
                    t_eval=tev, method="RK45", rtol=1e-6, atol=1e-8)

    R, A1, A2, Ald, V, S = sol.y
    D = np.array([_disturbio(t, drop) for t in sol.t])
    pam = PAM0 * (1 + p["c1"] * (A2 - 1.0)) * V ** p["c2"] * D

    r = lambda a: np.round(a, 4).tolist()
    return {"t": r(sol.t), "PAM": r(pam), "R": r(R), "A1": r(A1),
            "A2": r(A2), "Ald": r(Ald), "V": r(V),
            "t_end": float(t_end), "pam0": PAM0}


# =====================================================================
# 2. INTERFACE STREAMLIT
# =====================================================================
st.title("🫘 Simulador do Sistema Renina–Angiotensina–Aldosterona")
st.caption("Induza uma queda de pressão e observe a cascata hormonal se ativar ao longo do tempo.")

with st.sidebar:
    st.header("Cenário")
    queda = st.slider("Queda inicial da pressão (%)", 5, 40, 20, 1)
    dur = st.slider("Duração da simulação (min)", 10, 60, 30, 5)
    st.markdown("---")
    st.markdown(
        "**Como ler o diagrama**\n\n"
        "- A **cor** de cada caixa indica a concentração do mediador "
        "(mais escura = mais alta).\n"
        "- A **velocidade das setas** indica a taxa de conversão/transporte.\n"
        "- Os **vasos** se estreitam com a Angiotensina II.\n"
        "- Os **túbulos renais** reabsorvem mais Na⁺/H₂O com a aldosterona.\n\n"
        "**Modelo:** 6 EDOs acopladas (renina, Ang I, Ang II, aldosterona, "
        "volume, tônus simpático) com retroalimentação sobre a PAM."
    )

dados = simular(drop=queda / 100.0, t_end=float(dur))
payload = json.dumps(dados)

# =====================================================================
# 3. TEMPLATE HTML/SVG/JS (autocontido)
# =====================================================================
TEMPLATE = r"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><style>
:root{--renina:#7b52ab;--ang2:#e76f51;--aldo:#1d6fb8;--ink:#1b2b4b;--muted:#7a8ba3;--line:#e3e9f2;}
*{box-sizing:border-box;}
body{margin:0;background:#fff;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
.wrap{padding:6px 12px 4px;}
.top{display:flex;align-items:center;gap:16px;margin-bottom:2px;}
.pam{font-size:30px;font-weight:800;color:#c0392b;line-height:1;min-width:170px;}
.pam small{display:block;font-size:10.5px;font-weight:600;color:var(--muted);letter-spacing:.07em;text-transform:uppercase;margin-top:3px;}
.gw{flex:1;position:relative;height:16px;}
.gauge{position:absolute;inset:0;background:#eef2f8;border-radius:8px;overflow:hidden;}
.gauge>i{position:absolute;left:0;top:0;bottom:0;width:75%;border-radius:8px;background:linear-gradient(90deg,#e74c3c,#f1c40f,#2ecc71);}
.mark{position:absolute;top:-4px;bottom:-4px;width:2px;background:#4a5b78;opacity:.5;}
.mark span{position:absolute;top:-15px;left:-12px;font-size:10px;color:var(--muted);}
.clock{font-variant-numeric:tabular-nums;font-size:15px;font-weight:700;color:#4a5b78;min-width:120px;text-align:right;}
svg{display:block;width:100%;height:auto;}
.bg{stroke:#dbe4f0;stroke-width:2;}
.ttl{font-size:15px;font-weight:700;fill:#22344f;text-anchor:middle;}
.sub{font-size:10.5px;fill:#7a8ba3;text-anchor:middle;}
.val{font-size:12px;font-weight:700;text-anchor:middle;font-variant-numeric:tabular-nums;}
.flow{fill:none;stroke-width:4;stroke-linecap:round;stroke-dasharray:9 11;}
.lbl{font-size:11.5px;font-weight:700;text-anchor:middle;}
.charts{display:flex;gap:10px;margin-top:0;}
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
</style></head><body><div class="wrap">

<div class="top">
  <div class="pam"><span id="pamv">90</span> <span style="font-size:14px;color:#7a8ba3">mmHg</span>
    <small>Pressão arterial média</small></div>
  <div class="gw"><div class="gauge"><i id="gfill"></i></div>
    <div class="mark" style="left:75%"><span>90</span></div></div>
  <div class="clock" id="clock">t = 0.0 min</div>
</div>

<svg id="dia" viewBox="0 0 920 400" preserveAspectRatio="xMidYMid meet">
  <g id="bxR"><rect class="bg" x="20" y="28" width="150" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="95" y="56">Rim</text>
    <text class="sub" x="95" y="73">cél. justaglomerulares</text>
    <text class="val" x="95" y="98" id="vR" fill="#7b52ab">renina 1.00</text></g>

  <g id="bxF"><rect class="bg" x="250" y="28" width="150" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="325" y="56">Fígado</text>
    <text class="sub" x="325" y="73">angiotensinogênio</text>
    <text class="val" x="325" y="98" id="vA1" fill="#2a9d8f">Ang I 1.00</text></g>

  <g id="bxP"><rect class="bg" x="480" y="28" width="150" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="555" y="56">Pulmão</text>
    <text class="sub" x="555" y="73">ECA</text>
    <text class="val" x="555" y="98" id="vA2" fill="#e76f51">Ang II 1.00</text></g>

  <g id="bxA"><rect class="bg" x="710" y="28" width="150" height="88" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="785" y="56">Adrenal</text>
    <text class="sub" x="785" y="73">zona glomerulosa</text>
    <text class="val" x="785" y="98" id="vAld" fill="#1d6fb8">aldosterona 1.00</text></g>

  <line class="flow" id="f1" x1="176" y1="72" x2="240" y2="72" stroke="#7b52ab"/>
  <polygon points="246,72 234,66 234,78" fill="#7b52ab"/>
  <text class="lbl" x="210" y="60" fill="#7b52ab">renina</text>

  <line class="flow" id="f2" x1="406" y1="72" x2="470" y2="72" stroke="#2a9d8f"/>
  <polygon points="476,72 464,66 464,78" fill="#2a9d8f"/>
  <text class="lbl" x="440" y="60" fill="#2a9d8f">Ang I</text>

  <line class="flow" id="f3" x1="636" y1="72" x2="700" y2="72" stroke="#e76f51"/>
  <polygon points="706,72 694,66 694,78" fill="#e76f51"/>
  <text class="lbl" x="670" y="60" fill="#e76f51">Ang II</text>

  <line class="flow" id="f4" x1="555" y1="118" x2="555" y2="262" stroke="#e76f51"/>
  <polygon points="555,270 549,256 561,256" fill="#e76f51"/>
  <text class="lbl" x="566" y="200" fill="#e76f51" text-anchor="start">Ang II</text>

  <line class="flow" id="f5" x1="785" y1="118" x2="785" y2="262" stroke="#1d6fb8"/>
  <polygon points="785,270 779,256 791,256" fill="#1d6fb8"/>
  <text class="lbl" x="796" y="200" fill="#1d6fb8" text-anchor="start">aldosterona</text>

  <g id="bxV"><rect class="bg" x="460" y="276" width="190" height="104" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="555" y="298" font-size="13">Vasos sanguíneos</text>
    <rect id="lumen" x="492" y="312" width="126" height="34" rx="10" fill="#f7d9d3" stroke="#e76f51" stroke-width="2"/>
    <text class="sub" x="555" y="368">vasoconstrição</text></g>

  <g id="bxT"><rect class="bg" x="690" y="276" width="190" height="104" rx="14" fill="#f7f9fc"/>
    <text class="ttl" x="785" y="298" font-size="13">Túbulos renais</text>
    <rect x="712" y="312" width="146" height="26" rx="13" fill="#e6eefb" stroke="#c3d5ee"/>
    <g id="dots"></g>
    <text class="sub" x="785" y="368">reabsorção Na⁺ / H₂O</text></g>
</svg>

<div class="charts">
  <div class="card"><h4>Pressão arterial média (mmHg)</h4>
    <canvas id="c1" width="700" height="300"></canvas></div>
  <div class="card"><h4>Renina · Ang II · Aldosterona (unid. arb.)</h4>
    <canvas id="c2" width="700" height="300"></canvas></div>
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
const T=D.t, PAM=D.PAM, RR=D.R, A1=D.A1, A2=D.A2, ALD=D.Ald;
const T0=T[0], T1=T[T.length-1];
const MIN_PER_SEC = 2.5;             // minutos simulados por segundo real (1x)
const lerp=(a,b,f)=>a+(b-a)*f;

function sample(arr,t){
  const n=T.length;
  if(t<=T[0])return arr[0];
  if(t>=T[n-1])return arr[n-1];
  let lo=0,hi=n-1;
  while(hi-lo>1){const m=(lo+hi)>>1; if(T[m]<=t)lo=m; else hi=m;}
  const f=(t-T[lo])/((T[hi]-T[lo])||1);
  return lerp(arr[lo],arr[hi],f);
}
const mx={R:Math.max(...RR),A1:Math.max(...A1),A2:Math.max(...A2),Ald:Math.max(...ALD)};
function nrm(v,m){ if(m<=1.02) return 0.30; return Math.min(1,Math.max(0,0.25+0.75*(v-1)/(m-1))); }
function h2r(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function mix(a,b,f){const A=h2r(a),B=h2r(b);
  return 'rgb('+Math.round(lerp(A[0],B[0],f))+','+Math.round(lerp(A[1],B[1],f))+','+Math.round(lerp(A[2],B[2],f))+')';}
function setBox(id,n,pale,sat){document.querySelector('#'+id+' .bg').setAttribute('fill',mix(pale,sat,n));}

/* ---------- gráficos ---------- */
function makeChart(cv,series,ymin,ymax,legend){
  const ctx=cv.getContext('2d'), W=cv.width, H=cv.height;
  const L=64,R=16,Tp=20,B=40;
  const px=t=>L+(t-T0)/(T1-T0)*(W-L-R);
  const py=v=>Tp+(1-(v-ymin)/(ymax-ymin))*(H-Tp-B);
  const off=document.createElement('canvas'); off.width=W; off.height=H;
  const o=off.getContext('2d');
  o.fillStyle='#fbfcfe'; o.fillRect(0,0,W,H);
  o.font='20px -apple-system,Segoe UI,Roboto,sans-serif';
  // grade
  for(let i=0;i<=4;i++){
    const v=ymin+(ymax-ymin)*i/4, y=py(v);
    o.strokeStyle='#eef2f8'; o.lineWidth=1.5;
    o.beginPath();o.moveTo(L,y);o.lineTo(W-R,y);o.stroke();
    o.fillStyle='#9aabc2';o.textAlign='right';o.textBaseline='middle';
    o.fillText(v.toFixed(v<3?1:0),L-10,y);
  }
  // eixo x
  for(let i=0;i<=6;i++){
    const t=T0+(T1-T0)*i/6,x=px(t);
    o.fillStyle='#9aabc2';o.textAlign='center';o.textBaseline='top';
    o.fillText(t.toFixed(0),x,H-B+10);
  }
  o.textAlign='right';o.fillText('min',W-R,H-B+10);
  // marcador do distúrbio
  o.strokeStyle='#e8b04b';o.lineWidth=2;o.setLineDash([7,7]);
  o.beginPath();o.moveTo(px(0),Tp);o.lineTo(px(0),H-B);o.stroke();o.setLineDash([]);
  o.fillStyle='#c99a2e';o.textAlign='left';o.textBaseline='top';
  o.fillText('hipotensão',px(0)+6,Tp+2);
  // curvas
  series.forEach(s=>{
    o.strokeStyle=s.color;o.lineWidth=3;o.lineJoin='round';o.beginPath();
    for(let i=0;i<T.length;i++){const x=px(T[i]),y=py(s.data[i]); i?o.lineTo(x,y):o.moveTo(x,y);}
    o.stroke();
  });
  // legenda
  if(legend){
    o.font='19px -apple-system,Segoe UI,Roboto,sans-serif';
    o.textAlign='left';o.textBaseline='middle';
    let lx=L+12;
    series.forEach(s=>{
      o.fillStyle=s.color;o.fillRect(lx,Tp+6,22,4);
      o.fillStyle='#5a6b85';o.fillText(s.name,lx+28,Tp+9);
      lx+=28+o.measureText(s.name).width+22;
    });
  }
  return {ctx,off,px,py,W,H,L,R,Tp,B,series,ymin,ymax};
}

const ch1=makeChart(document.getElementById('c1'),
  [{name:'PAM',data:PAM,color:'#c0392b'}],55,105,false);
const ch2=makeChart(document.getElementById('c2'),
  [{name:'renina',data:RR,color:'#7b52ab'},
   {name:'Ang II',data:A2,color:'#e76f51'},
   {name:'aldosterona',data:ALD,color:'#1d6fb8'}],
  0.4,Math.max(2.2,Math.max(mx.R,mx.A2,mx.Ald)*1.12),true);

function drawChart(ch,t){
  const c=ch.ctx;
  c.clearRect(0,0,ch.W,ch.H);
  c.drawImage(ch.off,0,0);
  const x=ch.px(t);
  c.strokeStyle='#22344f';c.lineWidth=2;c.beginPath();
  c.moveTo(x,ch.Tp);c.lineTo(x,ch.H-ch.B);c.stroke();
  ch.series.forEach(s=>{
    const v=sample(s.data,t), y=ch.py(v);
    c.beginPath();c.arc(x,y,7,0,Math.PI*2);
    c.fillStyle=s.color;c.fill();
    c.strokeStyle='#fff';c.lineWidth=3;c.stroke();
  });
}

/* ---------- partículas dos túbulos ---------- */
const dg=document.getElementById('dots'), dots=[];
for(let i=0;i<7;i++){
  const el=document.createElementNS('http://www.w3.org/2000/svg','circle');
  el.setAttribute('r',5); el.setAttribute('cy',325); el.setAttribute('fill','#1d6fb8');
  dg.appendChild(el); dots.push({el:el,x:712+i*21});
}

/* ---------- loop ---------- */
const off={f1:0,f2:0,f3:0,f4:0,f5:0};
let simT=T0, playing=true, speed=1, lastTs=null;

function render(dt){
  const pam=sample(PAM,simT), r=sample(RR,simT), a1=sample(A1,simT),
        a2=sample(A2,simT), al=sample(ALD,simT);
  const nR=nrm(r,mx.R), nA1=nrm(a1,mx.A1), nA2=nrm(a2,mx.A2), nAl=nrm(al,mx.Ald);

  document.getElementById('pamv').textContent=pam.toFixed(0);
  document.getElementById('gfill').style.width=Math.min(100,pam/120*100)+'%';
  document.getElementById('clock').textContent='t = '+simT.toFixed(1)+' min';

  setBox('bxR',nR,'#ece4f7','#7b52ab');
  setBox('bxF',nA1,'#ddf1ee','#2a9d8f');
  setBox('bxP',nA2,'#fbe3dc','#e76f51');
  setBox('bxA',nAl,'#dce9f8','#1d6fb8');

  document.getElementById('vR').textContent  ='renina '+r.toFixed(2);
  document.getElementById('vA1').textContent ='Ang I '+a1.toFixed(2);
  document.getElementById('vA2').textContent ='Ang II '+a2.toFixed(2);
  document.getElementById('vAld').textContent='aldosterona '+al.toFixed(2);

  // setas: velocidade + opacidade proporcionais ao mediador
  const arrows=[['f1',nR],['f2',nA1],['f3',nA2],['f4',nA2],['f5',nAl]];
  arrows.forEach(([id,n])=>{
    off[id]-= (18+170*n)*dt;
    const el=document.getElementById(id);
    el.setAttribute('stroke-dashoffset',off[id]);
    el.setAttribute('opacity',0.30+0.70*n);
  });

  // vasoconstrição
  const lum=document.getElementById('lumen');
  const h=34-18*nA2, cy=329;
  lum.setAttribute('height',h); lum.setAttribute('y',cy-h/2);
  lum.setAttribute('fill',mix('#f7d9d3','#e76f51',0.15+0.65*nA2));

  // reabsorção tubular
  dots.forEach(d=>{
    d.x += (8+95*nAl)*dt;
    if(d.x>856) d.x=712;
    d.el.setAttribute('cx',d.x);
    d.el.setAttribute('opacity',0.25+0.75*nAl);
    d.el.setAttribute('r',3.5+2.5*nAl);
  });

  drawChart(ch1,simT); drawChart(ch2,simT);

  // status
  const s=document.getElementById('status');
  if(simT<0) s.textContent='Pressão normal — SRAA em repouso';
  else if(simT<3) s.textContent='Queda de pressão! Renina sendo secretada ↑';
  else if(nA2>0.6) s.textContent='Angiotensina II alta — vasoconstrição e aldosterona ↑';
  else if(pam<86) s.textContent='Recuperação gradual da pressão…';
  else s.textContent='Pressão restaurada pelo SRAA ✔';
}

function loop(ts){
  if(lastTs===null) lastTs=ts;
  let dt=(ts-lastTs)/1000; lastTs=ts;
  dt=Math.min(dt,0.05);
  const adt = playing ? dt : 0;
  if(playing){
    simT += dt*MIN_PER_SEC*speed;
    if(simT>=T1){simT=T1;playing=false;btnPlay.textContent='▶ Reproduzir';}
  }
  render(adt);
  requestAnimationFrame(loop);
}

/* ---------- controles ---------- */
const btnPlay=document.getElementById('btnPlay');
btnPlay.onclick=()=>{
  if(!playing && simT>=T1){simT=T0;}
  playing=!playing;
  btnPlay.textContent = playing ? '⏸ Pausar' : '▶ Reproduzir';
};
document.getElementById('btnReset').onclick=()=>{
  simT=T0; playing=true; btnPlay.textContent='⏸ Pausar';
  for(const k in off) off[k]=0;
};
document.querySelectorAll('.speed button').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('.speed button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); speed=parseFloat(b.dataset.s);
  };
});

requestAnimationFrame(loop);
</script></body></html>
"""

html = TEMPLATE.replace("__DATA__", payload)
components.html(html, height=900, scrolling=False)

# =====================================================================
# 4. NOTAS DIDÁTICAS
# =====================================================================
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**① Queda de pressão** → menor perfusão renal e menor estiramento "
                "da arteríola aferente → células justaglomerulares secretam **renina**.")
with c2:
    st.markdown("**② Cascata** → renina converte angiotensinogênio em **Ang I**; a ECA "
                "pulmonar gera **Ang II**, potente vasoconstritor → sobe a RVP e a PAM.")
with c3:
    st.markdown("**③ Aldosterona** → Ang II estimula a adrenal; a aldosterona retém "
                "Na⁺ e H₂O → volume plasmático ↑ → recuperação sustentada da pressão.")