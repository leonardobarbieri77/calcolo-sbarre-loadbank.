import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================================
# CONFIGURAZIONE INTERFACCIA STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="LOADBANK-MASTER | Telema Busbar Calculator",
    page_icon="⚡",
    layout="wide"
)

# ==============================================================================
# 1. DATABASE MATERIALI E COEFFICIENTI (DIN 43671 / MELSOM & BOOTH / TELEMA)
# ==============================================================================
METALLI_K1 = {
    "Rame elettolitico ETP (rho=0,0178 uOhm m)": {
        "K1": 1.26, "rho": 0.0178, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0, "alpha": 0.00393, "alpha_L": 16.5e-6
    },
    "Rame elettolitico OFHC (rho=0,0176 uOhm m)": {
        "K1": 1.27, "rho": 0.0176, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0, "alpha": 0.00393, "alpha_L": 16.5e-6
    },
    "Alluminio purezza elettrica (rho=0,028 uOhm m)": {
        "K1": 1.02, "rho": 0.0280, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0, "alpha": 0.00403, "alpha_L": 23.0e-6
    },
    "Alluminio purezza elettrica (rho=0,029 uOhm m)": {
        "K1": 1.00, "rho": 0.0290, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0, "alpha": 0.00403, "alpha_L": 23.0e-6
    },
    "Lega d'alluminio EN-AW 6101B (rho=0,0325 uOhm m)": {
        "K1": 0.94, "rho": 0.0325, "mu_r": 1.000000, "dens": 2700.0, "csp": 880.0, "alpha": 0.00360, "alpha_L": 23.0e-6
    },
    "Acciaio Inox AISI 304 (rho=0,72 uOhm m)": {
        "K1": 0.17778 * (0.72**-0.487), "rho": 0.7200, "mu_r": 1.020000, "dens": 8000.0, "csp": 500.0, "alpha": 0.00100, "alpha_L": 16.0e-6
    },
    "Lega CuCr1Zr (rho=0,028 uOhm m)": {
        "K1": 0.17778 * (0.028**-0.487), "rho": 0.0280, "mu_r": 1.000000, "dens": 8910.0, "csp": 450.0, "alpha": 0.00300, "alpha_L": 17.0e-6
    },
}

VENTILAZIONE_K4 = {
    "Ambiente interno - ventilazione forzata leggera v ≈ 0,5 m/s (K4 = 1,10)": 1.10,
    "Ambiente interno - aria calma limitata / vano chiuso (K4 = 1,00)": 1.00,
    "Ambiente interno - aria calma non limitata / plenum aperto (K4 = 1,10)": 1.10,
    "Ambiente esterno - ventilazione normale v ≈ 1,0 m/s (K4 = 1,20)": 1.20,
    "Ambiente interno - conduttori chiusi in guaina non ventilati (K4 = 0,70)": 0.70,
}

SUPERFICIE_K5 = {
    "Barre grezze di estrusione o trafilatura (K5 = 1,00)": 1.00,
    "Rame verniciato o con guaina termorestringente nera (K5 = 1,08)": 1.08,
    "Alluminio verniciato o con guaina termorestringente nera (K5 = 1,12)": 1.12,
    "Rame verniciato nero - profili a C/L (K5 = 1,12)": 1.12,
    "Alluminio verniciato nero - profili a C/L (K5 = 1,20)": 1.20,
}

FORMA_K6 = {
    "Normale (piatti rettangolari, tondi, tubi) (K6 = 1,00)": 1.00,
    "Profili C o L accoppiati interni - dist. 1/4 h (K6 = 1,35)": 1.35,
    "Profili C accoppiati esterni - dist. 1/4 h (K6 = 1,43)": 1.43,
}

K7_BASE_TABLE = {1: 1.00, 2: 1.79, 3: 2.45, 4: 3.10, 5: 3.70, 6: 4.30}

# ==============================================================================
# 2. MOTORE DI CALCOLO ELETTROTERMICO E MECCANICO
# ==============================================================================
def esegui_calcolo_completo(h, s, n, d, giacitura, metallo_str, freq, t_amb, dt, vent_str, sup_str, forma_str, lungh_tratta):
    mat = METALLI_K1[metallo_str]
    k1 = mat["K1"]
    
    # K2: Deriva termica temperatura ambiente
    k2 = 1.04 if t_amb == 20 else (1.01 if t_amb == 35 else (1.00 if t_amb == 40 else (0.98 if t_amb == 50 else -0.002 * t_amb + 1.08)))
    
    # K3: Sovratemperatura ammissibile
    k3_map = {30: 0.86, 40: 1.00, 50: 1.14, 65: 1.38}
    k3 = k3_map.get(dt, 0.0148 * dt + 0.4086)
    
    k4 = VENTILAZIONE_K4[vent_str]
    k5 = SUPERFICIE_K5[sup_str]
    k6 = FORMA_K6[forma_str]

    # K8: Posa e sovrapposizione vincolata senza possibilita di contraddizione
    if "costa" in giacitura:
        k8 = 1.00
    else:
        k8_map = {1: 0.86, 2: 0.80, 3: 0.74, 4: 0.68}
        k8 = k8_map.get(n, max(0.50, 0.86 - 0.06 * (n - 1)))

    p_1 = 2.0 * (h + s)
    sez_1 = h * s
    sez_tot = h * s * n

    # K7: Accoppiamento continuo DIN 43671 e spessore efficace per skin effect
    if n == 1:
        k7 = 1.00
        s_eff = float(s)
    else:
        r_hs = h / s
        cr = 0.95 if r_hs <= 5.0 else (1.05 if r_hs >= 20.0 else 1.00)
        k7_std = K7_BASE_TABLE[n] * cr
        k7_att = ((h + n * s) / (h + s)) ** 0.39 * math.sqrt(n)
        if d <= 0:
            k7 = k7_att
            s_eff = float(s * n)
        elif d <= s:
            k7 = k7_att + (d / s) * (k7_std - k7_att)
            s_eff = float(s * n) - (d / s) * float(s * (n - 1))
        else:
            k7 = k7_std + min(1.0, (d - s) / max(1.0, h - s)) * (0.95 * n - k7_std)
            s_eff = float(s)

    # Profondita di penetrazione superficiale (Skin Depth) ed Effetto Pelle K9
    mu = 4.0 * math.pi * 1e-7 * mat["mu_r"]
    if freq <= 0:
        skin_depth = float("inf")
        k9 = 1.00
    else:
        skin_depth = 1000.0 * math.sqrt(2.0 * 1e-6 * mat["rho"] / (2.0 * math.pi * freq * mu))
        num = ((2.0 * (h - s_eff) + math.pi * s_eff - 2.0 * math.pi * skin_depth) * (1.0 - math.exp(-s_eff / (2.0 * skin_depth))) +
               math.pi * s_eff * math.exp(-s_eff / (2.0 * skin_depth)))
        den = math.sqrt((2.0 * (h - s_eff) - math.pi * skin_depth) * (1.0 - math.exp(-s_eff / skin_depth)) + math.pi * s_eff)
        pref = math.sqrt(2.0 * skin_depth / (s_eff * (h - s_eff) + math.pi * (s_eff ** 2) / 4.0))
        k9 = pref * num / den

    k_tot = k1 * k2 * k3 * k4 * k5 * k6 * k7 * k8 * k9
    i_amm = 5.0 * k_tot * (p_1 ** 0.39) * (sez_1 ** 0.5)

    # Densita di corrente J
    j_dens = i_amm / sez_tot

    # Cortocircuito adiabatico a 1 secondo (CEI EN 60865-1)
    kcw1 = math.sqrt((mat["csp"] / 385.0) * (mat["dens"] / 8890.0) * (0.0176 / mat["rho"]))
    dt_cw = max(0.0, 140.0 - t_amb - dt)
    kcw2 = math.sqrt(385.0 * 8890.0 / 0.0176 / 1e6 * dt_cw)
    icw_1s = (kcw1 * kcw2 * sez_tot) / 1000.0
    k2s2 = (kcw1 * kcw2 * sez_tot) ** 2 / 1e6

    # Bilancio termico dissipazione Joule (in watt e chilowatt)
    t_regime = t_amb + dt
    r_dc_20 = mat["rho"] / sez_tot
    r_ac_op = r_dc_20 * (1.0 + mat["alpha"] * (t_regime - 20.0)) / (k9 ** 2)
    p_joule_fase_watt_m = r_ac_op * (i_amm ** 2)
    p_joule_tot_3ph_kw = (3.0 * p_joule_fase_watt_m * lungh_tratta) / 1000.0

    # Dilatazione termica differenziale conduttore-carpenteria
    delta_l_mm = lungh_tratta * 1000.0 * mat["alpha_L"] * dt

    return {
        "p_1": p_1, "sez_1": sez_1, "sez_tot": sez_tot, "K1": k1, "K2": k2, "K3": k3, "K4": k4,
        "K5": k5, "K6": k6, "K7": k7, "K8": k8, "K9": k9, "Ktot": k_tot, "skin": skin_depth,
        "Iamm": i_amm, "J_dens": j_dens, "T_regime": t_regime, "Kcw1": kcw1, "Kcw2": kcw2,
        "Icw_1s": icw_1s, "K2S2": k2s2, "P_joule_fase_watt_m": p_joule_fase_watt_m,
        "P_joule_tot_3ph_kw": p_joule_tot_3ph_kw, "Delta_L_mm": delta_l_mm
    }

# ==============================================================================
# 3. PANNELLO LATERALE DI CONTROLLO
# ==============================================================================
with st.sidebar:
    st.header("🟢 Consolle di Comando Parametri")
    
    st.subheader("1. Quote Barra Singola")
    h_in = st.number_input("Altezza sbarra h [mm]", min_value=10, max_value=400, value=200, step=10)
    s_in = st.number_input("Spessore sbarra s [mm]", min_value=2, max_value=50, value=10, step=1)
    
    st.subheader("2. Architettura del Fascio")
    n_in = st.selectbox("Numero sbarre in parallelo // (n)", options=[1, 2, 3, 4, 5, 6], index=1)
    
    # Disabilitazione condizionale distanza per barra singola
    if n_in == 1:
        d_in = 0
        st.number_input("Distanza netta d [mm]", value=0, disabled=True, help="Non applicabile per conduttore singolo")
    else:
        d_in = st.number_input("Distanza netta tra sbarre d [mm]", min_value=0, max_value=150, value=10, step=1, help="Distanza aria libera tra le facce interne")
        
    posa_in = st.selectbox(
        "Giacitura conduttori nello spazio",
        options=["Di costa (lato h verticale)", "Di piatto (lato h orizzontale)"],
        index=0
    )
    
    st.subheader("3. Parametri Elettrici e Tratta")
    metallo_in = st.selectbox("Lega Conduttrice (K1)", options=list(METALLI_K1.keys()), index=0)
    freq_in = st.number_input("Frequenza di rete [Hz] (0 = DC)", min_value=0, max_value=400, value=60, step=10)
    lungh_in = st.slider("Lunghezza tratta condotto sbarre [m]", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    st.subheader("4. Ambiente e Ventilazione")
    tamb_in = st.selectbox("Temperatura Ambiente Tamb [°C] (K2)", options=[20, 25, 35, 40, 50, 60, 70], index=4)
    dt_in = st.selectbox("Sovratemperatura di progetto DT [K] (K3)", options=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70], index=5)
    vent_in = st.selectbox("Regime di Ventilazione (K4)", options=list(VENTILAZIONE_K4.keys()), index=0)
    sup_in = st.selectbox("Stato Superficie (K5)", options=list(SUPERFICIE_K5.keys()), index=0)
    forma_in = st.selectbox("Geometria Profilato (K6)", options=list(FORMA_K6.keys()), index=0)

# Esecuzione del calcolo
res = esegui_calcolo_completo(h_in, s_in, n_in, d_in, posa_in, metallo_in, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in, lungh_in)

# ==============================================================================
# 4. INTESTAZIONE CON RIFERIMENTI NORMATIVI
# ==============================================================================
st.title("⚡ LOADBANK-MASTER: Dimensionamento Busbar di Potenza")
st.markdown(
    """
    **Quadro Normativo Applicato:**  
    `DIN 43671 / DIN 43670` (Portata in corrente permanente) • 
    `Metodo Melsom & Booth / CDA Pub. No. 22` (Bilancio convettivo/irraggiamento) • 
    `CEI EN 61439-1/2` (Limiti termici morsetti quadri BT $\le 105\text{ °C}$) • 
    `CEI EN 60865-1 / IEC 60364-5-54` (Tenuta termica adiabatica di cortocircuito a 1s) • 
    `Equazioni di Maxwell / Dwight` (Penetrazione ed effetto pelle $50/60\text{ Hz}$)
    """
)
st.markdown("---")

# Verifiche di sicurezza e alert di conformita
if res["T_regime"] > 130:
    st.error(f"⚠️ **ALLARME CRITICO ISOLAMENTO**: Temperatura di regime calcolata a {res['T_regime']:.1f} °C. Superato il limite termico continuo per isolatori portanti a pettine in poliestere SMC/DMC e lastre dielettriche in vetronite G10/FR4 (classe 130 °C). Rischio cedimento strutturale!")
elif res["T_regime"] > 105:
    st.warning(f"⚠️ **AVVISO CONFORMITÀ CEI EN 61439-1**: Temperatura di regime a {res['T_regime']:.1f} °C. I morsetti degli interruttori scatolati/aperti (ACB/MCCB) non devono superare i 105 °C. Ridurre il carico, incrementare la sezione o potenziare la ventilazione.")

if res["J_dens"] > 3.0:
    st.warning(f"⚠️ **DENSITÀ DI CORRENTE ELEVATA**: J = {res['J_dens']:.2f} A/mm². Accettabile esclusivamente con ventilazione forzata diretta ad alta portata. Monitorare le dilatazioni termiche.")

# ==============================================================================
# 5. SCHEDE KPI E VALORI STRATEGICI
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Corrente Ammissibile (Iamm)",
    value=f"{res['Iamm']:,.1f} A",
    delta=f"Densità J: {res['J_dens']:.2f} A/mm²",
    delta_color="normal"
)

delta_k7_str = "Sbarra singola (N/D)" if n_in == 1 else ("A contatto (d = 0 mm)" if d_in <= 0 else f"Distanza d = {d_in} mm")
col2.metric(
    label="Accoppiamento (K7)",
    value=f"{res['K7']:.3f}",
    delta=delta_k7_str,
    delta_color="off"
)

col3.metric(
    label="Tenuta Cortocircuito (Icw 1s)",
    value=f"{res['Icw_1s']:,.1f} kA",
    delta=f"K²S² = {res['K2S2']:,.0f} kA²s",
    delta_color="normal"
)

col4.metric(
    label="Dissipazione Joule Trifase",
    value=f"{res['P_joule_tot_3ph_kw']:,.2f} chilowatt",
    delta=f"{res['P_joule_fase_watt_m']:,.1f} watt per metro di fase",
    delta_color="inverse"
)

# ==============================================================================
# 6. GRAFICI: DISEGNO CAD QUOTATO 2D + CURVE PARAMETRICHE
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1.35]})
fig.patch.set_facecolor('#F8F9FA')

# Disegno CAD 2D Sezione Trasversale
ax1.set_facecolor('#FFFFFF')
colore_metallo = '#D97724' if ("Rame" in metallo_in or "Cu" in metallo_in) else ('#6C757D' if "AISI" in metallo_in else '#ADB5BD')
sk = res["skin"]

if "costa" in posa_in:
    tot_w = n_in * s_in + (n_in - 1) * d_in
    x_start = -tot_w / 2.0
    for i in range(n_in):
        x0 = x_start + i * (s_in + d_in)
        rect = patches.Rectangle((x0, 0), s_in, h_in, lw=1.8, edgecolor='#3A1C00', facecolor=colore_metallo, alpha=0.9)
        ax1.add_patch(rect)
        # Nucleo non interessato da effetto pelle
        if sk != float("inf") and s_in > 2 * sk and h_in > 2 * sk:
            core = patches.Rectangle((x0 + sk, sk), s_in - 2*sk, h_in - 2*sk, lw=1.0, linestyle='--', edgecolor='#222222', facecolor='#F3B684', alpha=0.5)
            ax1.add_patch(core)
        # Frecce flusso d'aria ascensionale
        if d_in > 0 and i < n_in - 1:
            xm = x0 + s_in + d_in / 2.0
            ax1.annotate('', xy=(xm, h_in * 0.70), xytext=(xm, h_in * 0.25),
                         arrowprops=dict(arrowstyle='->', color='#0077B6', lw=2.2))
    
    quota_desc = "" if n_in == 1 else (f" - a contatto (d = 0)" if d_in == 0 else f" - d = {d_in} mm")
    simbolo_posa = "[ | ]" if n_in == 1 else (f"[ {'|'*n_in} ]" if d_in == 0 else f"[ {' | '*n_in} ]")
    ax1.set_title(f"Sezione Meccanica Posa di Costa {simbolo_posa}\n{n_in}x({h_in}x{s_in} mm){quota_desc}", fontweight='bold', fontsize=11)
    ax1.set_xlim(-max(55, tot_w * 0.85), max(55, tot_w * 0.85))
    ax1.set_ylim(-35, h_in + 40)
    
    # Quota altezza h
    ax1.annotate('', xy=(x_start - 12, h_in), xytext=(x_start - 12, 0), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax1.text(x_start - 15, h_in/2, f"h = {h_in} mm", va='center', ha='right', rotation=90, fontweight='bold', fontsize=9)

else:
    # Posa orizzontale di piatto
    d_vert = d_in if d_in > 0 else s_in
    tot_h = n_in * s_in + (n_in - 1) * d_vert
    for i in range(n_in):
        y0 = i * (s_in + d_vert)
        rect = patches.Rectangle((-h_in / 2.0, y0), h_in, s_in, lw=1.8, edgecolor='#3A1C00', facecolor=colore_metallo, alpha=0.9)
        ax1.add_patch(rect)
        
    quota_desc = "" if n_in == 1 else (f" - a contatto" if d_in == 0 else f" - interasse sovrapposizione")
    simbolo_posa = "[ — ]" if n_in == 1 else ("[ = ]" if n_in == 2 else "[ ≡ ]")
    ax1.set_title(f"Sezione Meccanica Posa di Piatto {simbolo_posa}\n{n_in}x({h_in}x{s_in} mm){quota_desc}", fontweight='bold', fontsize=11)
    ax1.set_xlim(-h_in * 0.75, h_in * 0.75)
    ax1.set_ylim(-30, max(110, tot_h + 40))

ax1.set_xlabel("Larghezza [mm]")
ax1.set_ylabel("Altezza [mm]")
ax1.grid(True, linestyle=':', alpha=0.6)

# Grafico Parametrico Portata vs Distanza
d_axis = np.linspace(0, max(30, s_in * 3), 60)
for n_idx in [1, 2, 3, 4]:
    curve_iam = [esegui_calcolo_completo(h_in, s_in, n_idx, dv, posa_in, metallo_in, freq_in, tamb_in, dt_in, vent_in, sup_in, forma_in, lungh_in)["Iamm"] for dv in d_axis]
    lw_line = 3.2 if n_idx == n_in else 1.5
    ax2.plot(d_axis, curve_iam, lw=lw_line, label=f"n = {n_idx} sbarre // ({n_idx * h_in * s_in} mm²)")

ax2.scatter([d_in], [res["Iamm"]], color='#D62828', s=90, zorder=6, label=f"Punto di lavoro ({res['Iamm']:.0f} A)")
ax2.axvline(s_in, color='#D62828', linestyle='--', alpha=0.7, label=f"Passo naturale d = s ({s_in} mm)")
ax2.set_title("Andamento Portata Ammissibile Iamm [A] vs Distanza Netta d [mm]", fontweight='bold', fontsize=11)
ax2.set_xlabel("Distanza netta d tra i conduttori [mm]")
ax2.set_ylabel("Portata Ammissibile Iamm [A]")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc="lower right", frameon=True)

st.pyplot(fig)

# ==============================================================================
# 7. SPECIFICA TECNICA PER UFFICIO TECNICO E PRODUZIONE (DFM)
# ==============================================================================
st.subheader("📋 Specifica di Industrializzazione e Linee Guida DFM")

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    st.markdown("**Verifiche Termofluidodinamiche**")
    st.write(f"- Temperatura di regime: **{res['T_regime']:.1f} °C**")
    st.write(f"- Dissipazione termica per fase: **{res['P_joule_fase_watt_m']:.1f} watt per metro**")
    st.write(f"- Potenza termica trifase rilasciata nel vano: **{res['P_joule_tot_3ph_kw']:.2f} chilowatt** (su {lungh_in:.1f} m)")
    st.write(f"- Profondità di penetrazione $\\delta$: **{res['skin']:.2f} mm** a {freq_in} Hz")

with col_d2:
    st.markdown("**Sollecitazioni Meccaniche e Carpenteria**")
    st.write(f"- Dilatazione termica differenziale: **$\\Delta L$ = {res['Delta_L_mm']:.2f} mm**")
    st.write(f"- Luce minima asola su supporti CAD: **{max(6.0, res['Delta_L_mm'] + 4.0):.1f} mm**")
    st.write(f"- Passo massimo spaziatori intermedi anti-attrazione: **$\\le 450$ mm**")
    st.write(f"- Finitura contatto consigliata: **Stagnatura galvanica 10-15 $\\mu$m sui terminali**")

with col_d3:
    st.markdown("**Quadro Normativo e Componenti**")
    st.write(f"- Tenuta termica di cortocircuito (1s): **{res['Icw_1s']:.1f} kA**")
    st.write(f"- Energia specifica passante ($K^2S^2$): **{res['K2S2']:,.0f} kA²s**")
    st.write(f"- Isolatori portanti prescritti: **Serie a pettine DMC/SMC 1000V (passo {s_in} mm)**")
    st.write(f"- Norme di conformità quadro: **CEI EN 61439-1/2, CEI EN 60204-1**")

# ==============================================================================
# 8. TABELLA DEI COEFFICIENTI ED ESPORTAZIONE DATI
# ==============================================================================
st.subheader("📊 Tabella Analitica dei Coefficienti K1..K9")

df_export = pd.DataFrame([{
    "Configurazione Posa": f"{n_in}x({h_in}x{s_in} mm) {posa_in}",
    "Metallo": metallo_in,
    "Sezione Totale [mm²]": res["sez_tot"],
    "Distanza d [mm]": d_in if n_in > 1 else 0,
    "Iamm [A]": round(res["Iamm"], 1),
    "Densità J [A/mm²]": round(res["J_dens"], 2),
    "K1 (Materiale)": round(res["K1"], 3),
    "K2 (Tamb)": round(res["K2"], 3),
    "K3 (DeltaT)": round(res["K3"], 3),
    "K4 (Ventilazione)": round(res["K4"], 3),
    "K5 (Superficie)": round(res["K5"], 3),
    "K6 (Forma)": round(res["K6"], 3),
    "K7 (Accoppiamento)": round(res["K7"], 3),
    "K8 (Giacitura)": round(res["K8"], 3),
    "K9 (Skin Effect)": round(res["K9"], 4),
    "Ktot": round(res["Ktot"], 4),
    "P_Joule_Fase [watt/m]": round(res["P_joule_fase_watt_m"], 1),
    "P_Joule_3Fasi [chilowatt]": round(res["P_joule_tot_3ph_kw"], 2),
    "Icw 1s [kA]": round(res["Icw_1s"], 1),
    "K2S2 [kA²s]": round(res["K2S2"], 0),
    "Dilatazione Delta_L [mm]": round(res["Delta_L_mm"], 2)
}])

st.dataframe(df_export, use_container_width=True)

csv_data = df_export.to_csv(index=False, sep=";").encode('utf-8')
st.download_button(
    label="💾 Scarica Report Tecnico Elettrotermico (CSV)",
    data=csv_data,
    file_name=f"Report_Busbar_{n_in}x{h_in}x{s_in}mm.csv",
    mime="text/csv"
)