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
    page_title="LOADBANK-MASTER | Telema Busbar Calculator Rev. 2.1",
    page_icon="⚡",
    layout="wide"
)

# ==============================================================================
# 1. DATABASE MATERIALI E COEFFICIENTI (DIN 43671 / MELSOM & BOOTH / CEI EN 60865-1)
# ==============================================================================
METALLI_K1 = {
    "Rame elettrolitico ETP CW004A (rho=0,0178 uOhm m)": {
        "K1": 1.26, "rho": 0.0178, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0,
        "alpha": 0.00393, "alpha_L": 16.5e-6, "rp02": 180.0
    },
    "Rame elettrolitico OFHC CW008A (rho=0,0176 uOhm m)": {
        "K1": 1.27, "rho": 0.0176, "mu_r": 0.999994, "dens": 8890.0, "csp": 385.0,
        "alpha": 0.00393, "alpha_L": 16.5e-6, "rp02": 180.0
    },
    "Lega CuCr1Zr CW106C invecchiata (rho=0,0223 uOhm m)": {
        "K1": 0.17778 * (0.0223**-0.487), "rho": 0.0223, "mu_r": 1.000000, "dens": 8890.0, "csp": 385.0,
        "alpha": 0.00320, "alpha_L": 17.0e-6, "rp02": 300.0
    },
    "Alluminio purezza elettrica EN-AW 1350A (rho=0,028 uOhm m)": {
        "K1": 1.02, "rho": 0.0280, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0,
        "alpha": 0.00403, "alpha_L": 23.0e-6, "rp02": 70.0
    },
    "Alluminio purezza elettrica (rho=0,029 uOhm m)": {
        "K1": 1.00, "rho": 0.0290, "mu_r": 1.000022, "dens": 2700.0, "csp": 880.0,
        "alpha": 0.00403, "alpha_L": 23.0e-6, "rp02": 70.0
    },
    "Lega d'alluminio EN-AW 6101B T6 (rho=0,0325 uOhm m)": {
        "K1": 0.94, "rho": 0.0325, "mu_r": 1.000000, "dens": 2700.0, "csp": 880.0,
        "alpha": 0.00360, "alpha_L": 23.0e-6, "rp02": 160.0
    },
    "Acciaio Inox AISI 304 EN 1.4301 (rho=0,72 uOhm m)": {
        "K1": 0.17778 * (0.72**-0.487), "rho": 0.7200, "mu_r": 1.020000, "dens": 8000.0, "csp": 500.0,
        "alpha": 0.00100, "alpha_L": 16.0e-6, "rp02": 210.0
    },
    "Lega CuCr1Zr [Valore Tabella Excel Storica] (rho=0,028 uOhm m)": {
        "K1": 0.17778 * (0.028**-0.487), "rho": 0.0280, "mu_r": 1.000000, "dens": 8910.0, "csp": 450.0,
        "alpha": 0.00300, "alpha_L": 17.0e-6, "rp02": 280.0
    },
}

VENTILAZIONE_K4 = {
    "Ambiente interno - aria calma limitata / vano chiuso (K4 = 1,00)": 1.00,
    "Ambiente interno - aria calma non limitata / grande volume (K4 = 1,06)": 1.06,
    "Ambiente interno - ventilazione forzata leggera v ≈ 0,5 m/s (K4 = 1,10)": 1.10,
    "Ambiente esterno / condotto ventilazione normale v ≈ 1,0 m/s (K4 = 1,20)": 1.20,
    "Ambiente interno - conduttori chiusi in guaina non ventilati (K4 = 0,70)": 0.70,
}

SUPERFICIE_K5 = {
    "Barre grezze di estrusione o trafilatura (emissività ~0,4)": "grezza",
    "Barre verniciate opache o con termorestringente nero (emissività ~0,9)": "verniciata",
}

FORMA_K6 = {
    "Normale (piatti rettangolari) (K6 = 1,00)": 1.00,
    "Profili C o L accoppiati interni - dist. 1/4 h (K6 = 1,35)": 1.35,
    "Profili C accoppiati esterni - dist. 1/4 h (K6 = 1,43)": 1.43,
}

K7_BASE_TABLE = {1: 1.00, 2: 1.79, 3: 2.45, 4: 3.10, 5: 3.70, 6: 4.30}

# ==============================================================================
# 2. MOTORE DI CALCOLO ELETTROTERMICO, CORTOCIRCUITO ED ELETTRODINAMICA
# ==============================================================================
def esegui_calcolo_completo(
    h, s, n, d, giacitura, metallo_str, freq, t_amb, dt,
    vent_str, sup_str, forma_str, lungh_tratta, t_lim_cw, ik_impianto_ka
):
    mat = METALLI_K1[metallo_str]
    k1 = mat["K1"]

    k2 = 1.04 if t_amb == 20 else (
        1.01 if t_amb == 35 else (
            1.00 if t_amb == 40 else (
                0.98 if t_amb == 50 else -0.002 * t_amb + 1.08
            )
        )
    )

    k3_map = {30: 0.86, 40: 1.00, 50: 1.14, 65: 1.38}
    k3 = k3_map.get(dt, 0.0148 * dt + 0.4086)
    k4 = VENTILAZIONE_K4[vent_str]
    k6 = FORMA_K6[forma_str]
    is_profilo_cl = k6 > 1.00
    is_alluminio = "Alluminio" in metallo_str or "alluminio" in metallo_str

    if SUPERFICIE_K5[sup_str] == "grezza":
        k5 = 1.00
    else:
        if is_profilo_cl:
            k5 = 1.20 if is_alluminio else 1.12
        else:
            k5 = 1.12 if is_alluminio else 1.08

    if "costa" in giacitura:
        k8 = 1.00
    else:
        k8_map = {1: 0.86, 2: 0.80, 3: 0.74, 4: 0.68}
        k8 = k8_map.get(n, max(0.50, 0.86 - 0.06 * (n - 1)))

    p_1 = 2.0 * (h + s)
    sez_1 = float(h * s)
    sez_tot = float(h * s * n)

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

    mu = 4.0 * math.pi * 1e-7 * mat["mu_r"]
    if freq <= 0:
        skin_depth = float("inf")
        k9 = 1.00
    else:
        skin_depth = 1000.0 * math.sqrt(2.0 * 1e-6 * mat["rho"] / (2.0 * math.pi * freq * mu))
        lato_mag = max(float(h), s_eff)
        lato_min = min(float(h), s_eff)
        area_eq = lato_min * (lato_mag - lato_min) + math.pi * (lato_min ** 2) / 4.0
        
        num = (
            (2.0 * (lato_mag - lato_min) + math.pi * lato_min - 2.0 * math.pi * skin_depth)
            * (1.0 - math.exp(-lato_min / (2.0 * skin_depth)))
            + math.pi * lato_min * math.exp(-lato_min / (2.0 * skin_depth))
        )
        den_rad = (2.0 * (lato_mag - lato_min) - math.pi * skin_depth) * (
            1.0 - math.exp(-lato_min / skin_depth)
        ) + math.pi * lato_min
        
        if den_rad <= 0 or area_eq <= 0:
            k9 = 1.00
        else:
            pref = math.sqrt(2.0 * skin_depth / area_eq)
            k9_raw = pref * num / math.sqrt(den_rad)
            k9 = max(0.10, min(1.00, k9_raw))

    k_tot = k1 * k2 * k3 * k4 * k5 * k6 * k7 * k8 * k9
    i_amm = 5.0 * k_tot * (p_1 ** 0.39) * (sez_1 ** 0.5)
    j_dens = i_amm / sez_tot
    t_regime = t_amb + dt

    dt_cw = max(0.0, t_lim_cw - t_regime)
    if dt_cw <= 0:
        kcw_iec = 0.0
        icw_1s_iec = 0.0
        icw_1s_excel = 0.0
    else:
        num_ln = 1.0 + mat["alpha"] * (t_lim_cw - 20.0)
        den_ln = 1.0 + mat["alpha"] * (t_regime - 20.0)
        kcw_iec = math.sqrt(
            (mat["csp"] * mat["dens"]) / (mat["rho"] * mat["alpha"] * 1e6) * math.log(num_ln / den_ln)
        )
        icw_1s_iec = (kcw_iec * sez_tot) / 1000.0

        kcw1_ex = math.sqrt((mat["csp"] / 385.0) * (mat["dens"] / 8890.0) * (0.0176 / mat["rho"]))
        kcw2_ex = math.sqrt(385.0 * 8890.0 / 0.0176 / 1e6 * dt_cw)
        icw_1s_excel = (kcw1_ex * kcw2_ex * sez_tot) / 1000.0

    i2t_ammissibile = icw_1s_iec ** 2

    r_dc_20 = mat["rho"] / sez_tot
    r_ac_op = r_dc_20 * (1.0 + mat["alpha"] * (t_regime - 20.0)) / (k9 ** 2)
    p_joule_fase_watt_m = r_ac_op * (i_amm ** 2)
    p_joule_tot_3ph_kw = (3.0 * p_joule_fase_watt_m * lungh_tratta) / 1000.0

    t_posa = 20.0
    delta_l_dt_mm = lungh_tratta * 1000.0 * mat["alpha_L"] * dt
    delta_l_tot_posa_mm = lungh_tratta * 1000.0 * mat["alpha_L"] * max(0.0, t_regime - t_posa)

    i_picco_ka = 2.0 * math.sqrt(2.0) * ik_impianto_ka
    mu0_2pi = 2.0e-7

    if n > 1 and d > 0:
        i_picco_sub_a = (i_picco_ka * 1000.0) / n
        dist_baricentri_m = (d + s) / 1000.0
        f_sub_nm = mu0_2pi * (i_picco_sub_a ** 2) / dist_baricentri_m
        w_lamina = (h * (s ** 2)) / 6.0 if "costa" in giacitura else (s * (h ** 2)) / 6.0
        sigma_amm = 0.8 * mat["rp02"]
        l_max_spacers_mm = math.sqrt((12.0 * sigma_amm * w_lamina) / (f_sub_nm / 1000.0))
        l_max_spacers_mm = min(1000.0, max(150.0, l_max_spacers_mm))
    else:
        f_sub_nm = 0.0
        l_max_spacers_mm = 500.0

    return {
        "p_1": p_1, "sez_1": sez_1, "sez_tot": sez_tot,
        "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5, "K6": k6, "K7": k7, "K8": k8, "K9": k9, "Ktot": k_tot,
        "skin": skin_depth, "Iamm": i_amm, "J_dens": j_dens, "T_regime": t_regime,
        "Kcw_IEC": kcw_iec, "Icw_1s_IEC": icw_1s_iec, "Icw_1s_Excel": icw_1s_excel, "I2t": i2t_ammissibile,
        "P_joule_fase_watt_m": p_joule_fase_watt_m, "P_joule_tot_3ph_kw": p_joule_tot_3ph_kw,
        "Delta_L_DT_mm": delta_l_dt_mm, "Delta_L_posa_mm": delta_l_tot_posa_mm,
        "F_sub_Nm": f_sub_nm, "L_max_spacers_mm": l_max_spacers_mm, "i_picco_kA": i_picco_ka
    }

# ==============================================================================
# 3. PANNELLO LATERALE DI CONTROLLO
# ==============================================================================
with st.sidebar:
    st.header("🟢 Consolle di Comando Parametri")

    st.subheader("1. Quote Barra Singola")
    h_in = st.number_input("Altezza / Larghezza maggiore h [mm]", min_value=15, max_value=400, value=200, step=5)
    s_max_consentito = max(2, int(h_in // 2))
    s_in = st.number_input(
        "Spessore sbarra s [mm] (vincolo s <= h/2)",
        min_value=2,
        max_value=min(50, s_max_consentito),
        value=min(10, s_max_consentito),
        step=1
    )

    st.subheader("2. Architettura del Fascio")
    n_in = st.selectbox("Numero sbarre in parallelo // (n)", options=[1, 2, 3, 4, 5, 6], index=1)

    if n_in == 1:
        d_in = 0
        st.number_input("Distanza netta d [mm]", value=0, disabled=True, help="Non applicabile per conduttore singolo")
    else:
        d_in = st.number_input(
            "Distanza netta tra sbarre d [mm]",
            min_value=0, max_value=150, value=10, step=1,
            help="Distanza aria libera tra le facce interne"
        )

    posa_in = st.selectbox(
        "Giacitura conduttori nello spazio",
        options=["Di costa (lato h verticale)", "Di piatto (lato h orizzontale)"],
        index=0
    )

    st.subheader("3. Parametri Elettrici e Cortocircuito")
    metallo_in = st.selectbox("Lega Conduttrice (K1)", options=list(METALLI_K1.keys()), index=0)
    freq_in = st.number_input("Frequenza di rete [Hz] (0 = DC)", min_value=0, max_value=400, value=60, step=10)
    lungh_in = st.slider("Lunghezza tratta condotto sbarre [m]", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    t_lim_cw_in = st.selectbox(
        "Temp. limite finale cortocircuito Tlim [°C]",
        options=[140, 160, 180, 200],
        index=0,
        help="140 °C: giunzioni a contatto con isolanti standard | 160-200 °C: sbarre nude in aria (CEI EN 60865-1)"
    )
    ik_in = st.number_input(
        "Corrente di cortocircuito impianto Ik'' [kA RMS]",
        min_value=5.0, max_value=150.0, value=50.0, step=5.0,
        help="Corrente presunta di cortocircuito del quadro per verifica elettrodinamica spaziatori"
    )

    st.subheader("4. Ambiente e Ventilazione")
    tamb_in = st.selectbox("Temperatura Ambiente Tamb [°C] (K2)", options=[20, 25, 35, 40, 50, 60, 70], index=4)
    dt_in = st.selectbox("Sovratemperatura di progetto DT [K] (K3)", options=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70], index=5)
    vent_in = st.selectbox("Regime di Ventilazione (K4)", options=list(VENTILAZIONE_K4.keys()), index=0)
    sup_in = st.selectbox("Stato Superficie (K5)", options=list(SUPERFICIE_K5.keys()), index=0)
    forma_in = st.selectbox("Geometria Profilato (K6)", options=list(FORMA_K6.keys()), index=0)

res = esegui_calcolo_completo(
    h_in, s_in, n_in, d_in, posa_in, metallo_in, freq_in,
    tamb_in, dt_in, vent_in, sup_in, forma_in, lungh_in, t_lim_cw_in, ik_in
)

# ==============================================================================
# 4. INTESTAZIONE CON RIFERIMENTI NORMATIVI E AVVISI DI CONFORMITÀ
# ==============================================================================
st.title("⚡ LOADBANK-MASTER: Dimensionamento Busbar di Potenza (Rev. 2.1)")
st.markdown(
    """
    **Quadro Normativo Applicato:**  
    `DIN 43671 / DIN 43670` (Coefficienti correttivi portata permanente) • 
    `Metodo Melsom & Booth / CDA Pub. No. 22` (Equazione base DC e mutuo accoppiamento K7) • 
    `CEI EN 61439-1/2` (Limiti di sovratemperatura DeltaT <= 70 K e morsetti apparecchiature) • 
    `CEI EN 60865-1 / IEC 60949` (Cortocircuito adiabatico logaritmico con rho(T) variabile e forze elettrodinamiche) • 
    `Equazioni di Maxwell / Dwight` (Effetto pelle AC K9)
    """
)
st.markdown("---")

if res["T_regime"] > 130:
    st.error(
        f"⚠️ **ALLARME CRITICO ISOLAMENTO**: Temperatura assoluta di regime a {res['T_regime']:.1f} °C "
        f"(Tamb {tamb_in} °C + DT {dt_in} K). Superata la classe termica (130 °C) degli isolatori portanti "
        f"in poliestere SMC/DMC e vetronite G10/FR4!"
    )
elif dt_in > 65 or res["T_regime"] > 105:
    st.warning(
        f"⚠️ **AVVISO INTERFACCE APPARECCHIATURE (CEI EN 61439-1 Tab. 6)**: Temperatura assoluta sbarra = {res['T_regime']:.1f} °C "
        f"(Sovratemperatura DeltaT = {dt_in} K). La norma ammette DeltaT <= 70 K sui terminali nudi, ma i costruttori "
        f"di interruttori (ACB/MCCB) prescrivono il declassamento per temperature ai morsetti superiori a 100 - 105 °C."
    )

if res["Icw_1s_IEC"] < ik_in:
    st.error(
        f"⚠️ **VERIFICA CORTOCIRCUITO NON SODDISFATTA**: La tenuta adiabatica a 1s della sbarra ({res['Icw_1s_IEC']:.1f} kA) "
        f"è inferiore alla corrente di cortocircuito dell'impianto Ik'' ({ik_in:.1f} kA)!"
    )

if res["J_dens"] > 3.0:
    st.warning(
        f"⚠️ **DENSITÀ DI CORRENTE ELEVATA**: J = {res['J_dens']:.2f} A/mm². Ammissibile solo con ventilazione forzata diretta."
    )

# ==============================================================================
# 5. SCHEDE KPI E VALORI STRATEGICI
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Corrente Ammissibile (Iamm)",
    value=f"{res['Iamm']:,.1f} A",
    delta=f"Densità J: {res['J_dens']:.2f} A/mm² (Sez: {int(res['sez_tot'])} mm²)",
    delta_color="normal"
)

delta_k7_str = (
    "Sbarra singola (N/D)" if n_in == 1
    else ("A contatto (d = 0 mm)" if d_in <= 0 else f"Distanza d = {d_in} mm")
)
col2.metric(
    label="Accoppiamento (K7)",
    value=f"{res['K7']:.3f}",
    delta=delta_k7_str,
    delta_color="off"
)

col3.metric(
    label=f"Tenuta Cortocircuito IEC (Tlim={t_lim_cw_in}°C)",
    value=f"{res['Icw_1s_IEC']:,.1f} kA (1s)",
    delta=f"Modello Excel rho20 cost.: {res['Icw_1s_Excel']:,.1f} kA",
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

ax1.set_facecolor('#FFFFFF')
colore_metallo = (
    '#D97724' if ("Rame" in metallo_in or "Cu" in metallo_in)
    else ('#6C757D' if "AISI" in metallo_in else '#ADB5BD')
)
sk = res["skin"]

if "costa" in posa_in:
    tot_w = n_in * s_in + (n_in - 1) * d_in
    x_start = -tot_w / 2.0
    for i in range(n_in):
        x0 = x_start + i * (s_in + d_in)
        rect = patches.Rectangle((x0, 0), s_in, h_in, lw=1.8, edgecolor='#3A1C00', facecolor=colore_metallo, alpha=0.9)
        ax1.add_patch(rect)
        if sk != float("inf") and s_in > 2 * sk and h_in > 2 * sk:
            core = patches.Rectangle(
                (x0 + sk, sk), s_in - 2 * sk, h_in - 2 * sk,
                lw=1.0, linestyle='--', edgecolor='#222222', facecolor='#F3B684', alpha=0.5
            )
            ax1.add_patch(core)
        if d_in > 0 and i < n_in - 1:
            xm = x0 + s_in + d_in / 2.0
            ax1.annotate(
                '', xy=(xm, h_in * 0.70), xytext=(xm, h_in * 0.25),
                arrowprops=dict(arrowstyle='->', color='#0077B6', lw=2.2)
            )

    quota_desc = "" if n_in == 1 else (" - a contatto (d = 0 mm)" if d_in == 0 else f" - d = {d_in} mm")
    simbolo_posa = "[ | ]" if n_in == 1 else (f"[ {'|'*n_in} ]" if d_in == 0 else f"[ {' | '*n_in} ]")
    ax1.set_title(f"Sezione Meccanica Posa di Costa {simbolo_posa}\n{n_in}x({h_in}x{s_in} mm){quota_desc}", fontweight='bold', fontsize=11)
    ax1.set_xlim(-max(55, tot_w * 0.85), max(55, tot_w * 0.85))
    ax1.set_ylim(-35, h_in + 40)

    ax1.annotate('', xy=(x_start - 12, h_in), xytext=(x_start - 12, 0), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax1.text(x_start - 15, h_in / 2.0, f"h = {h_in} mm", va='center', ha='right', rotation=90, fontweight='bold', fontsize=9)

else:
    d_vert = float(d_in)
    tot_h = n_in * s_in + (n_in - 1) * d_vert
    for i in range(n_in):
        y0 = i * (s_in + d_vert)
        rect = patches.Rectangle((-h_in / 2.0, y0), h_in, s_in, lw=1.8, edgecolor='#3A1C00', facecolor=colore_metallo, alpha=0.9)
        ax1.add_patch(rect)

    quota_desc = "" if n_in == 1 else (" - a contatto (d = 0 mm)" if d_in == 0 else f" - distanziate d = {d_in} mm")
    simbolo_posa = "[ — ]" if n_in == 1 else ("[ = ]" if n_in == 2 else "[ ≡ ]")
    ax1.set_title(f"Sezione Meccanica Posa di Piatto {simbolo_posa}\n{n_in}x({h_in}x{s_in} mm){quota_desc}", fontweight='bold', fontsize=11)
    ax1.set_xlim(-h_in * 0.75, h_in * 0.75)
    ax1.set_ylim(-30, max(110, tot_h + 40))

ax1.set_xlabel("Larghezza [mm]")
ax1.set_ylabel("Altezza [mm]")
ax1.grid(True, linestyle=':', alpha=0.6)

d_axis = np.linspace(0, max(30, s_in * 3), 60)
for n_idx in [1, 2, 3, 4]:
    curve_iam = [
        esegui_calcolo_completo(
            h_in, s_in, n_idx, dv, posa_in, metallo_in, freq_in,
            tamb_in, dt_in, vent_in, sup_in, forma_in, lungh_in, t_lim_cw_in, ik_in
        )["Iamm"]
        for dv in d_axis
    ]
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

skin_str = "∞ (Corrente Continua DC)" if res["skin"] == float("inf") else f"{res['skin']:.2f} mm a {freq_in} Hz"

with col_d1:
    st.markdown("**Verifiche Termofluidodinamiche**")
    st.write(f"- Temperatura assoluta di regime: **{res['T_regime']:.1f} °C** (Tamb = {tamb_in} °C, DeltaT = {dt_in} K)")
    st.write(f"- Dissipazione termica per fase: **{res['P_joule_fase_watt_m']:.1f} watt per metro**")
    st.write(f"- Potenza termica trifase nel vano: **{res['P_joule_tot_3ph_kw']:.2f} chilowatt** (su {lungh_in:.1f} m)")
    st.write(f"- Profondità di penetrazione: **{skin_str}**")

with col_d2:
    st.markdown("**Dilatazioni Termiche ed Elettrodinamica**")
    st.write(f"- Dilatazione da posa in officina (20 °C -> {res['T_regime']:.0f} °C): **{res['Delta_L_posa_mm']:.2f} mm**")
    st.write(f"- Dilatazione per solo salto termico (DeltaT = {dt_in} K): **{res['Delta_L_DT_mm']:.2f} mm**")
    st.write(f"- Luce minima asola su supporti CAD: **{max(6.0, res['Delta_L_posa_mm'] + 4.0):.1f} mm**")
    if n_in > 1 and d_in > 0:
        st.write(f"- Forza attrattiva sub-conduttori (Ip = {res['i_picco_kA']:.0f} kA): **{res['F_sub_Nm']:,.0f} N/m**")
        st.write(f"- Passo max calcolato spaziatori intermedi: **<= {res['L_max_spacers_mm']:.0f} mm**")
    else:
        st.write("- Passo consigliato supporti portanti: **<= 500 mm**")

with col_d3:
    st.markdown("**Tenuta al Cortocircuito e Componenti**")
    st.write(f"- Icw 1s rigorosa CEI EN 60865-1 (rho(T) variabile): **{res['Icw_1s_IEC']:.1f} kA**")
    st.write(f"- Icw 1s modello Excel (rho20 costante): **{res['Icw_1s_Excel']:.1f} kA**")
    st.write(f"- Integrale di Joule ammissibile (I^2 t a 1s): **{res['I2t']:,.0f} kA²s**")
    st.write(f"- Isolatori portanti prescritti: **Serie a pettine DMC/SMC 1000V (passo {s_in} mm)**")

# ==============================================================================
# 8. TABELLA DEI COEFFICIENTI ED ESPORTAZIONE DATI
# ==============================================================================
st.subheader("📊 Tabella Analitica dei Coefficienti K1..K9")

df_export = pd.DataFrame([{
    "Configurazione Posa": f"{n_in}x({h_in}x{s_in} mm) {posa_in}",
    "Metallo": metallo_in,
    "Sezione Totale [mm²]": int(res["sez_tot"]),
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
    "Skin Depth [mm]": round(res["skin"], 2) if res["skin"] != float("inf") else "Inf (DC)",
    "K9 (Skin Effect)": round(res["K9"], 4),
    "Ktot": round(res["Ktot"], 4),
    "P_Joule_Fase [watt/m]": round(res["P_joule_fase_watt_m"], 1),
    "P_Joule_3Fasi [chilowatt]": round(res["P_joule_tot_3ph_kw"], 2),
    "Icw 1s IEC [kA]": round(res["Icw_1s_IEC"], 1),
    "Icw 1s Excel [kA]": round(res["Icw_1s_Excel"], 1),
    "Delta_L da 20°C [mm]": round(res["Delta_L_posa_mm"], 2),
    "Passo Max Spaziatori [mm]": round(res["L_max_spacers_mm"], 0)
}])

st.dataframe(df_export, use_container_width=True)

csv_data = df_export.to_csv(index=False, sep=";").encode('utf-8')
st.download_button(
    label="💾 Scarica Report Tecnico Elettrotermico Rev. 2.1 (CSV)",
    data=csv_data,
    file_name=f"Report_Busbar_Rev21_{n_in}x{h_in}x{s_in}mm.csv",
    mime="text/csv"
)
