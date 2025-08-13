import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Leer datos desde archivo ---
df_irr = pd.read_excel(
    r"C:\Users\adria\OneDrive\Escritorio\informacion proyecto final\programas_python\ghi_corregido.xlsx",
    usecols=["GHI_corr_seguidor"]           # usecols=["GHI_corr_seguidor"]            usecols=["GHI_corr_estacional"]      usecols=["GHI_corr_fijo"]
)

# --- Extraer valores como array plano ---
wh_2020 = df_irr["GHI_corr_seguidor"].values.flatten() ####aqui tambien cambiar el nombre de la columna (son los mismos de arriba)

# Temperatura de la celda
ruta_temp_pv = r"D:\Datos_python\temperatura_celda.csv"
temp_pv = pd.read_csv(ruta_temp_pv)["Temperatura_celda"].values.flatten()

# --- 2. Función para curvas I-V y P-V ---
def iv_curve(Va, Suns, TaC):
    k = 1.38e-23
    q = 1.6e-19
    A = 0.6
    Vg = 0.595
    Ns = 144
    T1 = 273 + 43
    Voc_T1 = 52.9 / Ns
    Isc_T1 = 10.74
    T2 = 273 + 80
    Voc_T2 = 52.8112 / Ns
    Isc_T2 = 10.7548
    TarK = 273 + TaC

    Iph_T1 = Isc_T1 * Suns
    a = (Isc_T2 - Isc_T1) / Isc_T1 * 1 / (T2 - T1)
    Iph = Iph_T1 * (1 + a * (TarK - T1))
    Vt_T1 = k * T1 / q
    Ir_T1 = Isc_T1 / (np.exp(Voc_T1 / (A * Vt_T1)) - 1)
    b = Vg * q / (A * k)
    Ir = Ir_T1 * (TarK / T1) ** (3 / A) * np.exp(-b * (1 / TarK - 1 / T1))
    X2v = Ir_T1 / (A * Vt_T1) * np.exp(Voc_T1 / (A * Vt_T1))
    dVdI_Voc = -0.3 / Ns / 2
    Rs = -dVdI_Voc - 1 / X2v
    Vt_Ta = A * k * TarK / q
    Vc = Va / Ns

    Ia = np.zeros_like(Vc)
    for _ in range(10):
        Ia = Ia - (Iph - Ia - Ir * (np.exp((Vc + Ia * Rs) / Vt_Ta) - 1)) / \
             (-1 - Ir * (np.exp((Vc + Ia * Rs) / Vt_Ta) - 1) * Rs / Vt_Ta)

    Ia = np.maximum(Ia, 0)
    Ppv = Va * Ia
    return Ia, Ppv

# --- 3. Calcular MPPT ---
Va = np.linspace(0, 52, 500)
P_MPPT = []
V_MPPT = []

for G, T in zip(wh_2020, temp_pv):
    Suns = G / 1000
    Ipv, Ppv = iv_curve(Va, Suns, T)
    if np.max(Ppv) > 0:
        idx = np.argmax(Ppv)
        P_MPPT.append(Ppv[idx])
        V_MPPT.append(Va[idx])
    else:
        P_MPPT.append(np.nan)
        V_MPPT.append(np.nan)

P_MPPT = np.array(P_MPPT)
V_MPPT = np.array(V_MPPT)

# Crear eje X continuo (0 a 8760)
horas_anio = np.arange(len(V_MPPT))

# --- 4. Graficar resultados ---
plt.figure(figsize=(15, 4))
plt.plot(horas_anio, V_MPPT, color='orange')
plt.title('Tensión MPPT por hora del año (un panel)')
plt.xlabel('Hora del año')
plt.ylabel('Tensión (V)')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 4))
plt.plot(horas_anio, P_MPPT, color='red')
plt.title('Potencia MPPT por hora del año (un panel)')
plt.xlabel('Hora del año ')
plt.ylabel('Potencia (W)')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 4))
plt.plot(horas_anio, P_MPPT/V_MPPT, color='red')
plt.title('Corriente MPPT por hora del año (un panel)')
plt.xlabel('Hora del año ')
plt.ylabel('Potencia (W)')
plt.grid(True, which='both')
plt.tight_layout()
plt.show()
