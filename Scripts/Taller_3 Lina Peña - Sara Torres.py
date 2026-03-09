import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from io import BytesIO
from scipy.stats import pearsonr
import numpy as np
import matplotlib.ticker as ticker


# Configuración de directorios


output_graphs = Path("Outputs/Graphs")
output_graphs.mkdir(parents=True, exist_ok=True)  # Carpeta para gráficos

derived_data = Path("Data/Derived")
derived_data.mkdir(parents=True, exist_ok=True)  # Carpeta para bases combinadas

raw_data = Path("Data/Raw")
raw_data.mkdir(parents=True, exist_ok=True)    # Carpeta para bases originales


# Cargar datos de anomalías de temperatura


df = pd.read_csv(
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv",
    skiprows=1,
    na_values=["***"]
)

df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
df.to_csv(raw_data / "nh_temperature_anomalies_giss.csv", index=False)


print("Primeras filas de df:")
df.head()
print("\nInformación general de df:")
df.info()


# Gráfico 1 (i) 1.1.2,1.1.3 - Mes Julio


plt.plot(df["Year"], df["Jul"], color="red", label="Anomalía Julio")
plt.axhline(0, color='black', linestyle='--', label="Promedio 1951-1980")
plt.xlabel("Año")
plt.ylabel("Anomalía de temperatura (°C)")
plt.title("Anomalía de temperatura - Julio (1880-presente)")
plt.legend()
plt.grid(True)
plt.savefig(output_graphs / "Temp_Julio.png", dpi=300)
plt.show()


# Gráfico 2 (ii) 1.1.2,1.1.3 - Estaciones


plt.plot(df["Year"], df["DJF"], label="Invierno (DJF)")
plt.plot(df["Year"], df["MAM"], label="Primavera (MAM)")
plt.plot(df["Year"], df["JJA"], label="Verano (JJA)")
plt.plot(df["Year"], df["SON"], label="Otoño (SON)")
plt.axhline(0, color='black', linestyle='--', label="Promedio 1951-1980")
plt.xlabel("Año")
plt.ylabel("Anomalía de temperatura (°C)")
plt.title("Anomalías de temperatura por estación (1880-presente)")
plt.legend()
plt.grid(True)
plt.show()


# Gráfico 3 (iii) 1.1.2,1.1.3 - Promedio Anual


plt.figure(figsize=(10,6))
plt.plot(df["Year"], df["J-D"], color="blue", label="Promedio anual (J-D)")
plt.axhline(0, color='black', linestyle='--', label="Promedio 1951-1980")
plt.xlabel("Año")
plt.ylabel("Anomalía de temperatura (°C)")
plt.title("Anomalía de temperatura anual (1880-presente)")
plt.legend()
plt.grid(True)
plt.savefig(output_graphs / "Temp_Anual.png", dpi=300)
plt.show()



# Funcion para obtener anomalias mensuales de un periodo


months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def get_anomalias(df, y0, y1):
    mask = (df["Year"] >= y0) & (df["Year"] <= y1)
    values = df.loc[mask, months].values.flatten()
    return values[~np.isnan(values)]


# Tabla 1.2.1 - Tablas de frecuencias


print("Tabla 1.2.1 - Tablas de frecuencias")

for nombre, y0, y1 in [("1951-1980", 1951, 1980), ("1981-2010", 1981, 2010)]:
    anom = get_anomalias(df, y0, y1)
    bins = np.arange(np.floor(anom.min() * 4) / 4,
                     np.ceil(anom.max() * 4) / 4 + 0.25, 0.25)
    counts, edges = np.histogram(anom, bins=bins)
    tabla = pd.DataFrame({
        "Intervalo": [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(counts))],
        "Frecuencia": counts,
        "Frec. rel. (%)": np.round(counts / counts.sum() * 100, 2)
    })
    print(f"\nPeriodo {nombre} (n={len(anom)})")
    print(tabla.to_string(index=False))


# Grafico (iv) 1.2.2 - Histogramas


print("Grafico (iv) 1.2.2 - Histogramas")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
periodos = [("1951-1980", 1951, 1980, "#3B82F6"), ("1981-2010", 1981, 2010, "#EF4444")]

for ax, (nombre, y0, y1, color) in zip(axes, periodos):
    anom = get_anomalias(df, y0, y1)
    ax.hist(anom, bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", label="Promedio 1951-1980")
    media = np.mean(anom)
    ax.axvline(media, color="navy" if color == "#3B82F6" else "darkred",
               linewidth=1.8, label=f"Media = {media:.3f} C")
    ax.set_title(f"Anomalias de temperatura {nombre}", fontsize=12)
    ax.set_xlabel("Anomalia de temperatura (C)", fontsize=11)
    ax.set_ylabel("Frecuencia", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Hemisferio Norte - Anomalias mensuales (NASA GISS)", fontsize=13)
plt.tight_layout()
plt.savefig("1_2_2_histogramas.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: 1_2_2_histogramas.png")


# Grafico (v) 1.2.3 - Deciles 3 y 7 en 1951-1980


print("Grafico (v) 1.2.3 - Deciles 3 y 7")

anom_base = get_anomalias(df, 1951, 1980)
d3 = np.quantile(anom_base, 0.30)
d7 = np.quantile(anom_base, 0.70)
print(f"Decil 3 (frio)    = {d3:.4f} C")
print(f"Decil 7 (caliente) = {d7:.4f} C")


# Grafico (vi) 1.2.4 - % calientes en 1981-2010


print("Grafico (vi) 1.2.4 - Temperaturas calientes en 1981-2010")

anom_reciente = get_anomalias(df, 1981, 2010)
n_calientes = np.sum(anom_reciente > d7)
pct_calientes = n_calientes / len(anom_reciente) * 100
print(f"Anomalias sobre decil 7: {n_calientes} de {len(anom_reciente)} ({pct_calientes:.2f}%)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(anom_base, bins=20, alpha=0.6, color="#3B82F6", label="1951-1980", edgecolor="white")
ax.hist(anom_reciente, bins=20, alpha=0.6, color="#EF4444", label="1981-2010", edgecolor="white")
ax.axvline(d3, color="gray", linestyle="--", linewidth=1.5, label=f"Decil 3 = {d3:.2f} C")
ax.axvline(d7, color="black", linestyle="--", linewidth=1.5, label=f"Decil 7 = {d7:.2f} C")
ax.set_xlabel("Anomalia de temperatura (C)", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.set_title("Comparacion de distribuciones - Deciles 3 y 7 (NASA GISS)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("1_2_4_comparacion_deciles.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: 1_2_4_comparacion_deciles.png")


# Grafico (vii) 1.2.5 - Media y varianza por estacion


print("Grafico (vii) 1.2.5 - Media y varianza por estacion")

estaciones = ["DJF", "MAM", "JJA", "SON"]
periodos_p = [("1921-1950", 1921, 1950), ("1951-1980", 1951, 1980), ("1981-2010", 1981, 2010)]

filas = []
for nombre, y0, y1 in periodos_p:
    mask = (df["Year"] >= y0) & (df["Year"] <= y1)
    for s in estaciones:
        vals = pd.to_numeric(df.loc[mask, s], errors="coerce").dropna()
        filas.append({
            "Periodo": nombre,
            "Estacion": s,
            "Media": round(float(vals.mean()), 4),
            "Varianza": round(float(vals.var(ddof=1)), 4)
        })

stats_df = pd.DataFrame(filas)
print(stats_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(estaciones))
width = 0.25
colores = ["#60A5FA", "#34D399", "#F87171"]

for i, (nombre, _, _) in enumerate(periodos_p):
    varianzas = [stats_df[(stats_df["Periodo"] == nombre) &
                          (stats_df["Estacion"] == s)]["Varianza"].values[0]
                 for s in estaciones]
    ax.bar(x + i * width, varianzas, width, label=nombre, color=colores[i], edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels(estaciones, fontsize=11)
ax.set_xlabel("Estacion", fontsize=11)
ax.set_ylabel("Varianza (C^2)", fontsize=11)
ax.set_title("Varianza de anomalias por estacion y periodo (NASA GISS)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("1_2_5_varianza_estaciones.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: 1_2_5_varianza_estaciones.png")



# Cargar datos de CO2 - Parte 1.3


url = "https://www.core-econ.org/wp-content/uploads/2018/03/1_CO2-data.xlsx"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
response.raise_for_status()

df2 = pd.read_excel(BytesIO(response.content), engine="openpyxl")
df2.to_csv(raw_data / "CO2_data.csv", index=False)
df2["date"] = df2["Year"] + (df2["Month"] - 1) / 12

print("Primeras filas de df2 (CO2):")
df2.head()
print("\nInformación general de df2:")
df2.info()



# Preparar datos para correlación: Mes Julio


temp_jul = df[['Year', 'Jul']].copy()
temp_jul.rename(columns={'Jul':'Temp_Jul'}, inplace=True)

co2_jul = df2[df2['Month'] == 7][['Year', 'Trend']].copy()
co2_jul.rename(columns={'Trend':'CO2_Trend'}, inplace=True)

df_combined = pd.merge(temp_jul, co2_jul, on='Year', how='inner')

# Guardar las bases combinadas
temp_jul.to_csv(derived_data / "Temp_Julio.csv", index=False)
co2_jul.to_csv(derived_data / "CO2_Julio.csv", index=False)
df_combined.to_csv(derived_data / "Temp_vs_CO2_Julio.csv", index=False)



# Grafico (viii) 1.3.3 - Grafico CO2 vs tiempo


print("Grafico (viii) 1.3.3 - Grafico CO2")

fig, ax = plt.subplots(figsize=(12, 5))
co2_plot = df2[df2["Year"] >= 1960].copy()

ax.plot(co2_plot["date"], co2_plot["Interpolated"],
        color="#3B82F6", linewidth=1.0, alpha=0.8, label="CO2 interpolado")
ax.plot(co2_plot["date"], co2_plot["Trend"],
        color="#EF4444", linewidth=2.0, label="Tendencia")

ax.set_xlabel("Anio", fontsize=11)
ax.set_ylabel("Concentracion de CO2 (ppm)", fontsize=11)
ax.set_title("Concentracion de CO2 en Mauna Loa (1960-2017) - NOAA", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.tight_layout()
plt.savefig("1_3_3_co2_tendencia.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: 1_3_3_co2_tendencia.png")


# Gráfico (ix) - 1.3.4: CO2 vs Temperatura 


plt.scatter(df_combined['Temp_Jul'],df_combined['CO2_Trend'], color='green', alpha=0.7)
plt.title('Anomalías de temperatura en CO₂ atmosférico vs Julio')
plt.xlabel('Anomalía de Temp (°C)')
plt.ylabel('CO₂ (Trend, ppm)')
plt.grid(True)
plt.savefig(output_graphs / "CO2_Julio_vs_Temp.png", dpi=300)
plt.show()


# Correlación 


corr = pearsonr(df_combined['Temp_Jul'], df_combined['CO2_Trend'])[0]
print(f"Coeficiente de correlación: {corr:.3f}")


