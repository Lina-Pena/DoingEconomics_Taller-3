import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from io import BytesIO
from scipy.stats import pearsonr

# -------------------------------
# Configuración de directorios
# -------------------------------
output_graphs = Path("Outputs/Graphs")
output_graphs.mkdir(parents=True, exist_ok=True)  # Carpeta para gráficos

derived_data = Path("Data/Derived")
derived_data.mkdir(parents=True, exist_ok=True)  # Carpeta para bases combinadas

raw_data = Path("Data/Raw")
raw_data.mkdir(parents=True, exist_ok=True)    # Carpeta para bases originales

# -------------------------------
# Cargar datos de anomalías de temperatura
# -------------------------------
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

# -------------------------------
# Gráfico 1 (i) 1.1.2,1.1.3 - Mes Julio
# -------------------------------
plt.plot(df["Year"], df["Jul"], color="red", label="Anomalía Julio")
plt.axhline(0, color='black', linestyle='--', label="Promedio 1951-1980")
plt.xlabel("Año")
plt.ylabel("Anomalía de temperatura (°C)")
plt.title("Anomalía de temperatura - Julio (1880-presente)")
plt.legend()
plt.grid(True)
plt.savefig(output_graphs / "Temp_Julio.png", dpi=300)
plt.show()

# -------------------------------
# Gráfico 2 (ii) 1.1.2,1.1.3 - Estaciones
# -------------------------------
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

# -------------------------------
# Gráfico 3 (iii) 1.1.2,1.1.3 - Promedio Anual
# -------------------------------
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

# -------------------------------
# Cargar datos de CO2 - Parte 1.3
# -------------------------------
url = "https://www.core-econ.org/wp-content/uploads/2018/03/1_CO2-data.xlsx"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
response.raise_for_status()

df2 = pd.read_excel(BytesIO(response.content), engine="openpyxl")
df2.to_csv(raw_data / "CO2_data.csv", index=False)

print("Primeras filas de df2 (CO2):")
df2.head()
print("\nInformación general de df2:")
df2.info()


# -------------------------------
# Preparar datos para correlación: Mes Julio
# -------------------------------
temp_jul = df[['Year', 'Jul']].copy()
temp_jul.rename(columns={'Jul':'Temp_Jul'}, inplace=True)

co2_jul = df2[df2['Month'] == 7][['Year', 'Trend']].copy()
co2_jul.rename(columns={'Trend':'CO2_Trend'}, inplace=True)

df_combined = pd.merge(temp_jul, co2_jul, on='Year', how='inner')

# Guardar las bases combinadas
temp_jul.to_csv(derived_data / "Temp_Julio.csv", index=False)
co2_jul.to_csv(derived_data / "CO2_Julio.csv", index=False)
df_combined.to_csv(derived_data / "Temp_vs_CO2_Julio.csv", index=False)

# -------------------------------
# Gráfico 4 - 1.3.4: CO2 vs Temperatura 
# -------------------------------

plt.scatter(df_combined['Temp_Jul'],df_combined['CO2_Trend'], color='green', alpha=0.7)
plt.title('Anomalías de temperatura en CO₂ atmosférico vs Julio')
plt.xlabel('Anomalía de Temp (°C)')
plt.ylabel('CO₂ (Trend, ppm)')
plt.grid(True)
plt.savefig(output_graphs / "CO2_Julio_vs_Temp.png", dpi=300)
plt.show()

# -------------------------------
# Correlación 
# -------------------------------
corr = pearsonr(df_combined['Temp_Jul'], df_combined['CO2_Trend'])[0]
print(f"Coeficiente de correlación: {corr:.3f}")



