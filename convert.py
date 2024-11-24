

import geopandas as gpd
import matplotlib.pyplot as plt

# Cargar el archivo GeoJSON
gdf = gpd.read_file("mc_comunas_leaflet.json")

# Crear el gráfico
gdf.plot(edgecolor="black", facecolor="lightblue", figsize=(10, 10))

# Agregar título y mostrar el gráfico
plt.title("Mapa de Comunas")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()
