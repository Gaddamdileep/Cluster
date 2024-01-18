# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import importlib as imlib
import errors as err

# Loading the data from Excel files
consumption = pd.read_excel("totalenergyconsumption.xlsx")
production = pd.read_excel("totalenergyproduction.xlsx")

# Setting 'country' as the index and removing the redundant column
consumption.set_index("country", inplace=True)
production.set_index("country", inplace=True)

# Selecting data for India
con_India = consumption.loc["India"]
pro_India = production.loc["India"]
con_India.name = "Production"
pro_India.name = "Consumption"

# Combining Production and Consumption data
total_India = pd.concat([con_India, pro_India], axis=1)

# Scaling the data using RobustScaler
scaler = pp.RobustScaler()
df_clust = total_India[["Production", "Consumption"]]
scaler.fit(df_clust)
df_norm = scaler.transform(df_clust)

# Function to calculate silhouette score for clustering
def one_silhoutte(xy, n_clusters):
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = skmet.silhouette_score(xy, labels)
    return score

# Evaluating silhouette scores for different numbers of clusters
for n_clusters in range(2, 11):
    score = one_silhoutte(df_norm, n_clusters)
    print(f"The silhouette score for {n_clusters} clusters is {score:.4f}")

# Clustering and visualization
kmeans = cluster.KMeans(n_clusters=2, n_init=20)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = scaler.inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 8))
plt.scatter(total_India["Production"], total_India["Consumption"], c=labels, cmap="Paired")
plt.scatter(cen[:, 0], cen[:, 1], color="k", marker="d", label="Cluster Centers")
plt.xlabel("Production")
plt.ylabel("Consumption")
plt.title("India Production vs Consumption Clustering")
plt.legend()
plt.savefig("cluster.png", dpi=300)
plt.show()

# Function for exponential fitting
def exponential(t, n0, g):
    t = t - 1990  # Normalize years
    return n0 * np.exp(g * t)

# Curve fitting for Production
param, covar = opt.curve_fit(exponential, total_India.index.astype(int), total_India["Production"])
print("Parameters for Production:", param)

# Error calculation and forecast for Production
imlib.reload(err)
forecast = exponential(2030, *param)
sigma = err.error_prop(2030, exponential, param, covar)
print(f"2030 Production Forecast: {forecast:.3e} +/- {sigma:.3e}")

# Forecast visualization for Production
years = np.linspace(1960, 2030, 100)
forecast = exponential(years, *param)
sigma = err.error_prop(years, exponential, param, covar)
plt.figure()
plt.plot(total_India.index, total_India["Production"], label="Actual Production")
plt.plot(years, forecast, label="Forecast")
plt.fill_between(years, forecast - sigma, forecast + sigma, color="yellow", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Production")
plt.title("Production Forecast")
plt.legend()
plt.savefig("fittingproduction.png", dpi=300)
plt.show()

# Curve fitting for Consumption
param, covar = opt.curve_fit(exponential, total_India.index.astype(int), total_India["Consumption"], p0=(278, 0.0))
print("Parameters for Consumption:", param)

# Forecast for Consumption
forecast = exponential(2030, *param)
sigma = err.error_prop(2030, exponential, param, covar)
print(f"2030 Consumption Forecast: {forecast:.3e} +/- {sigma:.3e}")

# Forecast visualization for Consumption
forecast = exponential(years, *param)
sigma = err.error_prop(years, exponential, param, covar)
plt.figure()
plt.plot(total_India.index, total_India["Consumption"], label="Actual Consumption")
plt.plot(years, forecast, label="Forecast")
plt.fill_between(years, forecast - sigma, forecast + sigma, color="yellow", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Consumption")
plt.title("Consumption Forecast")
plt.legend()
plt.savefig("fittingconsumption.png", dpi=300)
plt.show()
