import requests
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Q1. Data collection

url = "https://restcountries.com/v3.1/region/europe"
response = requests.get(url)

print("Status code:", response.status_code)

data = response.json()

with open("q1data_raw.json", "w") as f:
    json.dump(data, f, indent=2)

df_raw = pd.json_normalize(data)

#Q2. Data wrangling

print("=== RAW DATA ===")
df_raw.info()

df_clean = df_raw.copy()

df_clean.drop(columns = ["translations", "flags", "coatOfArms", "maps", "demonyms", "timezones", "continents", "altSpellings", "tld", "idd", "startOfWeek"],
            inplace = True, errors = "ignore")

if "name.common" in df_clean.columns:
    df_clean["name"] = df_clean["name.common"]


elif "name" in df_clean.columns:
    df_clean["name"] = df_clean["name"].apply(lambda x: x.get("common") if isinstance(x, dict) else x)

df_clean.drop(columns = ["name.common", "name.official", "name.nativeName"], errors = "ignore", inplace = True)

print(df_clean[["name"]].head())

df_clean["population"] = pd.to_numeric(df_clean["population"], errors="coerce")
df_clean["area"] = pd.to_numeric(df_clean["area"], errors="coerce")

df_clean["pop_density"] = df_clean["population"] / df_clean["area"]

df_clean["density_bin"] = pd.qcut(df_clean["pop_density"], q = 5, labels = ["Very Low", "Low", "Medium", "High", "Very High"])

grouped_df = df_clean.groupby("subregion").agg(mean_pop = ("population", "mean"), max_pop = ("population", "max"))

pivot = pd.pivot_table(df_clean, index = "subregion", columns = "density_bin", values = "name", aggfunc = "count", fill_value = 0)

print(grouped_df)

df_clean = df_clean[(df_clean["area"] > 0) & df_clean["pop_density"].notna()].copy()

keep_cols = ["name", "region", "subregion", "population", "area", "pop_density", "density_bin"]
df_clean = df_clean[keep_cols]


df_clean.to_csv("q2data_cleaned.csv", index=False)


#Q3. Visualizations

# Distribution
plt.figure(figsize=(10,6))
sns.histplot(df_clean[df_clean["pop_density"] < 2000]["pop_density"], bins=30, kde=True)
plt.xscale("log")
plt.title("Population Density - Europe")
plt.xlabel("Population Density (people per km²)")
plt.ylabel("Number of Countries")
plt.tight_layout()
plt.savefig("v1_density_hist.png", dpi=150)
plt.show()

# Relationship
plt.figure(figsize=(10,6))
sns.scatterplot(x = "area", y = "population", data = df_clean, hue = "subregion")
plt.xscale("log"); plt.yscale("log")
plt.title("Correlation Between Area and Population - Europe")
plt.xlabel("Area")
plt.ylabel("Population")
plt.tight_layout()
plt.savefig("v2_area_population_scatter.png", dpi=150)
plt.show()

# Comparison
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot = True, fmt = "d", cmap = "YlGnBu")
plt.title("Country Count by Subregion x Density - Europe")
plt.xlabel("Population Density")
plt.ylabel("Subregion")
plt.tight_layout()
plt.savefig("v3_subregion_density_heatmap.png", dpi=150)
plt.show()

print(df_clean.columns.tolist())


