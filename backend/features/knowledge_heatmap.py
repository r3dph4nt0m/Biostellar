import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import json

with open("json_db.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

categories_keywords = {
    "Microgravity": ["microgravity", "spaceflight", "simulated microgravity", "space"],
    "Animal Studies": ["mice", "mouse", "murine", "C. elegans", "worm", "animal"],
    "Plant Biology": ["Arabidopsis", "plant", "photosynthesis", "root", "seedling"],
    "Molecular Biology": ["protein", "phosphoprotein", "ROS", "gene", "molecular", "DNA", "RNA"],
    "Radiation Effects": ["irradiation", "radiation", "cosmic", "ionizing"],
    "COVID-19": ["SARS-CoV-2", "COVID-19", "virus", "infection"],
}

category_year_counts = defaultdict(lambda: defaultdict(int))

for doc in documents:
    text = (doc.get("title", "") + " " + doc.get("summary", "")).lower()

    year = str(doc.get("year", "Unknown"))
    if not year.isdigit() or len(year) != 4:
        year = "Unknown"

    for category, keywords in categories_keywords.items():
        if any(keyword.lower() in text for keyword in keywords):
            category_year_counts[category][year] += 1

df = pd.DataFrame.from_dict(category_year_counts, orient="index").fillna(0).astype(int)

sorted_cols = sorted([c for c in df.columns if c.isdigit()]) + [c for c in df.columns if not c.isdigit()]
df = df[sorted_cols]

plt.figure(figsize=(10, 6))
sns.heatmap(df, cmap="YlOrRd", annot=True, fmt="d")
plt.title("NASA Bioscience Publications by Topic and Year", fontsize=14)
plt.xlabel("Publication Year")
plt.ylabel("Research Category")
plt.tight_layout()
plt.show() 

