# UFC Fighter Style Analysis
> A machine learning portfolio project exploring fighting styles across UFC weight classes using unsupervised and supervised learning.

---

## To view ipynb files
- [01 — Data Prep](https://nbviewer.org/github/brianjenkins0103/ufc-fighting-style/blob/main/01_dataprep_eda.ipynb)
- [02 — K-Means](https://nbviewer.org/github/brianjenkins0103/ufc-fighting-style/blob/main/02_kmeans.ipynb)
- [03 — Decision Tree](https://nbviewer.org/github/brianjenkins0103/ufc-fighting-style/blob/main/03_decision_tree.ipynb)

## Overview
This project analyzes UFC fighter statistics to discover natural fighting style clusters and build a classifier that can predict a fighter's style based on their stats and physical attributes.

Since no "fighting style" label exists in the raw data, **K-Means clustering** is used first to discover natural groupings, which are then used as labels for a **Decision Tree classifier**. The result is a fully interpretable model that can classify any fighter given their stats.

---

## Project Structure
```
ufc-fighting-style/
├── 01_data_prep.ipynb        # Data cleaning, feature engineering, EDA
├── 02_kmeans.ipynb           # Unsupervised clustering to discover styles
├── 03_decision_tree.ipynb    # Supervised classification using K-Means labels
├── ufc_fighters_final.csv    # Raw dataset
├── fighters_clean.csv        # Cleaned dataset
├── fighter_styles_clean.csv  # cleaned dataset with fighting styles found in Kmeans dataset
├── info.md                   # info about variables inn dataset
├── spider_charts.pdf         # Spider/Radar charts for fighting styles
└── decision_tree.png         # Visualization of trained decision tree
```

---

## Tools & Libraries
- **Python** — pandas, numpy, matplotlib
- **scikit-learn** — KMeans, DecisionTreeClassifier, StandardScaler, train_test_split

---

## Workflow

### 01 — Data Preparation
- Parsed height (`5'11"` → inches), reach (`72"` → inches), and weight (`155 lbs.` → float) from raw string formats
- Engineered **Ape Index** (`Reach - Height`) to capture reach advantage
- Mapped fighters to official UFC weight classes based on weight
- Converted percentage stats (`Str_Acc`, `Str_Def`, `TD_Acc`, `TD_Def`) from strings to decimals
- Imputed missing stance values with Orthodox (74.8% of dataset)
- Final dataset: **2,514 fighters, 21 features**

### 02 — K-Means Clustering
- Selected 9 behavior-based features: `SLpM`, `Str_Acc`, `SApM`, `Str_Def`, `TD_Avg`, `TD_Acc`, `TD_Def`, `Sub_Avg`, `Ape_Index`
- Applied `StandardScaler` to normalize features before clustering
- Used the elbow method to evaluate K=2 through K=10
- Selected **K=4** based on elbow curve and MMA domain knowledge
- Interpreted and labeled clusters based on centroid means:

| Cluster | Style | Key Characteristics |
|---|---|---|
| 0 | Brawler | Highest striking volume, absorbs most damage |
| 1 | Grappler | High takedowns and submissions |
| 2 | Defensive | Low output across all stats, counter-fighting style |
| 3 | Balanced | Well-rounded, largest group, highest ape index |

### 03 — Decision Tree Classifier
- Trained on K-Means labels using stats + Ape Index + weight as features
- 80/20 train/test split
- `max_depth=4` to prevent overfitting
- **71% overall accuracy** on test set

| Style | Precision | Recall | F1 |
|---|---|---|---|
| Balanced | 0.68 | 0.89 | 0.77 |
| Brawler | 0.81 | 0.51 | 0.63 |
| Defensive | 0.69 | 0.77 | 0.73 |
| Grappler | 0.69 | 0.49 | 0.58 |

---


---

## Key Findings
- **Balanced** fighters are the most common style across all weight classes
- **Brawlers** had the highest striking volume but absorbed the most damage in return
- **Grapplers** were the hardest style to classify, likely due to overlap with Balanced fighters
- **Ape Index** averaged ~1.7 inches across all fighters, with Heavyweights showing the most variation

---

## About
This is a personal portfolio project built to apply and demonstrate machine learning concepts using real-world sports data. Built while learning data science fundamentals step by step.

📎 [LinkedIn](https://www.linkedin.com/in/bri-jen)

