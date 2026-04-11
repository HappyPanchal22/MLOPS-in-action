# Lab 6: Advanced Querying and Visualization in BigQuery

## Video Game Sales Analysis

This lab performs advanced SQL analytics on the **Video Game Sales** dataset (16,598 games, 1980–2020) using Google BigQuery, exploring publisher performance, genre evolution, platform wars, regional market dynamics, and blockbuster hit rates.

**Dataset:** `mlops-lab6-bhumi.lab6_dataset.vgsales`
**Source:** [Kaggle — Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) (originally scraped from VGChartz)
**Columns:** Rank, Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales

---

## Dataset Setup

1. Downloaded `vgsales.csv` from [Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales)
2. Created dataset `lab6_dataset` in BigQuery under project `mlops-lab6-bhumi`
3. Uploaded CSV as table `vgsales` with auto-detected schema

---

## Queries

| # | File | Analysis | SQL Techniques |
|---|------|----------|----------------|
| 1 | `queries/01_publisher_power_rankings.sql` | Top 25 publishers ranked by total global sales with regional breakdowns | CTE, ROUND, multi-level ORDER BY, HAVING filter |
| 2 | `queries/02_genre_dominance_by_era.sql` | Top 5 genres per region across 4 gaming eras | CASE for era bucketing, UNION ALL across 3 regions, RANK() window function, QUALIFY |
| 3 | `queries/03_console_wars.sql` | Nintendo vs Sony vs Microsoft vs PC performance comparison | Complex CASE for platform family mapping, conditional classification |
| 4 | `queries/04_regional_taste_divergence.sql` | Which genres over/under-perform in each region vs global average | Multiple CTEs with scalar subqueries, divergence index calculation |
| 5 | `queries/05_blockbuster_hit_rate.sql` | Publisher hit rate — what % of titles become blockbusters | CASE for sales tiering, conditional aggregation, HAVING |

---

## Visualizations

### BigQuery UI (screenshots in `screenshots/` folder)

1. **Bar Chart** — Top 25 Publishers by Global Sales (from Query 1)
2. **Scatter Plot** — Hit Rate vs Total Revenue by Publisher (from Query 5)

### Looker Studio Dashboard

- **Scorecard:** Total global sales across all games
- **Scorecard:** Average sales per title
- **Bar Chart:** Console Wars — platform family comparison
- **Stacked Bar:** Genre dominance shifts across eras
- **Table:** Regional taste divergence with bias labels

Screenshot: `screenshots/looker_studio_dashboard.png`

---

## Export

- **Google Sheets:** Save Results → Google Sheets
- **CSV:** Save Results → CSV (local download)

---

## How to Reproduce

1. Download `vgsales.csv` from [Kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales)
2. In BigQuery, create dataset `lab6_dataset` and upload CSV as table `vgsales`
3. Run each `.sql` file from the `queries/` folder
4. Use **Explore Data** in BigQuery to create visualizations
5. Connect to [Looker Studio](https://lookerstudio.google.com/) for the dashboard