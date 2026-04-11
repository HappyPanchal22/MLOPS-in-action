-- Query 4: Regional Taste Divergence Index
-- Compares each genre's share of sales in a region vs its global share
-- A positive divergence means the region over-indexes on that genre
-- Demonstrates: CTEs with multiple references, window functions, ROUND, complex calculations

WITH GenreGlobalShare AS (
    SELECT
        Genre,
        SUM(Global_Sales) AS GenreGlobalSales,
        SUM(Global_Sales) / (SELECT SUM(Global_Sales) FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`) * 100 AS GlobalSharePct
    FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE Genre IS NOT NULL
    GROUP BY Genre
),
RegionalShares AS (
    SELECT
        Genre,
        ROUND(SUM(NA_Sales) / (SELECT SUM(NA_Sales) FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`) * 100, 2) AS NA_SharePct,
        ROUND(SUM(EU_Sales) / (SELECT SUM(EU_Sales) FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`) * 100, 2) AS EU_SharePct,
        ROUND(SUM(JP_Sales) / (SELECT SUM(JP_Sales) FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`) * 100, 2) AS JP_SharePct
    FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE Genre IS NOT NULL
    GROUP BY Genre
)
SELECT
    g.Genre,
    ROUND(g.GenreGlobalSales, 2) AS GlobalSales_M,
    ROUND(g.GlobalSharePct, 2) AS GlobalShare,
    r.NA_SharePct,
    r.EU_SharePct,
    r.JP_SharePct,
    ROUND(r.NA_SharePct - g.GlobalSharePct, 2) AS NA_Divergence,
    ROUND(r.EU_SharePct - g.GlobalSharePct, 2) AS EU_Divergence,
    ROUND(r.JP_SharePct - g.GlobalSharePct, 2) AS JP_Divergence,
    CASE
        WHEN r.JP_SharePct - g.GlobalSharePct > 5 THEN 'Japan Favorite'
        WHEN r.NA_SharePct - g.GlobalSharePct > 5 THEN 'NA Favorite'
        WHEN r.EU_SharePct - g.GlobalSharePct > 5 THEN 'EU Favorite'
        ELSE 'Globally Balanced'
    END AS RegionalBias
FROM
    GenreGlobalShare g
JOIN
    RegionalShares r ON g.Genre = r.Genre
ORDER BY
    g.GenreGlobalSales DESC;
