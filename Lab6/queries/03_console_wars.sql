-- Query 3: Console Wars — Platform Head-to-Head Performance
-- Groups platforms into families (Nintendo/Sony/Microsoft/PC/Other)
-- and compares sales performance across regions
-- Demonstrates: CTE, complex CASE, GROUP BY, HAVING, calculated metrics

WITH PlatformFamily AS (
    SELECT
        Name,
        Platform,
        Genre,
        Year,
        CASE
            WHEN Platform IN ('Wii', 'NES', 'SNES', 'N64', 'GC', 'WiiU', 'GB', 'GBA', 'DS', '3DS') THEN 'Nintendo'
            WHEN Platform IN ('PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV') THEN 'Sony'
            WHEN Platform IN ('XB', 'X360', 'XOne') THEN 'Microsoft'
            WHEN Platform = 'PC' THEN 'PC'
            ELSE 'Other'
        END AS Family,
        NA_Sales,
        EU_Sales,
        JP_Sales,
        Other_Sales,
        Global_Sales
    FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE Publisher IS NOT NULL
)
SELECT
    Family,
    COUNT(*) AS TotalTitles,
    COUNT(DISTINCT Platform) AS PlatformCount,
    ROUND(SUM(Global_Sales), 2) AS TotalGlobalSales,
    ROUND(AVG(Global_Sales), 3) AS AvgSalesPerTitle,
    ROUND(SUM(NA_Sales) / SUM(Global_Sales) * 100, 1) AS NA_RevenueShare,
    ROUND(SUM(EU_Sales) / SUM(Global_Sales) * 100, 1) AS EU_RevenueShare,
    ROUND(SUM(JP_Sales) / SUM(Global_Sales) * 100, 1) AS JP_RevenueShare,
    ROUND(MAX(Global_Sales), 2) AS BestSellerSales,
    CASE
        WHEN AVG(Global_Sales) >= 1.0 THEN 'Blockbuster Factory'
        WHEN AVG(Global_Sales) >= 0.5 THEN 'Strong Performer'
        WHEN AVG(Global_Sales) >= 0.2 THEN 'Steady Catalog'
        ELSE 'Volume Play'
    END AS MarketStrategy
FROM
    PlatformFamily
WHERE
    Family != 'Other'
GROUP BY
    Family
ORDER BY
    TotalGlobalSales DESC;
