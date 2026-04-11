-- Query 1: Publisher Power Rankings
-- Aggregates total games, sales by region, and average sales per title
-- to rank publishers like a league table
-- Demonstrates: CTE, aggregate functions, ROUND, multi-level ORDER BY

WITH PublisherMetrics AS (
    SELECT
        Publisher,
        COUNT(*) AS TotalGames,
        SUM(NA_Sales) AS NA_Total,
        SUM(EU_Sales) AS EU_Total,
        SUM(JP_Sales) AS JP_Total,
        SUM(Other_Sales) AS Other_Total,
        SUM(Global_Sales) AS Global_Total
    FROM
        `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE
        Publisher IS NOT NULL
        AND Publisher != 'N/A'
    GROUP BY
        Publisher
)
SELECT
    Publisher,
    TotalGames,
    ROUND(Global_Total, 2) AS GlobalSales_M,
    ROUND(NA_Total, 2) AS NASales_M,
    ROUND(EU_Total, 2) AS EUSales_M,
    ROUND(JP_Total, 2) AS JPSales_M,
    ROUND(Other_Total, 2) AS OtherSales_M,
    ROUND(Global_Total / TotalGames, 2) AS AvgSalesPerTitle,
    ROUND(NA_Total * 100 / Global_Total, 1) AS NA_SharePct,
    ROUND(JP_Total * 100 / Global_Total, 1) AS JP_SharePct
FROM
    PublisherMetrics
WHERE
    TotalGames >= 10
ORDER BY
    GlobalSales_M DESC, AvgSalesPerTitle DESC, TotalGames DESC
LIMIT 25;
