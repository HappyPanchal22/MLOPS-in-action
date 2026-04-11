-- Query 2: Genre Dominance Across Gaming Eras
-- Buckets release years into eras and finds the top genre per era per region
-- Demonstrates: CTE, CASE statements, UNION ALL, aggregate functions, subqueries

WITH EraGenreSales AS (
    SELECT
        Genre,
        CASE
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 1980 AND 1995 THEN '1980-1995 (Retro)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 1996 AND 2005 THEN '1996-2005 (Golden Age)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 2006 AND 2012 THEN '2006-2012 (Modern)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 2013 AND 2020 THEN '2013-2020 (Current)'
            ELSE 'Unknown'
        END AS Era,
        'North America' AS Region,
        SUM(NA_Sales) AS RegionSales
    FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE Genre IS NOT NULL AND Year != 'N/A'
    GROUP BY Genre, Era

    UNION ALL

    SELECT
        Genre,
        CASE
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 1980 AND 1995 THEN '1980-1995 (Retro)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 1996 AND 2005 THEN '1996-2005 (Golden Age)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 2006 AND 2012 THEN '2006-2012 (Modern)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 2013 AND 2020 THEN '2013-2020 (Current)'
            ELSE 'Unknown'
        END AS Era,
        'Japan' AS Region,
        SUM(JP_Sales) AS RegionSales
    FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE Genre IS NOT NULL AND Year != 'N/A'
    GROUP BY Genre, Era

    UNION ALL

    SELECT
        Genre,
        CASE
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 1980 AND 1995 THEN '1980-1995 (Retro)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 1996 AND 2005 THEN '1996-2005 (Golden Age)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 2006 AND 2012 THEN '2006-2012 (Modern)'
            WHEN SAFE_CAST(Year AS INT64) BETWEEN 2013 AND 2020 THEN '2013-2020 (Current)'
            ELSE 'Unknown'
        END AS Era,
        'Europe' AS Region,
        SUM(EU_Sales) AS RegionSales
    FROM `mlops-lab6-bhumi.lab6_dataset.vgsales`
    WHERE Genre IS NOT NULL AND Year != 'N/A'
    GROUP BY Genre, Era
)
SELECT
    Era,
    Region,
    Genre,
    ROUND(RegionSales, 2) AS Sales_Millions,
    RANK() OVER (PARTITION BY Era, Region ORDER BY RegionSales DESC) AS GenreRank
FROM
    EraGenreSales
WHERE
    Era != 'Unknown'
QUALIFY
    GenreRank <= 5
ORDER BY
    Era, Region, GenreRank;
