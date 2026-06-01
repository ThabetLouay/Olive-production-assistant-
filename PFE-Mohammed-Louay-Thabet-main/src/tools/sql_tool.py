# src/tools/sql_tool.py

import logging
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import DB_URL

log = logging.getLogger(__name__)
_engine = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL)
    return _engine


def query_climate(
    start_year: int = None,
    end_year: int = None,
    columns: list[str] = None,
) -> pd.DataFrame:
    """
    Query monthly climate data from olive_climate_monthly.

    Available columns:
        date, tmp, dew, slp, wns, rainfall_mm,
        gdd, gdd_annual, chilling_hour, chilling_annual, spi_30d
    """
    all_cols = [
        "date", "tmp", "dew", "slp", "wns", "rainfall_mm",
        "gdd", "gdd_annual", "chilling_hour", "chilling_annual", "spi_30d",
    ]
    cols    = columns if columns else all_cols
    col_str = ", ".join(cols)

    conditions = []
    if start_year:
        conditions.append(f"EXTRACT(YEAR FROM date) >= {start_year}")
    if end_year:
        conditions.append(f"EXTRACT(YEAR FROM date) <= {end_year}")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql   = f"SELECT {col_str} FROM olive_climate_monthly {where} ORDER BY date"

    df = pd.read_sql(sql, _get_engine())
    log.info(f"query_climate returned {len(df)} rows")
    return df


def get_annual_production(start_year: int = None, end_year: int = None) -> pd.DataFrame:
    """
    Returns one row per year joining annual production with mean climate.
    Production comes from olive_production (true annual values, no fill hacks).
    """
    conditions = ["p.year IS NOT NULL"]
    if start_year:
        conditions.append(f"p.year >= {start_year}")
    if end_year:
        conditions.append(f"p.year <= {end_year}")

    where = f"WHERE {' AND '.join(conditions)}"

    sql = f"""
        SELECT
            p.year,
            p.production_tonnes,
            AVG(c.tmp)           AS avg_tmp,
            AVG(c.dew)           AS avg_dew,
            SUM(c.gdd)           AS total_gdd,
            SUM(c.chilling_hour) AS total_chilling,
            AVG(c.spi_30d)       AS avg_spi,
            SUM(c.rainfall_mm)   AS total_rainfall_mm
        FROM olive_production p
        LEFT JOIN olive_climate_monthly c
            ON EXTRACT(YEAR FROM c.date)::int = p.year
        {where}
        GROUP BY p.year, p.production_tonnes
        ORDER BY p.year
    """
    df = pd.read_sql(sql, _get_engine())
    log.info(f"get_annual_production returned {len(df)} years")
    return df


def get_drought_years(spi_threshold: float = -0.5) -> pd.DataFrame:
    """
    Returns years where average monthly SPI was below threshold,
    joined with actual production from olive_production.
    """
    sql = f"""
        SELECT
            c.year,
            c.avg_spi,
            c.avg_tmp,
            c.total_rainfall_mm,
            p.production_tonnes
        FROM (
            SELECT
                EXTRACT(YEAR FROM date)::int AS year,
                AVG(spi_30d)                 AS avg_spi,
                AVG(tmp)                     AS avg_tmp,
                SUM(rainfall_mm)             AS total_rainfall_mm
            FROM olive_climate_monthly
            GROUP BY EXTRACT(YEAR FROM date)
            HAVING AVG(spi_30d) < {spi_threshold}
        ) c
        LEFT JOIN olive_production p ON p.year = c.year
        ORDER BY c.avg_spi ASC
    """
    df = pd.read_sql(sql, _get_engine())
    log.info(f"get_drought_years found {len(df)} years below SPI {spi_threshold}")
    return df


def get_summary_stats() -> dict:
    """Quick summary of both tables for the agent context header."""
    climate_sql = """
        SELECT
            MIN(date)        AS start_date,
            MAX(date)        AS end_date,
            COUNT(*)         AS total_months,
            AVG(tmp)         AS avg_temperature,
            AVG(rainfall_mm) AS avg_monthly_rainfall_mm
        FROM olive_climate_monthly
    """
    prod_sql = """
        SELECT
            MIN(year)                  AS first_year,
            MAX(year)                  AS last_year,
            COUNT(*)                   AS total_years,
            MAX(production_tonnes)     AS max_production,
            MIN(production_tonnes)     AS min_production,
            AVG(production_tonnes)     AS avg_production
        FROM olive_production
    """
    climate_row = pd.read_sql(climate_sql, _get_engine()).iloc[0].to_dict()
    prod_row    = pd.read_sql(prod_sql,    _get_engine()).iloc[0].to_dict()
    return {**climate_row, **prod_row}
