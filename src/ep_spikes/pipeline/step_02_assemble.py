"""Step 2: build the feature panel per node, written to data/processed/."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .. import paths
from ..config import load_settings
from ..data import noaa_fetch, pjm_fetch, weather_fetch
from ..features.assemble import build_panel

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths.ensure_all()

    storms = noaa_fetch.fetch_storm_events(
        int(settings.data.start_date[:4]), int(settings.data.end_date[:4]),
        cache_dir=paths.RAW_NOAA, states=settings.data.storm_states, use_cache=True,
    )

    for node, site in settings.data.weather_sites.items():
        log.info("Assemble panel for %s", node)
        weather = weather_fetch.fetch_openmeteo_range(
            lat=site.lat, lon=site.lon,
            start_date=settings.data.start_date, end_date=settings.data.end_date,
            cache_dir=paths.RAW_WEATHER, site_name=site.name,
            variables=settings.features.weather_hourly_vars, use_cache=True,
        )
        if settings.features.include_weather_anomaly and "temperature_2m" in weather.columns:
            clim = weather_fetch.build_climatology(weather, column="temperature_2m")
            weather = weather_fetch.attach_temperature_anomaly(weather, clim)

        try:
            prices = pjm_fetch.fetch_da_lmp_zonal(
                node, settings.data.start_date, settings.data.end_date,
                cache_dir=paths.RAW_PJM, use_cache=True,
            )
        except RuntimeError as e:
            log.error("PJM DA LMP fetch failed: %s. Skipping %s panel.", e, node)
            continue

        if prices.empty:
            log.error("No PJM prices for %s. Skipping panel.", node)
            continue

        try:
            metered = pjm_fetch.fetch_metered_load_zonal(
                node, settings.data.start_date, settings.data.end_date,
                cache_dir=paths.RAW_PJM, use_cache=True,
            )
        except Exception as e:
            log.warning("Metered load fetch failed (%s); using empty.", e)
            metered = pd.DataFrame({"load_mw": []}, index=pd.DatetimeIndex([], tz="UTC"))

        try:
            fcst = pjm_fetch.fetch_da_load_forecast(
                node, settings.data.start_date, settings.data.end_date,
                cache_dir=paths.RAW_PJM, use_cache=True,
            )
        except Exception as e:
            log.warning("DA load forecast fetch failed (%s); using empty.", e)
            fcst = pd.DataFrame(columns=["target_utc", "issue_utc", "forecast_mw", "zone"])

        panel = build_panel(node, prices, metered, fcst, weather, storms, settings)
        out = paths.PROCESSED_DIR / f"panel_{node.lower()}.parquet"
        panel.to_parquet(out)
        log.info("Wrote %s (%d rows, %d cols)", out, len(panel), panel.shape[1])


if __name__ == "__main__":
    main()
