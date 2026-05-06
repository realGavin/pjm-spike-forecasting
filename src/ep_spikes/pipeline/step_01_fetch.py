"""Step 1: fetch raw data (weather, NOAA, PJM). Idempotent, cached per year."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from .. import paths
from ..config import load_settings
from ..data import noaa_fetch, pjm_fetch, weather_fetch

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-pjm", action="store_true", help="Skip PJM fetch (e.g. no API key yet)")
    args = parser.parse_args()

    load_dotenv(paths.ROOT / ".env")
    settings = load_settings(args.config)
    logging.basicConfig(level=settings.runtime.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    paths.ensure_all()

    # 1) Weather
    for node, site in settings.data.weather_sites.items():
        log.info("Weather fetch: %s (%s)", node, site.name)
        weather_fetch.fetch_openmeteo_range(
            lat=site.lat, lon=site.lon,
            start_date=settings.data.start_date, end_date=settings.data.end_date,
            cache_dir=paths.RAW_WEATHER, site_name=site.name,
            variables=settings.features.weather_hourly_vars,
            use_cache=settings.runtime.cache,
        )

    # 2) NOAA storms
    log.info("NOAA storm events fetch")
    s = int(settings.data.start_date[:4])
    e = int(settings.data.end_date[:4])
    noaa_fetch.fetch_storm_events(
        s, e, cache_dir=paths.RAW_NOAA, states=settings.data.storm_states,
        use_cache=settings.runtime.cache,
    )

    # 3) PJM (needs API key)
    if args.skip_pjm:
        log.warning("--skip-pjm set; skipping PJM fetches")
        return
    for node in settings.data.nodes:
        log.info("PJM DA LMP fetch: %s", node)
        pjm_fetch.fetch_da_lmp_zonal(
            node, settings.data.start_date, settings.data.end_date,
            cache_dir=paths.RAW_PJM, use_cache=settings.runtime.cache,
        )
        log.info("PJM metered load fetch: %s", node)
        pjm_fetch.fetch_metered_load_zonal(
            node, settings.data.start_date, settings.data.end_date,
            cache_dir=paths.RAW_PJM, use_cache=settings.runtime.cache,
        )
        log.info("PJM DA load forecast fetch: %s", node)
        try:
            pjm_fetch.fetch_da_load_forecast(
                node, settings.data.start_date, settings.data.end_date,
                cache_dir=paths.RAW_PJM, use_cache=settings.runtime.cache,
            )
        except Exception as e:
            log.error("DA load forecast fetch failed (%s). Pipeline can proceed without it.", e)


if __name__ == "__main__":
    main()
