"""Typed config loader. YAML -> frozen dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WeatherSite:
    lat: float
    lon: float
    name: str


@dataclass(frozen=True)
class DataCfg:
    start_date: str
    end_date: str
    nodes: list[str]
    weather_sites: dict[str, WeatherSite]
    pjm_tz: str
    forecast_origin_hour_local: int
    storm_states: list[str]


@dataclass(frozen=True)
class LabelCfg:
    absolute_threshold: float
    seasonal_quantile: float
    season_map: dict[str, list[int]]


@dataclass(frozen=True)
class FeaturesCfg:
    price_lags_hours: list[int]
    price_roll_windows_hours: list[int]
    load_lags_hours: list[int]
    storm_windows_hours: list[int]
    weather_hourly_vars: list[str]
    include_weather_anomaly: bool
    weather_forecast_proxy: str


@dataclass(frozen=True)
class ModelsCfg:
    logistic: dict[str, Any]
    xgb_spike: dict[str, Any]
    xgb_quantile: dict[str, Any]
    cv_n_splits: int
    lear: dict[str, Any] = field(default_factory=dict)
    mlp: dict[str, Any] = field(default_factory=dict)
    lstm: dict[str, Any] = field(default_factory=dict)
    extra_models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Fold:
    train_start: str
    train_end: str
    test_year: str   # can be "2022" or "2023-08--09" for smoke

    @property
    def fold_id(self) -> str:
        return f"train_{self.train_start}_{self.train_end}_test_{self.test_year}".replace("-", "")


@dataclass(frozen=True)
class CvarCfg:
    alpha: float
    hedge_thresholds: list[float]
    forward_price: float


@dataclass(frozen=True)
class RuntimeCfg:
    log_level: str = "INFO"
    cache: bool = True


@dataclass(frozen=True)
class Settings:
    data: DataCfg
    labels: LabelCfg
    features: FeaturesCfg
    models: ModelsCfg
    folds: list[Fold]
    cvar: CvarCfg
    runtime: RuntimeCfg
    config_path: Path = field(default_factory=Path)


def load_settings(yaml_path: str | Path) -> Settings:
    yaml_path = Path(yaml_path)
    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    sites = {
        k: WeatherSite(lat=float(v["lat"]), lon=float(v["lon"]), name=str(v["name"]))
        for k, v in raw["data"]["weather_sites"].items()
    }
    data = DataCfg(
        start_date=str(raw["data"]["start_date"]),
        end_date=str(raw["data"]["end_date"]),
        nodes=list(raw["data"]["nodes"]),
        weather_sites=sites,
        pjm_tz=str(raw["data"]["pjm_tz"]),
        forecast_origin_hour_local=int(raw["data"]["forecast_origin_hour_local"]),
        storm_states=list(raw["data"]["storm_states"]),
    )
    labels = LabelCfg(
        absolute_threshold=float(raw["labels"]["absolute_threshold"]),
        seasonal_quantile=float(raw["labels"]["seasonal_quantile"]),
        season_map={k: list(v) for k, v in raw["labels"]["season_map"].items()},
    )
    features = FeaturesCfg(
        price_lags_hours=list(raw["features"]["price_lags_hours"]),
        price_roll_windows_hours=list(raw["features"]["price_roll_windows_hours"]),
        load_lags_hours=list(raw["features"]["load_lags_hours"]),
        storm_windows_hours=list(raw["features"]["storm_windows_hours"]),
        weather_hourly_vars=list(raw["features"]["weather_hourly_vars"]),
        include_weather_anomaly=bool(raw["features"]["include_weather_anomaly"]),
        weather_forecast_proxy=str(raw["features"]["weather_forecast_proxy"]),
    )
    models = ModelsCfg(
        logistic=dict(raw["models"]["logistic"]),
        xgb_spike=dict(raw["models"]["xgb_spike"]),
        xgb_quantile=dict(raw["models"]["xgb_quantile"]),
        cv_n_splits=int(raw["models"]["cv_n_splits"]),
        lear=dict(raw["models"].get("lear", {})),
        mlp=dict(raw["models"].get("mlp", {})),
        lstm=dict(raw["models"].get("lstm", {})),
        extra_models=list(raw["models"].get("extra_models", [])),
    )
    folds = [
        Fold(
            train_start=str(f["train_start"]),
            train_end=str(f["train_end"]),
            test_year=str(f["test_year"]),
        )
        for f in raw["folds"]
    ]
    cvar = CvarCfg(
        alpha=float(raw["cvar"]["alpha"]),
        hedge_thresholds=list(raw["cvar"]["hedge_thresholds"]),
        forward_price=float(raw["cvar"]["forward_price"]),
    )
    rt_raw = raw.get("runtime", {})
    runtime = RuntimeCfg(
        log_level=str(rt_raw.get("log_level", "INFO")),
        cache=bool(rt_raw.get("cache", True)),
    )
    return Settings(
        data=data, labels=labels, features=features, models=models,
        folds=folds, cvar=cvar, runtime=runtime, config_path=yaml_path,
    )
