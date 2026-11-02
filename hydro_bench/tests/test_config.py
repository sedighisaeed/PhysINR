"""Tests for the shipped YAML configuration files.

``load_configs`` used to store each file under its own key *and* keep the
file's own top-level section key, so ``load_configs()["methods"]`` was
``{"methods": {...}}`` instead of the methods mapping. That made
``method_config`` unusable and silently disconnected every ``params:``
declaration in ``methods.yaml`` from the CLI. These tests pin the unwrapped
shape.
"""

from __future__ import annotations

import pytest

from hydro_bench.config_defaults import (METHODS_CONFIG, load_configs,
                                         method_config)


class TestLoadConfigs:
    def test_methods_section_is_the_mapping_not_a_rewrapped_file(self):
        methods = load_configs()["methods"]
        assert "Water-Net" in methods  # the section key must not shadow entries
        assert isinstance(methods["Water-Net"], dict)

    def test_every_declared_method_states_its_mode(self):
        for name, entry in load_configs()["methods"].items():
            assert "mode" in entry, f"{name} is missing its mode"
            assert entry["mode"] in ("online", "offline")

    def test_online_methods_declare_an_adapter(self):
        for name, entry in load_configs()["methods"].items():
            if entry["mode"] == "online":
                assert entry.get("adapter"), f"{name} has no adapter path"

    def test_dataset_and_downstream_sections_load(self):
        configs = load_configs()
        assert "UIEB" in configs["datasets"]
        assert "detector" in configs["downstream"]

    def test_missing_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config file missing"):
            load_configs(tmp_path)


class TestMethodConfig:
    def test_lookup_returns_one_method_entry(self):
        entry = method_config("UDCP")
        assert entry["adapter"].endswith("udcp.adapter")

    def test_params_are_readable(self):
        assert method_config("Retinex").get("params", {}).get("sigma") == 30.0

    def test_unknown_method_raises(self):
        with pytest.raises(KeyError, match="not in"):
            method_config("No-Such-Method")

    def test_methods_config_path_is_inside_the_package(self):
        assert METHODS_CONFIG.is_file()
        assert METHODS_CONFIG.parent.name == "configs"
