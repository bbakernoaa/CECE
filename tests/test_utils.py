import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from cece.config import CeceConfig
from cece.exceptions import CeceConfigError
from cece.utils import (
    dict_to_yaml,
    get_logger,
    load_config,
    setup_logging,
    validate_array_contiguity,
    validate_array_dimensions,
    validate_array_dtype,
    yaml_to_dict,
)


def test_load_config_with_cece_config():
    config = CeceConfig()
    result = load_config(config)
    assert result is config


def test_load_config_with_dict():
    config_dict = {
        "species": {
            "CO": [{"field": "CO_FIELD", "operation": "add", "scale": 1.0}]
        }
    }
    result = load_config(config_dict)
    assert isinstance(result, CeceConfig)
    assert "CO" in result.species
    assert result.species["CO"][0].field_name == "CO_FIELD"


def test_load_config_with_invalid_dict():
    # Missing required field in EmissionLayer (internal ValueError caught and re-raised as CeceConfigError)
    config_dict = {
        "species": {
            "CO": [{"operation": "invalid"}]
        }
    }
    with pytest.raises(CeceConfigError, match="Invalid configuration dict"):
        load_config(config_dict)


def test_load_config_with_yaml_file(tmp_path):
    config_dict = {
        "species": {
            "CO": [{"field": "CO_FIELD", "operation": "add", "scale": 1.0}]
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    result = load_config(str(config_file))
    assert isinstance(result, CeceConfig)
    assert "CO" in result.species


def test_load_config_with_invalid_yaml_file(tmp_path):
    config_file = tmp_path / "invalid_config.yaml"
    with open(config_file, "w") as f:
        f.write("invalid: [yaml: content")

    with pytest.raises(CeceConfigError, match="Failed to load config from file"):
        load_config(str(config_file))


def test_load_config_with_yaml_string():
    yaml_str = """
species:
  CO:
    - field: CO_FIELD
      operation: add
      scale: 1.0
"""
    result = load_config(yaml_str)
    assert isinstance(result, CeceConfig)
    assert "CO" in result.species


def test_load_config_with_invalid_yaml_string():
    yaml_str = "invalid: [yaml: content"
    with pytest.raises(CeceConfigError, match="Failed to parse config YAML"):
        load_config(yaml_str)


def test_load_config_with_invalid_type():
    with pytest.raises(CeceConfigError, match="Invalid config type"):
        load_config(123)


def test_validate_array_dimensions():
    array = np.zeros((2, 3, 4))
    # Should not raise
    validate_array_dimensions(array, (2, 3, 4))

    with pytest.raises(ValueError, match="Array shape \(2, 3, 4\) doesn't match expected \(1, 2, 3\)"):
        validate_array_dimensions(array, (1, 2, 3))


def test_validate_array_dtype():
    array_ok = np.zeros((2, 2), dtype=np.float64)
    # Should not raise
    validate_array_dtype(array_ok)

    array_bad = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="Array dtype must be float64, got float32"):
        validate_array_dtype(array_bad)


def test_validate_array_contiguity():
    # C-contiguous
    arr_c = np.zeros((2, 2), order='C')
    validate_array_contiguity(arr_c)

    # F-contiguous
    arr_f = np.zeros((2, 2), order='F')
    validate_array_contiguity(arr_f)

    # Non-contiguous
    arr_non = np.zeros((4, 4))[::2, ::2]
    assert not arr_non.flags["C_CONTIGUOUS"]
    assert not arr_non.flags["F_CONTIGUOUS"]
    with pytest.raises(ValueError, match="Array must be C-contiguous or Fortran-contiguous"):
        validate_array_contiguity(arr_non)


def test_dict_to_yaml():
    d = {"a": 1, "b": [2, 3]}
    yaml_str = dict_to_yaml(d)
    assert "a: 1" in yaml_str
    assert "b:" in yaml_str
    # Round trip
    assert yaml.safe_load(yaml_str) == d


def test_yaml_to_dict():
    yaml_str = "a: 1\nb: [2, 3]"
    d = yaml_to_dict(yaml_str)
    assert d == {"a": 1, "b": [2, 3]}


def test_yaml_to_dict_invalid():
    # Not a dictionary
    with pytest.raises(ValueError, match="YAML must represent a dictionary"):
        yaml_to_dict("- item1\n- item2")

    # Malformed YAML
    with pytest.raises(ValueError, match="Invalid YAML"):
        yaml_to_dict("a: : b")


def test_setup_logging(caplog):
    setup_logging("DEBUG")
    logger = logging.getLogger("cece_test")
    with caplog.at_level(logging.DEBUG):
        logger.debug("test message")
    assert "test message" in caplog.text


def test_setup_logging_invalid():
    with pytest.raises(ValueError, match="Invalid log level: INVALID"):
        setup_logging("INVALID")


def test_get_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
