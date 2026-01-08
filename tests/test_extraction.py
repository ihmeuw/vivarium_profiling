"""Unit tests for extraction utilities."""

import tempfile
from pathlib import Path

import pytest

from vivarium_profiling.tools.extraction import (
    CallPattern,
    ExtractionConfig,
    bottleneck_config,
    extract_runtime,
    parse_function_metrics,
    phase_config,
)


class TestCallPattern:
    """Tests for CallPattern dataclass."""

    def test_call_pattern_defaults(self):
        """Test CallPattern with default extraction flags."""
        pattern = CallPattern(name="test_func", filename="test.py", function_name="test_func")

        assert pattern.name == "test_func"
        assert pattern.filename == "test.py"
        assert pattern.function_name == "test_func"
        assert pattern.pattern == r"test\.py:\d+\(test_func\)"
        assert pattern.extract_cumtime is True
        assert pattern.extract_percall is False
        assert pattern.extract_ncalls is False
        assert pattern.cumtime_col == "test_func_cumtime"
        assert pattern.percall_col == "test_func_percall"
        assert pattern.ncalls_col == "test_func_ncalls"
        assert pattern.columns == ["test_func_cumtime"]

    def test_call_pattern_all_extracts(self):
        """Test CallPattern with all extraction flags enabled."""
        pattern = CallPattern(
            name="bottleneck",
            filename="test.py",
            function_name="bottleneck",
            extract_cumtime=True,
            extract_percall=True,
            extract_ncalls=True,
        )

        assert pattern.pattern == r"test\.py:\d+\(bottleneck\)"
        assert pattern.columns == [
            "bottleneck_cumtime",
            "bottleneck_percall",
            "bottleneck_ncalls",
        ]

    def test_call_pattern_custom_templates(self):
        """Test CallPattern with custom column templates."""
        pattern = CallPattern(
            name="phase",
            filename="engine.py",
            function_name="phase",
            cumtime_template="rt_{name}_s",
        )

        assert pattern.pattern == r"engine\.py:\d+\(phase\)"
        assert pattern.cumtime_col == "rt_phase_s"
        assert pattern.columns == ["rt_phase_s"]


class TestExtractionConfig:
    """Tests for ExtractionConfig class."""

    def test_extraction_config_defaults(self):
        """Test ExtractionConfig with default patterns."""
        config = ExtractionConfig()

        assert len(config.patterns) > 0
        assert len(config.metric_names) > 0
        assert len(config.metric_columns) > 0
        assert len(config.results_columns) > 4  # base + metrics

    def test_extraction_config_custom_patterns(self):
        """Test ExtractionConfig with custom patterns."""
        patterns = [
            CallPattern("func1", "test.py", "func1"),
            bottleneck_config("func2", "test.py", "func2"),
        ]
        config = ExtractionConfig(patterns=patterns)

        assert len(config.patterns) == 2
        assert config.metric_names == ["func1", "func2"]  # Only func2 has all extracts

    def test_metric_columns(self):
        """Test metric_columns property."""
        patterns = [
            CallPattern("a", "test.py", "a", extract_cumtime=True, extract_percall=True),
            CallPattern("b", "test.py", "b", extract_cumtime=True),
        ]
        config = ExtractionConfig(patterns=patterns)

        assert config.metric_columns == ["a_cumtime", "a_percall", "b_cumtime"]

    def test_results_columns(self):
        """Test results_columns includes base columns."""
        patterns = [CallPattern("test", "test.py", "test")]
        config = ExtractionConfig(patterns=patterns)

        cols = config.results_columns
        assert cols[:4] == ["model_spec", "run", "rt_s", "mem_mb"]
        assert "test_cumtime" in cols


class TestParseMetrics:
    """Tests for metric parsing functions."""

    def test_parse_function_metrics_basic(self, sample_stats_file):
        """Test parsing basic function metrics."""
        cumtime, percall, ncalls = parse_function_metrics(
            sample_stats_file, r"results/manager\.py:\d+\(gather_results\)"
        )

        assert cumtime == 2.500
        assert percall == 0.016
        assert ncalls == 160

    def test_parse_function_metrics_recursive_calls(self, sample_stats_file):
        """Test parsing function with recursive calls (format: 50/25)."""
        cumtime, percall, ncalls = parse_function_metrics(
            sample_stats_file, r"some/custom/module\.py:\d+\(custom_function\)"
        )

        assert cumtime == 0.600
        assert percall == 0.024
        assert ncalls == 25  # Should extract the primitive call count

    def test_extract_runtime(self, sample_stats_file):
        """Test extracting total runtime from stats file."""
        runtime = extract_runtime(sample_stats_file)

        assert runtime == 10.500

    def test_parse_function_metrics_not_found(self, sample_stats_file):
        """Test parsing when function is not found."""
        cumtime, percall, ncalls = parse_function_metrics(
            sample_stats_file, r"nonexistent\.py:\d+\(missing\)"
        )

        assert cumtime is None
        assert percall is None
        assert ncalls is None

    def test_parsing_invalid_file(self, tmp_path):
        """Test parsing with invalid file path."""
        cumtime, percall, ncalls = parse_function_metrics(
            tmp_path / "nonexistent.txt", r"pattern"
        )

        assert cumtime is None
        assert percall is None
        assert ncalls is None

        runtime = extract_runtime(tmp_path / "nonexistent.txt")

        assert runtime is None


class TestExtractionConfigExtract:
    """Tests for ExtractionConfig.extract_metrics method."""

    def test_extract_metrics_default_patterns(self, sample_stats_file):
        """Test extracting metrics with default patterns."""
        config = ExtractionConfig()
        metrics = config.extract_metrics(sample_stats_file)

        # default phase metrics
        assert metrics["rt_setup_s"] == 8.000
        assert metrics["rt_initialize_simulants_s"] == 2.000
        assert metrics["rt_run_s"] == 5.000
        assert metrics["rt_finalize_s"] == 0.300
        assert metrics["rt_report_s"] == 0.100

        # default bottleneck metrics
        assert metrics["gather_results_cumtime"] == 2.500
        assert metrics["pipeline_call_cumtime"] == 1.200
        assert metrics["population_get_cumtime"] == 0.800

    def test_extract_metrics_custom_patterns(self, sample_stats_file):
        """Test extracting metrics with custom patterns."""
        patterns = [
            CallPattern(
                "custom_func",
                "some/custom/module.py",
                "custom_function",
                extract_cumtime=True,
                extract_ncalls=True,
                extract_percall=False,
            ),
            phase_config("initialize_simulants"),
        ]
        config = ExtractionConfig(patterns=patterns)
        metrics = config.extract_metrics(sample_stats_file)

        assert metrics["custom_func_cumtime"] == 0.600
        assert metrics["custom_func_ncalls"] == 25  # Primitive calls
        assert "custom_func_percall" not in metrics  # Not extracted

        assert metrics["rt_initialize_simulants_s"] == 2.000

    def test_extract_metrics_missing_patterns(self, sample_stats_file):
        """Test extracting metrics when patterns don't match."""
        patterns = [CallPattern("missing", "nonexistent.py", "missing")]
        config = ExtractionConfig(patterns=patterns)
        metrics = config.extract_metrics(sample_stats_file)

        assert metrics["missing_cumtime"] is None


class TestYAMLParsing:
    """Tests for YAML configuration parsing."""

    def test_from_yaml_bottleneck_preset(self):
        """Test parsing YAML with bottleneck preset."""
        yaml_content = """
patterns:
  - name: gather_results
    preset: bottleneck
    filename: results/manager.py
    function_name: gather_results
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            config = ExtractionConfig.from_yaml(yaml_path)
            
            assert len(config.patterns) == 1
            pattern = config.patterns[0]
            assert pattern.name == "gather_results"
            assert pattern.filename == "results/manager.py"
            assert pattern.function_name == "gather_results"
            assert pattern.extract_cumtime is True
            assert pattern.extract_percall is True
            assert pattern.extract_ncalls is True
        finally:
            yaml_path.unlink()

    def test_from_yaml_phase_preset(self):
        """Test parsing YAML with phase preset."""
        yaml_content = """
patterns:
  - name: setup
    preset: phase
  - name: custom_phase
    preset: phase
    filename: /my/custom/engine.py
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            config = ExtractionConfig.from_yaml(yaml_path)
            
            assert len(config.patterns) == 2
            
            # First pattern with default filename
            pattern1 = config.patterns[0]
            assert pattern1.name == "setup"
            assert pattern1.filename == "/vivarium/framework/engine.py"
            assert pattern1.function_name == "setup"
            assert pattern1.extract_cumtime is True
            assert pattern1.extract_percall is False
            assert pattern1.extract_ncalls is False
            assert pattern1.cumtime_template == "rt_{name}_s"
            
            # Second pattern with custom filename
            pattern2 = config.patterns[1]
            assert pattern2.name == "custom_phase"
            assert pattern2.filename == "/my/custom/engine.py"
        finally:
            yaml_path.unlink()

    def test_from_yaml_custom_pattern(self):
        """Test parsing YAML with custom pattern (no preset)."""
        yaml_content = """
patterns:
  - name: my_function
    filename: my/module.py
    function_name: my_function
    extract_cumtime: true
    extract_percall: true
    extract_ncalls: false
    cumtime_template: "custom_{name}_time"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            config = ExtractionConfig.from_yaml(yaml_path)
            
            assert len(config.patterns) == 1
            pattern = config.patterns[0]
            assert pattern.name == "my_function"
            assert pattern.filename == "my/module.py"
            assert pattern.function_name == "my_function"
            assert pattern.extract_cumtime is True
            assert pattern.extract_percall is True
            assert pattern.extract_ncalls is False
            assert pattern.cumtime_col == "custom_my_function_time"
        finally:
            yaml_path.unlink()

    def test_from_yaml_mixed_patterns(self):
        """Test parsing YAML with mixed preset and custom patterns."""
        yaml_content = """
patterns:
  - name: gather_results
    preset: bottleneck
    filename: results/manager.py
    function_name: gather_results
  - name: setup
    preset: phase
  - name: custom_func
    filename: custom.py
    function_name: custom_func
    extract_cumtime: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            config = ExtractionConfig.from_yaml(yaml_path)
            
            assert len(config.patterns) == 3
            assert config.patterns[0].extract_percall is True  # bottleneck
            assert config.patterns[1].cumtime_template == "rt_{name}_s"  # phase
            assert config.patterns[2].extract_cumtime is True  # custom
            assert config.patterns[2].extract_percall is False  # custom default
        finally:
            yaml_path.unlink()

    def test_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="YAML config file not found"):
            ExtractionConfig.from_yaml("/nonexistent/file.yaml")

    def test_from_yaml_missing_patterns_key(self):
        """Test error when YAML is missing 'patterns' key."""
        yaml_content = """
some_other_key:
  - value: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="must contain a 'patterns' key"):
                ExtractionConfig.from_yaml(yaml_path)
        finally:
            yaml_path.unlink()

    def test_from_yaml_missing_name_field(self):
        """Test error when pattern is missing 'name' field."""
        yaml_content = """
patterns:
  - filename: test.py
    function_name: test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="missing required field 'name'"):
                ExtractionConfig.from_yaml(yaml_path)
        finally:
            yaml_path.unlink()

    def test_from_yaml_bottleneck_missing_fields(self):
        """Test error when bottleneck preset is missing required fields."""
        yaml_content = """
patterns:
  - name: my_bottleneck
    preset: bottleneck
    filename: test.py
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError,
                match="preset='bottleneck' requires 'filename' and 'function_name'",
            ):
                ExtractionConfig.from_yaml(yaml_path)
        finally:
            yaml_path.unlink()

    def test_from_yaml_custom_missing_fields(self):
        """Test error when custom pattern is missing required fields."""
        yaml_content = """
patterns:
  - name: my_func
    filename: test.py
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(
                ValueError, match="requires 'filename' and 'function_name' fields"
            ):
                ExtractionConfig.from_yaml(yaml_path)
        finally:
            yaml_path.unlink()

    def test_from_yaml_pattern_not_dict(self):
        """Test error when pattern is not a dictionary."""
        yaml_content = """
patterns:
  - not_a_dict
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="must be a dictionary"):
                ExtractionConfig.from_yaml(yaml_path)
        finally:
            yaml_path.unlink()

