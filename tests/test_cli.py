import json
from pathlib import Path
from unittest.mock import patch
from typing import List

import pytest
import yaml
from click.testing import CliRunner

from vivarium_profiling.tools.cli import profile_sim


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def model_spec(test_model_specs: list[Path]) -> str:
    """Create a minimal test model specification file."""
    return str(test_model_specs[0])


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Create a temporary results directory."""
    return tmp_path / "results"


def run_profile_sim_command(
    runner: CliRunner,
    model_spec: str,
    results_dir: Path,
    extra_args: List[str] | None = None,
) -> tuple[Path, object]:
    """Helper function to run profile_sim command with common setup.

    Returns:
        tuple: (results_dir, click_result)
    """

    if extra_args is None:
        extra_args = []

    args = [model_spec, "-o", str(results_dir)] + extra_args
    result = runner.invoke(profile_sim, args)
    assert result.exit_code == 0


def test_profile_sim_parameters():
    """Test that profile_sim has the expected parameters."""
    expected_parameters = {
        "model_specification",
        "results_directory",
        "skip_writing",
        "skip_processing",
        "profiler",
    }
    actual_parameters = {param.name for param in profile_sim.params}
    assert actual_parameters == expected_parameters


@patch("subprocess.run")
def test_profile_sim_scalene_default(
    mock_subprocess_run, runner: CliRunner, model_spec: str, results_dir: Path
):
    """Test profile_sim with scalene profiler (default)."""
    # Mock successful subprocess run
    mock_subprocess_run.return_value = None

    # Run the command
    run_profile_sim_command(runner, model_spec, results_dir)

    # Verify subprocess was called with correct scalene command
    mock_subprocess_run.assert_called_once()
    call_args = mock_subprocess_run.call_args[0][0]

    # Check that scalene command was constructed correctly
    assert call_args[0] == "scalene"
    assert "--json" in call_args
    assert "--outfile" in call_args
    assert "--off" in call_args
    assert model_spec in call_args
    assert "--config-override" in call_args

    # Check that outfile path contains .json extension
    outfile_idx = call_args.index("--outfile") + 1
    outfile_path = call_args[outfile_idx]
    assert outfile_path.endswith(".json")


@patch("subprocess.run")
@patch("pstats.Stats")
def test_profile_sim_cprofile_with_processing(
    mock_pstats, mock_subprocess_run, runner: CliRunner, model_spec: str, results_dir: Path
):
    """Test profile_sim with cprofile and post-processing enabled."""
    mock_subprocess_run.return_value = None

    mock_stats_instance = mock_pstats.return_value

    run_profile_sim_command(runner, model_spec, results_dir, ["--profiler", "cprofile"])

    # Verify that pstats was used for processing
    mock_pstats.assert_called_once()
    mock_stats_instance.sort_stats.assert_called_once_with("cumulative")
    mock_stats_instance.print_stats.assert_called_once()


@patch("subprocess.run")
def test_profile_sim_cprofile_skip_processing(
    mock_subprocess_run, runner: CliRunner, model_spec: str, results_dir: Path
):
    """Test profile_sim with cprofile and processing skipped."""
    mock_subprocess_run.return_value = None

    run_profile_sim_command(
        runner, model_spec, results_dir, ["--profiler", "cprofile", "--skip_processing"]
    )


@patch("subprocess.run")
def test_profile_sim_skip_writing(
    mock_subprocess_run, runner: CliRunner, model_spec: str, results_dir: Path
):
    """Test profile_sim with skip_writing flag."""
    # Mock successful subprocess run
    mock_subprocess_run.return_value = None

    # Run the command with skip_writing
    run_profile_sim_command(runner, model_spec, results_dir, ["--skip_writing"])

    # Verify the config override was empty (indicating no output directory set)
    call_args = mock_subprocess_run.call_args[0][0]
    config_override_idx = call_args.index("--config-override") + 1
    config_override = call_args[config_override_idx]
    assert config_override == "{}"


@patch("subprocess.run")
def test_profile_sim_results_directory_structure(
    mock_subprocess_run, runner: CliRunner, model_spec: str, results_dir: Path
):
    """Test that profile_sim creates the expected directory structure and file paths."""
    mock_subprocess_run.return_value = None
    run_profile_sim_command(runner, model_spec, results_dir)

    # Check that results directory exists and has timestamped subdirectory
    assert results_dir.exists()
    timestamped_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    assert len(timestamped_dirs) == 1

    # Verify the directory name follows expected timestamp format (YYYY_MM_DD_HH_MM_SS)
    dir_name = timestamped_dirs[0].name
    parts = dir_name.split("_")
    assert len(parts) == 6  # year, month, day, hour, minute, second
    assert all(part.isdigit() for part in parts)


@patch("subprocess.run")
def test_profile_sim_extra_args(
    mock_subprocess_run, runner: CliRunner, model_spec: str, results_dir: Path
):
    """Test that profile_sim passes extra arguments to scalene."""
    mock_subprocess_run.return_value = None
    run_profile_sim_command(runner, model_spec, results_dir, ["--cpu-only", "--html"])

    # Verify extra arguments were passed to scalene
    call_args = mock_subprocess_run.call_args[0][0]
    assert "--cpu-only" in call_args
    assert "--html" in call_args
