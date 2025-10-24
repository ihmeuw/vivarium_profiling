from pathlib import Path

import pytest

from vivarium_profiling.tools.run_profile import (
    run_profile_cprofile,
    run_profile_scalene,
)


@pytest.fixture
def model_spec(test_model_specs: list[Path]) -> str:
    """Get path to a test model specification file."""
    return str(test_model_specs[0])


def test_run_profile_scalene_uses_correct_context_manager(mocker, model_spec: str):
    """Test that run_profile_scalene uses scalene's enable_profiling context manager."""
    # Setup mocks
    mock_sim = mocker.MagicMock()
    mock_simulation_context = mocker.patch(
        "vivarium_profiling.tools.run_profile.SimulationContext"
    )
    mock_simulation_context.return_value = mock_sim

    mock_context_manager = mocker.MagicMock()
    mock_enable_profiling = mocker.patch("scalene.scalene_profiler.enable_profiling")
    mock_enable_profiling.return_value = mock_context_manager

    # Run the function
    configuration_override = {"test": "config"}
    run_profile_scalene(model_spec, configuration_override)

    # Verify SimulationContext was created with correct parameters
    mock_simulation_context.assert_called_once_with(
        model_spec, configuration=configuration_override
    )

    # Verify scalene's enable_profiling context manager was used
    mock_enable_profiling.assert_called_once()
    mock_context_manager.__enter__.assert_called_once()
    mock_context_manager.__exit__.assert_called_once()

    # Verify simulation was run
    mock_sim.run_simulation.assert_called_once()


def test_run_profile_cprofile_uses_correct_context_manager(
    mocker, model_spec: str, tmp_path: Path
):
    """Test that run_profile_cprofile uses cProfile.Profile context manager."""
    # Setup mocks
    mock_sim = mocker.MagicMock()
    mock_simulation_context = mocker.patch(
        "vivarium_profiling.tools.run_profile.SimulationContext"
    )
    mock_simulation_context.return_value = mock_sim

    mock_profiler = mocker.MagicMock()
    mock_profile_class = mocker.patch("cProfile.Profile")
    mock_profile_class.return_value = mock_profiler

    # The context manager should return the same profiler instance
    mock_profiler.__enter__.return_value = mock_profiler

    # Create output file path
    output_file = str(tmp_path / "output.stats")

    # Run the function
    configuration_override = {"test": "config"}
    run_profile_cprofile(model_spec, configuration_override, output_file)

    # Verify SimulationContext was created with correct parameters
    mock_simulation_context.assert_called_once_with(
        model_spec, configuration=configuration_override
    )

    # Verify cProfile.Profile context manager was used
    mock_profile_class.assert_called_once()
    mock_profiler.__enter__.assert_called_once()
    mock_profiler.__exit__.assert_called_once()

    # Verify simulation was run
    mock_sim.run_simulation.assert_called_once()

    # Verify stats were dumped to output file
    mock_profiler.dump_stats.assert_called_once_with(output_file)
