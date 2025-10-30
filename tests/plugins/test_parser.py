from layered_config_tree import LayeredConfigTree
from vivarium_public_health.disease import DiseaseModel

from vivarium_profiling.plugins.parser import ScalingComponentParser


def test_scaling_parser():
    """Test the ScalingComponentParser with a simple configuration."""

    # Create test configuration
    config_dict = {
        "causes": {
            "cause": "lower_respiratory_infections",
            "number": 3,
            "duration": "28",
        }
    }

    config = LayeredConfigTree(config_dict)

    # Create parser and parse configuration
    parser = ScalingComponentParser()
    components = parser.parse_component_config(config)

    assert len(components) == 3, f"Expected 3 components, got {len(components)}"

    expected_names = [
        "disease_model.lower_respiratory_infections_1",
        "disease_model.lower_respiratory_infections_2",
        "disease_model.lower_respiratory_infections_3",
    ]

    for i, component in enumerate(components):
        expected_name = expected_names[i]
        assert (
            component.name == expected_name
        ), f"Expected {expected_name}, got {component.name}"

        assert isinstance(component, DiseaseModel)
        assert (
            component._csmr_source
            == "cause.lower_respiratory_infections.cause_specific_mortality_rate"
        )
