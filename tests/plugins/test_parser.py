from layered_config_tree import LayeredConfigTree
from vivarium_public_health.disease import DiseaseModel

from vivarium_profiling.plugins.parser import ScalingComponentParser


def test_scaling_parser():
    """Test the ScalingComponentParser with a simple configuration."""

    # Create test configuration
    config_dict = {
        "causes": {
            "lower_respiratory_infections": {
                "number": 3,
                "duration": "28",
            }
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


def test_scaling_parser_multiple_causes():
    """Test the ScalingComponentParser with multiple different causes."""

    # Create test configuration with multiple causes
    config_dict = {
        "causes": {
            "lower_respiratory_infections": {
                "number": 2,
                "duration": "28",
            },
            "ischemic_heart_disease": {
                "number": 3,
                "duration": "14",
            },
        }
    }

    config = LayeredConfigTree(config_dict)

    # Create parser and parse configuration
    parser = ScalingComponentParser()
    components = parser.parse_component_config(config)

    assert len(components) == 5, f"Expected 5 components, got {len(components)}"

    # Check that we have the right component names
    component_names = [component.name for component in components]
    expected_names = [
        "disease_model.lower_respiratory_infections_1",
        "disease_model.lower_respiratory_infections_2",
        "disease_model.ischemic_heart_disease_1",
        "disease_model.ischemic_heart_disease_2",
        "disease_model.ischemic_heart_disease_3",
    ]

    # Sort both lists to make comparison order-independent
    component_names.sort()
    expected_names.sort()

    assert (
        component_names == expected_names
    ), f"Expected {expected_names}, got {component_names}"

    # Verify that all components are DiseaseModel instances
    for component in components:
        assert isinstance(component, DiseaseModel)

    # Check specific mortality sources for each cause type
    lri_components = [c for c in components if "lower_respiratory_infections" in c.name]
    ihd_components = [c for c in components if "ischemic_heart_disease" in c.name]

    assert len(lri_components) == 2
    assert len(ihd_components) == 3

    for component in lri_components:
        assert (
            component._csmr_source
            == "cause.lower_respiratory_infections.cause_specific_mortality_rate"
        )

    for component in ihd_components:
        assert (
            component._csmr_source
            == "cause.ischemic_heart_disease.cause_specific_mortality_rate"
        )
