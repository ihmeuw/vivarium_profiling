from layered_config_tree import LayeredConfigTree
from vivarium_public_health.disease import DiseaseModel

from vivarium_profiling.plugins.parser import MultiComponentParser


def test_multi_component_parser():
    """Test the MultiComponentParser with multiple different causes."""

    # Create test configuration with multiple causes
    config_dict = {
        "causes": {
            "lower_respiratory_infections": {
                "number": 2,
                "duration": "28",
            },
            "ischemic_stroke": {
                "number": 3,
                "duration": "14",
            },
        }
    }

    config = LayeredConfigTree(config_dict)

    # Create parser and parse configuration
    parser = MultiComponentParser()
    components = parser.parse_component_config(config)

    assert len(components) == 5

    # Check that we have the right component names
    component_names = [component.name for component in components]
    expected_names = [
        "disease_model.lower_respiratory_infections_1",
        "disease_model.lower_respiratory_infections_2",
        "disease_model.ischemic_stroke_1",
        "disease_model.ischemic_stroke_2",
        "disease_model.ischemic_stroke_3",
    ]

    assert component_names == expected_names

    # Verify that all components are DiseaseModel instances
    for component in components:
        assert isinstance(component, DiseaseModel)

    # Check specific mortality sources for each cause type
    lri_components = [c for c in components if "lower_respiratory_infections" in c.name]
    ihd_components = [c for c in components if "ischemic_stroke" in c.name]

    assert len(lri_components) == 2
    assert len(ihd_components) == 3

    for component in lri_components:
        assert (
            component._csmr_source
            == "cause.lower_respiratory_infections.cause_specific_mortality_rate"
        )
        disease_state = component.states[1]
        assert (
            disease_state._prevalence_source
            == "cause.lower_respiratory_infections.prevalence"
        )

    for component in ihd_components:
        assert component._csmr_source == "cause.ischemic_stroke.cause_specific_mortality_rate"
        disease_state = component.states[1]
        assert disease_state._prevalence_source == "cause.ischemic_stroke.prevalence"
        breakpoint()
