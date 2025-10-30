import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium import Component
from vivarium.framework.components import ComponentConfigurationParser
from vivarium.framework.components.parser import ParsingError

from vivarium_public_health.disease import (
    DiseaseModel,
    DiseaseState,
    SusceptibleState,
)

CAUSE_KEY = "causes"

class ScalingParsingErrors(ParsingError):
    """Error raised when there are any errors parsing a scaling configuration."""

    def __init__(self, messages: list[str]):
        super().__init__("\n - " + "\n - ".join(messages))


class ScalingComponentParser(ComponentConfigurationParser):
    """Parser for scaling component configurations.

    Component configuration parser that can automatically generate multiple
    instances of components based on a scaling configuration. Currently supports
    disease models like SIS_fixed_duration.

    Example configuration:

    .. code-block:: yaml

        components:
            causes:
                number: 5
                cause: lower_respiratory_infections
                duration: 28

    This will create 5 disease components named:
    - lower_respiratory_infections_1
    - lower_respiratory_infections_2
    - lower_respiratory_infections_3
    - lower_respiratory_infections_4
    - lower_respiratory_infections_5
    """

    def parse_component_config(self, component_config: LayeredConfigTree) -> list[Component]:
        """Parses the component configuration and returns a list of components.

        This method looks for a `diseases` key that contains scaling configuration
        for disease components.

        Parameters
        ----------
        component_config
            A LayeredConfigTree defining the components to initialize.

        Returns
        -------
            A list of initialized components.

        Raises
        ------
        ScalingParsingErrors
            If the scaling configuration is invalid
        """
        components = []

        if CAUSE_KEY in component_config:
            diseases_config = component_config[CAUSE_KEY]
            self._validate_diseases_config(diseases_config)
            components += self._get_scaled_disease_components(diseases_config)

        # Parse standard components (i.e. not scaled components)
        standard_component_config = component_config.to_dict()
        standard_component_config.pop(CAUSE_KEY, None)
        standard_components = (
            self.process_level(standard_component_config, [])
            if standard_component_config
            else []
        )

        return components + standard_components

    def _get_scaled_disease_components(
        self, diseases_config: LayeredConfigTree
    ) -> list[Component]:
        """Creates multiple disease components based on scaling configuration.

        Parameters
        ----------
        diseases_config
            A LayeredConfigTree defining the disease scaling configuration

        Returns
        -------
            A list of initialized disease components
        """
        components = []

        base_cause = diseases_config.get("cause")
        number = diseases_config.get("number", 1)
        duration = diseases_config.get("duration", "28")

        for i in range(number):
            cause_name = f"{base_cause}_{i+1}"
            disease_component = self._create_sis_fixed_duration(
                cause_name, duration, base_cause
            )
            components.append(disease_component)

        return components

    def _create_sis_fixed_duration(
        self, cause: str, duration: str, base_cause: str
    ) -> DiseaseModel:
        """Creates a SIS fixed duration disease model.

        Parameters
        ----------
        cause
            The name of the cause/disease (with suffix)
        duration
            The duration string (in days)
        base_cause
            The base cause name (without suffix) for mortality data

        Returns
        -------
            An initialized DiseaseModel component
        """
        duration_td = pd.Timedelta(
            days=float(duration) // 1, hours=(float(duration) % 1) * 24.0
        )

        healthy = SusceptibleState(cause, allow_self_transition=True)
        infected = DiseaseState(
            cause,
            get_data_functions={"dwell_time": lambda _, __: duration_td},
            allow_self_transition=True,
            prevalence=f"cause.{base_cause}.prevalence",
            disability_weight=f"cause.{base_cause}.disability_weight",
            excess_mortality_rate=f"cause.{base_cause}.excess_mortality_rate",
        )

        healthy.add_rate_transition(
            infected, transition_rate=f"cause.{base_cause}.incidence_rate"
        )
        infected.add_dwell_time_transition(healthy)

        return DiseaseModel(
            cause,  # This is the suffixed name for the component
            states=[healthy, infected],
            cause_specific_mortality_rate=f"cause.{base_cause}.cause_specific_mortality_rate",
        )

    def _validate_diseases_config(self, diseases_config: LayeredConfigTree) -> None:
        """Validates the diseases scaling configuration.

        Parameters
        ----------
        diseases_config
            A LayeredConfigTree defining the diseases scaling configuration

        Raises
        ------
        ScalingParsingErrors
            If the diseases scaling configuration is invalid
        """
        diseases_config_dict = diseases_config.to_dict()
        error_messages = []

        # Check required fields
        required_fields = [CAUSE_KEY, "number"]
        for field in required_fields:
            if field not in diseases_config_dict:
                error_messages.append(f"Missing required field: {field}")

        # Validate number
        if "number" in diseases_config_dict:
            try:
                number = int(diseases_config_dict["number"])
                if number <= 0:
                    error_messages.append("Number of components must be positive")
            except (ValueError, TypeError):
                error_messages.append("Number of components must be a valid integer")

        # Validate duration if provided
        if "duration" in diseases_config_dict:
            try:
                float(diseases_config_dict["duration"])
            except (ValueError, TypeError):
                error_messages.append("Duration must be a valid number")

        if error_messages:
            raise ScalingParsingErrors(error_messages)
