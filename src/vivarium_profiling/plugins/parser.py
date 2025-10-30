import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium import Component
from vivarium.framework.components import ComponentConfigurationParser
from vivarium.framework.components.parser import ParsingError
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState

CAUSE_KEY = "causes"


class ScalingParsingErrors(ParsingError):
    """Error raised when there are any errors parsing a scaling configuration."""

    def __init__(self, messages: list[str]):
        super().__init__("\n - " + "\n - ".join(messages))


class ScalingComponentParser(ComponentConfigurationParser):
    """Parser for scaling component configurations.

    Component configuration parser that can automatically generate multiple
    instances of components based on a scaling configuration. Currently implements
    disease models as SIS_fixed_duration.

    Example configuration:

    .. code-block:: yaml

        components:
            causes:
                lower_respiratory_infections:
                    number: 5
                    duration: 28
                ischemic_heart_disease:
                    number: 3
                    duration: 14

    This will create disease components named:
    - lower_respiratory_infections_1, lower_respiratory_infections_2, ..., lower_respiratory_infections_5
    - ischemic_heart_disease_1, ischemic_heart_disease_2, ischemic_heart_disease_3
    """

    def parse_component_config(self, component_config: LayeredConfigTree) -> list[Component]:
        """Parses the component configuration and returns a list of components.

        This method looks for a `causes` key that contains scaling configuration
        for disease components where each cause name is a key with its own
        scaling parameters.

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
            where each cause name is a key with scaling parameters

        Returns
        -------
            A list of initialized disease components
        """
        components = []

        # Iterate over each cause in the configuration
        for cause_name, cause_config in diseases_config.items():
            self._validate_cause_config(cause_name, cause_config)

            number = cause_config.get("number", 1)
            duration = cause_config.get("duration", 1)

            for i in range(number):
                components.append(
                    self._create_sis_fixed_duration(cause_name, duration, i + 1)
                )

        return components

    def _create_sis_fixed_duration(
        self, cause_name: str, duration: str, number: int
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
        suffixed_cause_name = f"{cause_name}_{number}"
        duration_td = pd.Timedelta(
            days=float(duration) // 1, hours=(float(duration) % 1) * 24.0
        )

        healthy = SusceptibleState(suffixed_cause_name, allow_self_transition=True)
        infected = DiseaseState(
            suffixed_cause_name,
            get_data_functions={"dwell_time": lambda _, __: duration_td},
            allow_self_transition=True,
            prevalence=f"cause.{cause_name}.prevalence",
            disability_weight=f"cause.{cause_name}.disability_weight",
            excess_mortality_rate=f"cause.{cause_name}.excess_mortality_rate",
        )

        healthy.add_rate_transition(
            infected, transition_rate=f"cause.{cause_name}.incidence_rate"
        )
        infected.add_dwell_time_transition(healthy)

        return DiseaseModel(
            suffixed_cause_name,  # This is the suffixed name for the component
            states=[healthy, infected],
            cause_specific_mortality_rate=f"cause.{cause_name}.cause_specific_mortality_rate",
        )

    def _validate_diseases_config(self, diseases_config: LayeredConfigTree) -> None:
        """Validates the diseases scaling configuration.

        Parameters
        ----------
        diseases_config
            A LayeredConfigTree defining the diseases scaling configuration
            where each cause name is a key with scaling parameters

        Raises
        ------
        ScalingParsingErrors
            If the diseases scaling configuration is invalid
        """
        error_messages = []

        # Validate each cause configuration
        for cause_name, cause_config in diseases_config.items():
            try:
                self._validate_cause_config(cause_name, cause_config)
            except ScalingParsingErrors as e:
                error_messages.append(f"Error in cause '{cause_name}': {str(e)}")

        if error_messages:
            raise ScalingParsingErrors(error_messages)

    def _validate_cause_config(
        self, cause_name: str, cause_config: LayeredConfigTree
    ) -> None:
        """Validates the configuration for a single cause.

        Parameters
        ----------
        cause_name
            The name of the cause
        cause_config
            A LayeredConfigTree defining the cause scaling configuration

        Raises
        ------
        ScalingParsingErrors
            If the cause scaling configuration is invalid
        """
        cause_config_dict = cause_config.to_dict()
        error_messages = []

        # Check required fields
        required_fields = ["number"]
        for field in required_fields:
            if field not in cause_config_dict:
                error_messages.append(f"Missing required field: {field}")

        # Validate number
        if "number" in cause_config_dict:
            try:
                number = int(cause_config_dict["number"])
                if number <= 0:
                    error_messages.append("Number of components must be positive")
            except (ValueError, TypeError):
                error_messages.append("Number of components must be a valid integer")

        # Validate duration if provided
        if "duration" in cause_config_dict:
            try:
                duration = float(cause_config_dict["duration"])
                if duration <= 0.0:
                    error_messages.append("Duration must be positive")
            except (ValueError, TypeError):
                error_messages.append("Duration must be a valid number")

        if error_messages:
            raise ScalingParsingErrors(error_messages)
