from loguru import logger

import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium import Component
from vivarium.framework.components import ComponentConfigurationParser
from vivarium.framework.components.parser import ParsingError
from vivarium_public_health.disease import DiseaseModel, DiseaseState, SusceptibleState
from vivarium_public_health.results import DiseaseObserver
from vivarium_public_health.results.risk import CategoricalRiskObserver
from vivarium_public_health.risks.effect import NonLogLinearRiskEffect, RiskEffect
from vivarium_public_health.risks.base_risk import Risk

CAUSE_KEY = "causes"
RISK_KEY = "risks"
DEFAULT_SIS_CONFIG = {"duration": 1, "number": 1, "observers": False}
DEFAULT_RISK_CONFIG = {
    "number": 1,
    "observers": False,
    "distribution_type": None,
    "affected_causes": {},
}
_CONTINUOUS_DISTRIBUTIONS = {"normal", "lognormal", "ensemble"}
_CATEGORICAL_DISTRIBUTIONS = {"dichotomous", "ordered_polytomous", "unordered_polytomous"}


class MultiComponentParsingErrors(ParsingError):
    """Error raised when there are any errors parsing a multi-configuration."""

    def __init__(self, messages: list[str]):
        super().__init__("\n - " + "\n - ".join(messages))


class MultiComponentParser(ComponentConfigurationParser):
    """Parser for multi-component configurations.

    Component configuration parser that can automatically generate multiple
    instances of components based on a multi-configuration. Currently implements
    disease models as SIS_fixed_duration and optionally creates disease observers.

    Example configuration:

    .. code-block:: yaml

        components:
            causes:
                lower_respiratory_infections:
                    number: 5
                    duration: 28
                    observers: True
                ischemic_heart_disease:
                    number: 3
                    duration: 14
                    observers: False

    This will create disease components named:
    - lower_respiratory_infections_1, lower_respiratory_infections_2, ..., lower_respiratory_infections_5
    - ischemic_heart_disease_1, ischemic_heart_disease_2, ischemic_heart_disease_3

    And if observers: True, it will also create corresponding DiseaseObserver components.
    """

    def parse_component_config(self, component_config: LayeredConfigTree) -> list[Component]:
        """Parses the component configuration and returns a list of components.

        This method looks for a `causes` key that contains multi-configuration
        for disease components where each cause name is a key with its own
        multi-parameters. We then separately parse standard (i.e. non-multi-) components.

        Parameters
        ----------
        component_config
            A LayeredConfigTree defining the components to initialize.

        Returns
        -------
            A list of initialized components.

        Raises
        ------
        MultiComponentParsingErrors
            If the multi-configuration is invalid
        """
        components = []

        # Parse standard components first so we can extract disease models for validation
        standard_component_config = component_config.to_dict()
        standard_component_config.pop(CAUSE_KEY, None)
        standard_component_config.pop(RISK_KEY, None)
        standard_components = (
            self.process_level(standard_component_config, [])
            if standard_component_config
            else []
        )
        components += standard_components

        # Extract normally-defined disease causes
        standard_causes = self._extract_disease_causes(standard_components)

        if CAUSE_KEY in component_config:
            causes_config = component_config[CAUSE_KEY]
            self._validate_causes_config(causes_config)
            components += self._get_multi_disease_components(causes_config)

        if RISK_KEY in component_config:
            risks_config = component_config[RISK_KEY]
            causes_config = component_config.get(CAUSE_KEY)
            self._validate_risks_config(risks_config, causes_config, standard_causes)
            components += self._get_multi_risk_components(
                risks_config, causes_config, standard_causes
            )

        return components

    def _get_multi_disease_components(
        self, causes_config: LayeredConfigTree
    ) -> list[Component]:
        """Creates multiple disease components based on multi-configuration.

        Parameters
        ----------
        causes_config
            A LayeredConfigTree defining the disease multi-configuration
            where each cause name is a key with multi-parameters

        Returns
        -------
            A list of initialized disease components and optionally disease observers
        """
        components = []

        # Iterate over each cause in the configuration
        for cause_name, cause_config in causes_config.items():

            number = cause_config.get("number", DEFAULT_SIS_CONFIG["number"])
            duration = cause_config.get("duration", DEFAULT_SIS_CONFIG["duration"])
            observers = cause_config.get("observers", DEFAULT_SIS_CONFIG["observers"])

            for i in range(number):
                components.append(
                    self._create_sis_fixed_duration(cause_name, duration, i + 1)
                )

                if observers:
                    components.append(DiseaseObserver(f"{cause_name}_{i + 1}"))

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

    def _validate_causes_config(self, causes_config: LayeredConfigTree) -> None:
        """Validates the diseases multi-configuration.

        Parameters
        ----------
        causes_config
            A LayeredConfigTree defining the diseases multi-configuration
            where each cause name is a key with multi-parameters

        Raises
        ------
        MultiComponentParsingErrors
            If the diseases multi-configuration is invalid
        """
        error_messages = []

        # Validate each cause configuration
        for cause_name, cause_config in causes_config.items():
            cause_errors = self._validate_cause_config(cause_name, cause_config)
            if cause_errors:
                error_messages.extend(
                    [
                        f"Error in cause '{cause_name}': {str(cause_error)}"
                        for cause_error in cause_errors
                    ]
                )

        if error_messages:
            raise MultiComponentParsingErrors(error_messages)

    def _validate_cause_config(
        self, cause_name: str, cause_config: LayeredConfigTree
    ) -> None:
        """Validates the configuration for a single cause.

        Parameters
        ----------
        cause_name
            The name of the cause
        cause_config
            A LayeredConfigTree defining the cause multi-configuration

        Raises
        ------
        MultiComponentParsingErrors
            If the cause multi-configuration is invalid
        """
        cause_config_dict = cause_config.to_dict()
        error_messages = []

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

        # Validate observers if provided
        if "observers" in cause_config_dict:
            observers = cause_config_dict["observers"]
            if not isinstance(observers, bool):
                error_messages.append("Observers must be a boolean value (True or False)")

        return error_messages

    def _get_multi_risk_components(
        self,
        risks_config: LayeredConfigTree,
        causes_config: LayeredConfigTree | None,
        standard_causes: set[str],
    ) -> list[Component]:
        components: list[Component] = []
        cause_counts = self._get_cause_counts(causes_config, standard_causes)

        for risk_name, risk_config in risks_config.items():
            number = int(risk_config.get("number", DEFAULT_RISK_CONFIG["number"]))
            distribution_type = risk_config.get(
                "distribution_type", DEFAULT_RISK_CONFIG["distribution_type"]
            )
            observers = risk_config.get("observers", DEFAULT_RISK_CONFIG["observers"])
            affected_causes = risk_config.get(
                "affected_causes", DEFAULT_RISK_CONFIG["affected_causes"]
            )

            risk_identifier = self._get_risk_identifier(risk_name)
            for i in range(number):
                suffixed_risk = f"{risk_identifier}_{i + 1}"
                components.append(Risk(suffixed_risk))
                components.extend(
                    self._build_risk_effects(
                        suffixed_risk, affected_causes, cause_counts
                    )
                )

                if observers:
                    if self._is_continuous_distribution(distribution_type):
                        logger.info(
                            "Skipping categorical risk observer for continuous risk '%s'",
                            suffixed_risk,
                        )
                    else:
                        components.append(
                            CategoricalRiskObserver(
                                self._get_risk_name_only(suffixed_risk)
                            )
                        )

        return components

    def _build_risk_effects(
        self,
        suffixed_risk: str,
        affected_causes: dict,
        cause_counts: dict[str, int],
    ) -> list[Component]:
        components: list[Component] = []
        for cause_name, cause_config in affected_causes.items():
            effect_number = int(cause_config.get("number", cause_counts.get(cause_name, 1)))
            effect_type = self._normalize_effect_type(
                cause_config.get("effect_type", "loglinear")
            ) or "loglinear"
            target_measure = cause_config.get("measure", "incidence_rate")

            effect_cls = NonLogLinearRiskEffect if effect_type == "nonloglinear" else RiskEffect

            for i in range(effect_number):
                components.append(
                    effect_cls(
                        suffixed_risk,
                        f"cause.{cause_name}_{i + 1}.{target_measure}",
                    )
                )

        return components

    def _validate_risks_config(
        self,
        risks_config: LayeredConfigTree,
        causes_config: LayeredConfigTree | None,
        standard_causes: set[str],
    ) -> None:
        error_messages = []
        cause_counts = self._get_cause_counts(causes_config, standard_causes)

        for risk_name, risk_config in risks_config.items():
            risk_errors = self._validate_risk_config(risk_name, risk_config, cause_counts)
            if risk_errors:
                error_messages.extend(
                    [
                        f"Error in risk '{risk_name}': {str(risk_error)}"
                        for risk_error in risk_errors
                    ]
                )

        if error_messages:
            raise MultiComponentParsingErrors(error_messages)

    def _validate_risk_config(
        self,
        risk_name: str,
        risk_config: LayeredConfigTree,
        cause_counts: dict[str, int],
    ) -> list[str]:
        risk_config_dict = risk_config.to_dict()
        error_messages = []

        if "number" in risk_config_dict:
            try:
                number = int(risk_config_dict["number"])
                if number <= 0:
                    error_messages.append("Number of components must be positive")
            except (ValueError, TypeError):
                error_messages.append("Number of components must be a valid integer")

        if "distribution_type" in risk_config_dict:
            distribution_type = risk_config_dict["distribution_type"]
            if distribution_type is not None and not isinstance(distribution_type, str):
                error_messages.append("Distribution type must be a string if provided")
            elif distribution_type is not None:
                normalized = distribution_type.lower()
                if normalized not in _CONTINUOUS_DISTRIBUTIONS | _CATEGORICAL_DISTRIBUTIONS:
                    error_messages.append(
                        "Distribution type must be one of "
                        f"{_CONTINUOUS_DISTRIBUTIONS | _CATEGORICAL_DISTRIBUTIONS}"
                    )

        if "observers" in risk_config_dict:
            observers = risk_config_dict["observers"]
            if not isinstance(observers, bool):
                error_messages.append("Observers must be a boolean value (True or False)")

        affected_causes = risk_config_dict.get("affected_causes", {})
        if not isinstance(affected_causes, dict):
            error_messages.append("affected_causes must be a dictionary of cause configs")
            return error_messages

        for cause_name, cause_config in affected_causes.items():
            if cause_name not in cause_counts:
                error_messages.append(
                    f"Affected cause '{cause_name}' is not defined. "
                    f"Define it either in the 'causes' multi-config block or as a standard component."
                )
                continue
            
            cause_count = cause_counts[cause_name]

            if not isinstance(cause_config, dict):
                error_messages.append(
                    f"Configuration for affected cause '{cause_name}' must be a dictionary"
                )
                continue

            if "number" in cause_config:
                try:
                    number = int(cause_config["number"])
                    if number <= 0:
                        error_messages.append(
                            f"Number of affected causes for '{cause_name}' must be positive"
                        )
                    elif number > cause_count:
                        error_messages.append(
                            f"Number of affected causes for '{cause_name}' exceeds available causes"
                        )
                except (ValueError, TypeError):
                    error_messages.append(
                        f"Number of affected causes for '{cause_name}' must be a valid integer"
                    )

            effect_type = cause_config.get("effect_type", "loglinear")
            if self._normalize_effect_type(effect_type) is None:
                error_messages.append(
                    f"Effect type '{effect_type}' for cause '{cause_name}' is not supported"
                )

            if "measure" in cause_config and not isinstance(cause_config["measure"], str):
                error_messages.append(
                    f"Measure for affected cause '{cause_name}' must be a string if provided"
                )

        return error_messages

    @staticmethod
    def _get_cause_counts(
        causes_config: LayeredConfigTree | None, standard_causes: set[str]
    ) -> dict[str, int]:
        """Get counts for all causes (multi-config and normally-defined).
        
        Parameters
        ----------
        causes_config
            Multi-config causes from the 'causes' key
        standard_causes
            Set of cause names from standard component definitions
            
        Returns
        -------
            Dictionary mapping cause names to their instance counts
        """
        cause_counts = {}
        
        # Add multi-config causes with their specified counts
        if causes_config:
            cause_counts.update({
                cause_name: int(cause_config.get("number", DEFAULT_SIS_CONFIG["number"]))
                for cause_name, cause_config in causes_config.items()
            })
        
        # Add normally-defined causes with count = 1
        for cause_name in standard_causes:
            if cause_name not in cause_counts:
                cause_counts[cause_name] = 1
        
        return cause_counts

    @staticmethod
    def _get_risk_identifier(risk_name: str) -> str:
        return risk_name if "." in risk_name else f"risk_factor.{risk_name}"

    @staticmethod
    def _get_risk_name_only(risk_identifier: str) -> str:
        return risk_identifier.split(".", maxsplit=1)[-1]

    @staticmethod
    def _normalize_effect_type(effect_type: str | None) -> str | None:
        if effect_type is None:
            return "loglinear"
        normalized = (
            effect_type.replace("-", "").replace("_", "").replace(" ", "").lower()
        )
        if normalized in {"loglinear"}:
            return "loglinear"
        if normalized in {"nonloglinear"}:
            return "nonloglinear"
        return None

    @staticmethod
    def _extract_disease_causes(components: list[Component]) -> set[str]:
        """Extract disease/cause names from DiseaseModel components.
        
        Parameters
        ----------
        components
            List of parsed components
            
        Returns
        -------
            Set of cause names found in DiseaseModel components
        """
        from vivarium_public_health.disease import DiseaseModel
        
        causes = set()
        for component in components:
            if isinstance(component, DiseaseModel):
                # Extract cause name from the component's state_column
                # which is in format like "lower_respiratory_infections" or "lower_respiratory_infections_1"
                cause_name = component.state_column
                # Remove numeric suffix if present
                if "_" in cause_name and cause_name.rsplit("_", 1)[-1].isdigit():
                    cause_name = cause_name.rsplit("_", 1)[0]
                causes.add(cause_name)
        return causes

    @staticmethod
    def _is_continuous_distribution(distribution_type: str | None) -> bool:
        if distribution_type is None:
            return False
        return distribution_type.lower() in _CONTINUOUS_DISTRIBUTIONS
