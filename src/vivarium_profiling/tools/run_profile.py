#!/usr/bin/env python3
"""Standalone script for profiling Vivarium simulations."""

import argparse
import ast
import cProfile
import sys

from vivarium.framework.engine import SimulationContext


def run_profile_scalene(model_specification_path: str, configuration_override: dict):
    """Run a Vivarium simulation for scalene profiling purposes."""
    from scalene.scalene_profiler import enable_profiling

    sim = SimulationContext(model_specification_path, configuration=configuration_override)
    with enable_profiling():
        sim.run_simulation()


def run_profile_cprofile(
    model_specification_path: str, configuration_override: dict, output_file: str
):
    """Run a Vivarium simulation for cProfile profiling purposes."""
    sim = SimulationContext(model_specification_path, configuration=configuration_override)

    with cProfile.Profile() as profiler:
        sim.run_simulation()
    profiler.dump_stats(output_file)


def main():
    parser = argparse.ArgumentParser(description="Run Vivarium simulation for profiling")
    parser.add_argument("model_specification", help="Path to model specification file")
    parser.add_argument(
        "--config-override", default="{}", help="Configuration override as JSON/dict string"
    )
    parser.add_argument(
        "--profiler",
        choices=["scalene", "cprofile"],
        default="scalene",
        help="Profiling backend to use",
    )
    parser.add_argument(
        "--output", help="Output file for cProfile stats (required when using cprofile)"
    )

    args = parser.parse_args()

    # Parse the configuration override
    try:
        configuration_override = ast.literal_eval(args.config_override)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing config override: {e}", file=sys.stderr)
        sys.exit(1)

    if args.profiler == "scalene":
        run_profile_scalene(args.model_specification, configuration_override)
    elif args.profiler == "cprofile":
        if not args.output:
            print("Error: --output is required when using cprofile", file=sys.stderr)
            sys.exit(1)
        run_profile_cprofile(args.model_specification, configuration_override, args.output)


if __name__ == "__main__":
    main()
