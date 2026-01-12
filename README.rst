===============================
vivarium_profiling
===============================

Vivarium simulation model for the vivarium_profiling project.

.. contents::
   :depth: 1

Installation
------------

You will need ``conda`` to install all of this repository's requirements.
We recommend installing `Miniforge <https://github.com/conda-forge/miniforge>`_.

Once you have conda installed, you should open up your normal shell
(if you're on linux or OSX) or the ``git bash`` shell if you're on windows.
You'll then make an environment, clone this repository, then install
all necessary requirements as follows::

  :~$ conda create --name=vivarium_profiling python=3.11 git git-lfs
  ...conda will download python and base dependencies...
  :~$ conda activate vivarium_profiling
  (vivarium_profiling) :~$ git clone https://github.com/ihmeuw/vivarium_profiling.git
  ...git will copy the repository from github and place it in your current directory...
  (vivarium_profiling) :~$ cd vivarium_profiling
  (vivarium_profiling) :~$ pip install -e .
  ...pip will install vivarium and other requirements...

Supported Python versions: 3.10, 3.11

Note the ``-e`` flag that follows pip install. This will install the python
package in-place, which is important for making the model specifications later.

To install requirements from a provided requirements.txt (e.g. installing an
archived repository with the exact same requirements it was run with), replace
`pip install -e .` with the following::

  (vivarium_profiling) :~$ pip install -r requirements.txt

Cloning the repository should take a fair bit of time as git must fetch
the data artifact associated with the demo (several GB of data) from the
large file system storage (``git-lfs``). **If your clone works quickly,
you are likely only retrieving the checksum file that github holds onto,
and your simulations will fail.** If you are only retrieving checksum
files you can explicitly pull the data by executing ``git-lfs pull``.

Vivarium uses the Hierarchical Data Format (HDF) as the backing storage
for the data artifacts that supply data to the simulation. You may not have
the needed libraries on your system to interact with these files, and this is
not something that can be specified and installed with the rest of the package's
dependencies via ``pip``. If you encounter HDF5-related errors, you should
install hdf tooling from within your environment like so::

  (vivarium_profiling) :~$ conda install hdf5

The ``(vivarium_profiling)`` that precedes your shell prompt will probably show
up by default, though it may not.  It's just a visual reminder that you
are installing and running things in an isolated programming environment
so it doesn't conflict with other source code and libraries on your
system.


Usage
-----

You'll find six directories inside the main
``src/vivarium_profiling`` package directory:

- ``artifacts``

  This directory contains all input data used to run the simulations.
  You can open these files and examine the input data using the vivarium
  artifact tools.  A tutorial can be found at https://vivarium.readthedocs.io/en/latest/tutorials/artifact.html#reading-data

- ``components``

  This directory is for Python modules containing custom components for
  the vivarium_profiling project. You should work with the
  engineering staff to help scope out what you need and get them built.

- ``data``

  If you have **small scale** external data for use in your sim or in your
  results processing, it can live here. This is almost certainly not the right
  place for data, so make sure there's not a better place to put it first.

- ``model_specifications``

  This directory should hold all model specifications and branch files
  associated with the project.

- ``results_processing``

  Any post-processing and analysis code or notebooks you write should be
  stored in this directory.

- ``tools``

  This directory hold Python files used to run scripts used to prepare input
  data or process outputs.


Running Simulations
-------------------

Before running a simulation, you should have a model specification file.
A model specification is a complete description of a vivarium model in
a yaml format.  An example model specification is provided with this repository
in the ``model_specifications`` directory.

With this model specification file and your conda environment active, you can then run simulations by, e.g.::

   (vivarium_profiling) :~$ simulate run -v /<REPO_INSTALLATION_DIRECTORY>/vivarium_profiling/src/vivarium_profiling/model_specifications/model_spec.yaml

The ``-v`` flag will log verbosely, so you will get log messages every time
step. For more ways to run simulations, see the tutorials at
https://vivarium.readthedocs.io/en/latest/tutorials/running_a_simulation/index.html
and https://vivarium.readthedocs.io/en/latest/tutorials/exploration.html


Profiling and Benchmarking
---------------------------

This repository provides tools for profiling and benchmarking Vivarium simulations
to analyze their performance characteristics.

Configuring Scaling Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This repository includes a custom ``MultiComponentParser`` plugin that allows you to
easily create scaling simulations by defining multiple instances of diseases and risks
using a simplified YAML syntax.

To use the parser, add it to your model specification::

    plugins:
        required:
            component_configuration_parser:
                controller: "vivarium_profiling.plugins.parser.MultiComponentParser"

Then use the ``causes`` and ``risks`` multi-config blocks:

**Causes Configuration**

Define multiple disease instances with automatic numbering::

    components:
        causes:
            lower_respiratory_infections:
                number: 4          # Creates 4 disease instances
                duration: 28       # Disease duration in days
                observers: True    # Auto-create DiseaseObserver components

This creates components named ``lower_respiratory_infections_1``,
``lower_respiratory_infections_2``, etc., each with its own observer if enabled.

**Risks Configuration**

Define multiple risk instances and their effects on causes::

    components:
        risks:
            high_systolic_blood_pressure:
                number: 2
                observers: False    # Set False for continuous risks
                affected_causes:
                    lower_respiratory_infections:
                        effect_type: nonloglinear
                        measure: incidence_rate
                        number: 2   # Affects first 2 LRI instances

            unsafe_water_source:
                number: 2
                observers: True     # Set True for categorical risks
                affected_causes:
                    lower_respiratory_infections:
                        effect_type: loglinear
                        number: 2

**Effect Types:**

- ``loglinear``: Creates standard ``RiskEffect`` components (for PAFs)
- ``nonloglinear``: Creates ``NonLogLinearRiskEffect`` components (for relative risk)

**Complete Example**

See ``model_specifications/model_spec_scaling.yaml`` for a complete working example
of a scaling simulation configuration.


Running Benchmark Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``run_benchmark`` command profiles multiple model specifications and collects
runtime and memory usage statistics::

    (vivarium_profiling) :~$ run_benchmark \
        -m "model_spec_baseline.yaml" \
        -m "model_spec_*.yaml" \
        -r 10 \
        -b 20 \
        -o /path/to/results

**Required Arguments:**

- ``-m, --model_specifications``: Model specification files (supports glob patterns).
  Can be specified multiple times. One model must be ``model_spec_baseline.yaml``.
- ``-r, --model-runs``: Number of runs for non-baseline models.
- ``-b, --baseline-model-runs``: Number of runs for baseline model (typically higher
  for better statistics).

**Optional Arguments:**

- ``-o, --output-dir``: Directory where results will be saved (default: current directory).
  Creates a timestamped subdirectory ``profile_YYYY_MM_DD_HH_MM_SS/``.
- ``--extraction-config``: Path to YAML file defining custom extraction patterns
  (see "Customizing Result Extraction" below).
- ``-v``: Increase logging verbosity (can be repeated: ``-vv``, ``-vvv``).
- ``--pdb``: Drop into debugger if an error occurs.

**Using External Model Specifications**

You can benchmark models from other repositories by providing full paths::

    (vivarium_profiling) :~$ run_benchmark \
        -m "model_spec_baseline.yaml" \
        -m "/path/to/other/repo/model_specs/model_spec_custom.yaml" \
        -r 10 \
        -b 20

**Output Files**

The command creates a timestamped directory containing:

- ``benchmark_results.csv``: Raw profiling data for each run
- ``summary.csv``: Aggregated statistics (automatically generated)
- ``performance_analysis.png``: Performance charts (automatically generated)
- Additional analysis plots for runtime phases and bottlenecks


Analyzing Benchmark Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``summarize`` command processes benchmark results and creates visualizations.
This runs automatically after ``run_benchmark``, but can also be run manually
for custom analysis::

    (vivarium_profiling) :~$ summarize benchmark_results.csv

**Optional Arguments:**

- ``--extraction-config``: Use custom extraction patterns (must match those used
  in ``run_benchmark``).
- ``--nb``: Generate an interactive Jupyter notebook instead of static plots.
- ``-v``: Increase logging verbosity.
- ``--pdb``: Drop into debugger if an error occurs.

**Generated Files**

By default (without ``--nb``):

- ``summary.csv``: Aggregated statistics with mean, median, std, min, max
  for all metrics, plus percent differences from baseline
- ``performance_analysis.png``: Runtime and memory usage comparison charts
- ``runtime_analysis_*.png``: Individual phase runtime charts (setup, run, etc.)
- ``bottleneck_fraction_*.png``: Bottleneck fraction scaling analysis

With ``--nb`` flag:

- ``analysis.ipynb``: Interactive Jupyter notebook with all plots and data
  exploration capabilities


Customizing Result Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the benchmarking tools extract standard profiling metrics:

- Simulation phases: setup, initialize_simulants, run, finalize, report
- Common bottlenecks: gather_results, pipeline calls, population views
- Memory usage and total runtime

You can customize which metrics to extract by creating an extraction config YAML file.
See ``extraction_config_example.yaml`` for a complete annotated example.

**Basic Pattern Structure**::

    patterns:
      - name: my_function          # Logical name for the metric
        filename: my_module.py     # Source file containing the function
        function_name: my_function # Function name to match
        extract_cumtime: true      # Extract cumulative time (default: true)
        extract_percall: false     # Extract time per call (default: false)
        extract_ncalls: false      # Extract number of calls (default: false)

**Pattern Types:**

1. **Bottleneck patterns** - Extract all metrics (cumtime, percall, ncalls)
   for detailed performance analysis of hotspots

2. **Phase patterns** - Extract only cumtime with custom column naming
   (e.g., ``rt_{name}_s``) for high-level simulation phases

3. **Custom patterns** - Mix and match metrics as needed

**Using Custom Extraction Config**

Provide the same config to both commands::

    (vivarium_profiling) :~$ run_benchmark \
        -m "model_spec_*.yaml" \
        -r 10 -b 20 \
        --extraction-config my_extraction_config.yaml

    (vivarium_profiling) :~$ summarize \
        results/profile_*/benchmark_results.csv \
        --extraction-config my_extraction_config.yaml

The extraction config defines which profiling metrics appear in your results
and how bottleneck fractions are calculated.
