# Purpose
Included here are files used for keeping track of the performance of different models over time, on a fixed set of datasets
Files are: 
- 'run_benchmarks.py': script that runs benchmark for a given model
    - git pull the given model repository
    - look through configs model and add them to the database
    - For each problem-spec and model-config slug combo:
        - run the benchmark
        - log the results to the database
- 'run-benchmarks.yml': github actions workflow that logs into the benchmark server and runs the benchmarks

The benchmark script assumes that chap_core server is running on localhost on the server and that the following folder structure exists:

```
models/
    ├── <model-slug>/
    │   ├── configs/
    │   │  <config-slug>_<n>.yaml
problem_config_mapping.yaml
```