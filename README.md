# Model Observability Service

Operational monitoring service for deployed machine learning models, providing registry validation, feature drift detection, prediction telemetry analysis, and metrics export for production inference systems.

This component runs as part of the post-deployment control surface of an ML platform and is responsible for ensuring model behaviour remains consistent with training-time expectations under live traffic conditions.

---

# Overview

Machine learning systems require continuous verification after deployment to detect distribution shift, schema divergence, and unintended behavioural changes.

This service implements the runtime monitoring layer responsible for:

- registry-aware model validation
- statistical feature drift detection
- batch monitoring execution workflows
- structured monitoring telemetry
- machine-readable drift reporting
- CI-enforced monitoring correctness
- operational task automation via Make targets

The repository represents the observability boundary between inference workloads and downstream alerting, dashboards, and retraining triggers.

---

# Architecture

```
model-observability-service
│
├── src/
│   ├── registry.py        # model metadata validation and version alignment
│   ├── drift.py           # statistical drift detection engine
│
├── jobs/
│   └── drift_job.py       # scheduled monitoring execution entrypoint
│
├── tests/
│   ├── test_registry.py
│   └── test_drift.py
│
├── data/                  # reference datasets for monitoring baselines
├── reports/               # generated monitoring artefacts
│
├── Makefile               # operational command surface
├── requirements.txt
├── pytest.ini
└── .github/workflows/ci.yml
```

The monitoring job executes independently of inference latency constraints and produces signals suitable for integration with alerting and retraining orchestration systems.

---

# Installation

Clone the repository and install runtime dependencies:

```
git clone <repo>
cd model-observability-service
make install
```

---

# Test Execution

```
make test
```

The test suite validates:

- registry consistency guarantees
- drift detection correctness
- monitoring job behaviour
- regression protection across monitoring logic

Coverage thresholds are enforced via CI to maintain monitoring reliability as the service evolves.

---

# Drift Monitoring Execution

```
make drift
```

This runs the monitoring workflow:

```
python -m jobs.drift_job \
  --reference data/reference.parquet \
  --current data/current.parquet \
  --threshold 0.2 \
  --output reports/drift_report.json
```

The monitoring job:

1. loads baseline reference distributions
2. evaluates incoming feature distributions
3. computes statistical drift metrics (KS-test)
4. generates structured monitoring reports
5. returns automation-compatible exit signals

Exit codes:

```
0 → distributions within tolerance
1 → drift threshold exceeded
2 → monitoring execution failure
```

These signals are designed for integration with CI gates, schedulers, or retraining pipelines.

---

# Operational Interface

Common operational tasks:

```
make install
make test
make coverage
make drift
make clean
```

The Makefile provides a reproducible command surface for local execution and CI environments.

---

# Platform Role

This repository implements the post-deployment monitoring layer of a production ML platform.

It is intended to operate downstream of inference services and upstream of:

- alerting infrastructure
- observability dashboards
- dataset reconstruction pipelines
- automated retraining triggers

By externalising monitoring from training workflows, the system enables continuous verification of model behaviour under live conditions.