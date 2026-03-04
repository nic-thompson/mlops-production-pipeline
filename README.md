# ML Monitoring Pipeline

A production-style ML monitoring system with model registry management, drift detection, CI testing, and operational tooling.

This repository demonstrates the core infrastructure used in real-world MLOps systems.

---

# Overview

This project implements the monitoring layer of a machine learning platform.

Key capabilities:

* Model version registry with rollback support
* Statistical data drift detection
* Batch monitoring job
* Structured logging
* JSON drift reports
* CI pipeline with coverage enforcement
* Operational interface via Makefile

The system is designed to mimic how production ML monitoring pipelines operate.

---

# Architecture

```
mlops-production-pipeline
│
├── src/
│   ├── registry.py        # model registry management
│   ├── drift.py           # statistical drift detection
│
├── jobs/
│   └── drift_job.py       # production monitoring job
│
├── tests/
│   ├── test_registry.py
│   └── test_drift.py
│
├── data/                  # example datasets
├── reports/               # generated monitoring reports
│
├── Makefile               # operational commands
├── requirements.txt
├── pytest.ini
└── .github/workflows/ci.yml
```

---

# Installation

Clone the repository and install dependencies.

```
git clone <repo>
cd mlops-production-pipeline
make install
```

---

# Running Tests

```
make test
```

Tests enforce:

* unit test coverage ≥ 85%
* registry integrity
* drift detection correctness

---

# Running Drift Monitoring

```
make drift
```

This runs the monitoring job:

```
python -m jobs.drift_job \
  --reference data/reference.parquet \
  --current data/current.parquet \
  --threshold 0.2 \
  --output reports/drift_report.json
```

The job:

1. Loads reference dataset
2. Loads current dataset
3. Computes feature drift using KS-tests
4. Produces a JSON drift report
5. Returns exit codes for automation

Exit codes:

```
0 → No drift
1 → Drift detected
2 → Job failure
```

---

# Operational Commands

```
make install
make test
make coverage
make drift
make clean
```

---

# Phase 1 Milestone

Phase 1 implements the **monitoring core of an ML platform**.

Features completed:

* Model registry
* Drift detection
* Batch monitoring job
* CI pipeline
* Test coverage enforcement
* Operational Makefile interface

Next phases will expand this into a full ML platform with:

* training pipelines
* model deployment
* automated retraining
