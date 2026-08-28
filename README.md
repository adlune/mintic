# mintic

**Pineda Lab**,
**ENES Morelia, UNAM**

## About

`mintic` is the core repository for the Data Mining course. It's purpose is to familiarize students with algorithms and how they work.

## Author
- Adrian Lara, 3rd year Data Science undergraduate.
- **Contact:** adrianlara.jpg@gmail.com

## Rules of development

- Algorithm logic was manually implemented by the student under `mintic/` subpackages.
- The only allowed library for numeric calculations is **NumPy**. The usage of `scikit-learn`, `scipy` or other libraries that already implement the assigned challenge are not allowed.
- `pandas` is allowed only for loading and initial manipulation of data.
- `matplotlib` is allowed for data visualization.

## Repository structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── data/
│   └── sample_dataset.csv
└── mintic/
    ├── __init__.py
    ├── eda/
    │   └── __init__.py
    ├── ensemble/
    │   └── __init__.py
    ├── kmeans/
    │   └── __init__.py
    ├── dbscan/
    │   └── __init__.py
    ├── apriori/
    │   └── __init__.py
    └── pca/
        └── __init__.py
```

Every subdirectory inside `mintic/` is a Python subpackage corresponding to a challenge in the course. The student must implement the solution in each one.

## Installation

**Windows**
```bash
python -m venv .venv
source .venv/Scripts/Activate
pip install -r requirements.txt
```

**Linux/macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execution

```bash
python main.py
```

`main.py` loads the sample dataset in `data/sample_dataset.csv` and acts as a starting point to test the subpackage that's being worked on.
