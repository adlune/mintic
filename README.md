# mintic

Repository for my Data Mining course at ENES Morelia, UNAM.

## About

This is where all my implementations for the course challenges are. Each challenge gets its own subpackage under `mintic/`.

## Established rules of development

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
├── data/                   # directory for datasets
└── mintic/                 # base directory for course challenges
    ├── eda/                # 1: exploratory data analysis
    ├── ensemble/           # 2: ensembles
    ├── kmeans/             # 3: k-means
    ├── dbscan/             # 4: dbscan
    ├── apriori/            # 5: apriori algorithm
    └── pca/                # 6: principal component analysis
```

## Setup

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
## Author, affiliation and contact
- Adrian Lara A. *[Undergraduate student in "Tecnologías para la Información en Ciencias" at Universidad Nacional Autónoma de México (UNAM)]*.
- **Contact:** adrianlara.jpg@gmail.com
