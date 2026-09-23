# mintic

Repository for my Data Mining course at ENES Morelia, UNAM.

## About

This is the archive for all of my implementations developed during my Data Mining course. 

Each challenge gets its own subpackage under `mintic/`.

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
- Adrian Lara *[Undergraduate student in "Tecnologías para la Información en Ciencias" at Universidad Nacional Autónoma de México (UNAM)]*.
- **Contact:** adrianlarasc@gmail.com
