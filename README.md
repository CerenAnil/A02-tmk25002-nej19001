# A02: Ping Pong - California Housing Regression

**OPIM 5512 | University of Connecticut**

## Partners
- Ceren Anil (tmk25002)
- Nehal Jaiswal (nej19001)

## Project Overview
This project trains a neural network regression model (MLPRegressor) on the California Housing dataset from scikit-learn. The goal is to predict median house values based on features like income, house age, and location.

## Workflow
All changes were made through a branch → pull request → review → merge workflow. Neither partner pushed directly to `main`.

| PR | Contributor | Description |
|----|-------------|-------------|
| #1 | Ceren Anil | Load dataset + train/test split |
| #2 | Nehal Jaiswal | Add MLPRegressor with early stopping |
| #3 | Ceren Anil | Add train predictions + plot |
| #4 | Nehal Jaiswal | Add test predictions + plot |
| #5 | Ceren Anil | Improve plots, labels, titles, hyperparameters |

## Project Structure
```
.
├── src/
│   └── ds_pipeline.py    # Main ML pipeline script
├── figures/
│   ├── train_actual_vs_pred.png
│   └── test_actual_vs_pred.png
├── requirements.txt
└── README.md
```

## How to Run

### Setup
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Run the pipeline
```bash
python src/ds_pipeline.py
```

Output plots are saved to the `figures/` directory.
