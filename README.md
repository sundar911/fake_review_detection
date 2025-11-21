# Fake Review Detection

## Project Structure

```
fake_review_detection/
├── README.md            # This guide
├── requirements.txt     # Python dependencies
├── run_best.py          # Entry point to execute the best configuration for each model
├── tuning_logs/         # JSON logs with the explored grids and selected hyper-parameters
└── fake_review_detection/
    ├── __init__.py
    ├── config.py        # Centralised record of best hyper-parameters used by run_best.py
    ├── data.py          # Dataset loading, tokenisation, and batching utilities
    ├── evaluation.py    # Accuracy / precision / recall / F1 helpers
    ├── models/          # HAN, Multi-Head HAN, TextCNN, TextBiLSTM implementations
    ├── simple_training.py
    └── training.py
```

The original dataset (`myle_ott.csv`) should live at the project root (one level above this README).

## Quick Start

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r fake_review_detection/requirements.txt
   ```

2. Run any model with its tuned hyper-parameters:

   | Category | Command |
   |----------|---------|
   | Baseline ML (TF–IDF + Logistic Regression) | `python fake_review_detection/run_best.py --model logistic` |
   | Simple DL – TextCNN | `python fake_review_detection/run_best.py --model textcnn` |
   | Simple DL – TextBiLSTM | `python fake_review_detection/run_best.py --model textbilstm` |
   | Complex DL – HAN | `python fake_review_detection/run_best.py --model han` |
   | Complex DL – Multi-Head HAN | `python fake_review_detection/run_best.py --model multihead-han` |
   | All models sequentially | `python fake_review_detection/run_best.py --model all` |

Each command prints validation and test metrics (accuracy, precision, recall, F1) for the selected configuration.

## Hyper-Parameter Logs

The `tuning_logs/` directory records the grids explored and the metrics observed for every model category:

- `logistic_regression.json`
- `textcnn.json`
- `textbilstm.json`
- `han.json`
- `multihead_han.json`

Each JSON file captures the search space, the metrics for every evaluated configuration, and the rationale for the chosen best settings. These logs provide provenance for the defaults in `fake_review_detection/config.py`.

## Extending or Retuning

- Update `fake_review_detection/config.py` if you identify improved hyper-parameters.
- Use the existing utilities (`data.py`, `simple_training.py`, `training.py`) to script additional experiments or integrate new model architectures.
- Store any new sweeps in `tuning_logs/` so results remain traceable.
