# realestate-trafficjam-area-classifier
Goal: classify areas as **traffic jam heavy** vs **not** using public traffic speed data (NYC DOT speeds).
Idea: investors care because traffic pain can influence desirability / demand / rent etc.

## steps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1) download
python -m src.data.pull_nyc_speed --limit 50000
# 2) aggregate
python -m src.data.make_table
# 3) train
python -m src.models.train_models
# 4) eval
python -m src.models.eval_models
```