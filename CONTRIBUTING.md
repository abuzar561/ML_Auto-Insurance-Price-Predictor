# Contributing

Thanks for your interest in improving this project.

## Local Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Development Checks

Run these checks before opening a pull request:

```bash
node scripts/validate-project.js
python -m py_compile app.py train_model.py src/insurance_pipeline.py
python train_model.py
```

## Guidelines

- Keep changes focused and easy to review.
- Do not commit generated model binaries or local secrets.
- Document any dataset, feature, or model-selection changes.
- Treat this project as an educational demo, not a production pricing system.
