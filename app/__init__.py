"""NeuroPredict backend package.

This file used to hold a full 784-line copy of the Flask app, including its own
`load_models()` and its own `/api/predict` with hardcoded accuracy figures.
Because `backend.py` does `from app.server import app`, that copy executed on
every boot: forty failed model-path lookups and a second load of the model,
roughly a second of startup time, for an app object nothing ever served.

The live application is `app.server`. Keep this file empty.
"""
