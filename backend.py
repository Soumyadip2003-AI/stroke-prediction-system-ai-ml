"""Gunicorn entrypoint. `Procfile` serves `backend:app`.

`models`, `models_loaded` and `logger` are re-exported so that importing this
module the way gunicorn does is enough to confirm the model actually loaded.
ml/test_gunicorn_import.py checks exactly that.
"""

from app.server import app, logger, models, models_loaded

__all__ = ["app", "logger", "models", "models_loaded"]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
