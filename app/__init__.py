"""NeuroPredict backend package.

The application itself lives in `app.server`. This module only re-exports it.

`app` is re-exported here on purpose: Render's start command is
`gunicorn app:app`, which resolves this package and looks for an attribute
named `app`. That used to work because this file held a complete 784-line
copy of the Flask application, which also meant it ran on every boot,
doing forty failed model-path lookups and loading the model a second time.
Deleting that copy removed the attribute and broke the deploy with:

    AttributeError: module 'app' has no attribute 'app'

Re-exporting keeps `gunicorn app:app` and `gunicorn backend:app` both
working, without a second copy of anything.
"""

from app.server import app

__all__ = ["app"]
