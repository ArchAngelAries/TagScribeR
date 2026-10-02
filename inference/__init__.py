"""AI inference layer: providers (local VLMs, remote APIs, taggers), model
discovery, a model manager that reuses loaded models, and a Qt batch worker.

The UI talks to this package only through ``ProviderSpec`` / ``ModelManager``
/ ``BatchWorker``; new model families are added as providers without touching
the tabs.
"""
