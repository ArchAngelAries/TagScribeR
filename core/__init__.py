# core/__init__.py
#
# Intentionally empty: importing `core.<module>` must stay cheap. Heavy
# dependencies (torch, transformers, onnxruntime) are imported lazily by the
# modules that need them so the UI can start without loading them.
