"""Ways to support TagScribeR, and where the project lives.

Plain links: the app opens them in the system browser and sends nothing. The sidebar's two small buttons, the Help
Center's Credits topic and the README use the same addresses.
"""
from __future__ import annotations

PROJECT_URL = "https://github.com/ArchAngelAries/TagScribeR"
ISSUES_URL = PROJECT_URL + "/issues"

# (id, label, url, tooltip)
SUPPORT_LINKS: tuple[tuple[str, str, str, str], ...] = (
    ("patreon", "Patreon", "https://www.patreon.com/c/ArchAngelAries",
     "Support TagScribeR's development on Patreon (opens your browser; nothing is sent from the app)"),
    ("kofi", "Ko-fi", "https://ko-fi.com/archangelaries",
     "Buy the developer a coffee on Ko-fi (opens your browser; nothing is sent from the app)"),
)
