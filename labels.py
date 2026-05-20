# Labels must match training class_indices exactly (flow_from_directory alphabetical order):
# Order: Columnaris, Healthy, MAS, Saprolegniasis, Streptococcosis

LABELS: list[str] = [
    "Columnaris",       # index 0
    "Healthy",          # index 1
    "MAS",              # index 2
    "Saprolegniasis",   # index 3
    "Streptococcosis",  # index 4
]

LABEL_FULL_NAMES: dict[str, str] = {
    "Columnaris":      "Columnaris",
    "Healthy":         "Healthy",
    "MAS":             "MAS",
    "Saprolegniasis":  "Saprolegniasis",
    "Streptococcosis": "Streptococcosis",
}
