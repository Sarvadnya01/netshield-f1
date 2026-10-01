"""Canonical label mapping for CICIoT2023 8-class classification."""

from __future__ import annotations

CLASS_NAMES: list[str] = [
    "Benign",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web",
    "BruteForce",
]

NUM_CLASSES: int = len(CLASS_NAMES)

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}

# Explicit mapping from CICIoT2023's 34 fine-grained labels to 8 classes
_EXPLICIT_MAP: dict[str, str] = {
    "BenignTraffic": "Benign",
    # DDoS
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    # DoS
    "DoS-UDP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Mirai
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Recon
    "Recon-PingSweep": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-HostDiscovery": "Recon",
    "VulnerabilityScan": "Recon",
    # Spoofing
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",
    # Web
    "SqlInjection": "Web",
    "XSS": "Web",
    "CommandInjection": "Web",
    "Backdoor_Malware": "Web",
    "Uploading_Attack": "Web",
    "BrowserHijacking": "Web",
    # BruteForce
    "DictionaryBruteForce": "BruteForce",
}

# Prefix-based fallback rules (checked in order)
_PREFIX_RULES: list[tuple[str, str]] = [
    ("DDoS-", "DDoS"),
    ("DoS-", "DoS"),
    ("Mirai-", "Mirai"),
    ("Recon-", "Recon"),
]


def map_label(raw_label: str) -> str:
    """Map a CICIoT2023 fine-grained label to one of the 8 canonical classes.

    Raises ValueError if the label is unknown and doesn't match any prefix rule.
    """
    # Try explicit map first
    mapped = _EXPLICIT_MAP.get(raw_label)
    if mapped is not None:
        return mapped

    # Try prefix rules
    for prefix, cls in _PREFIX_RULES:
        if raw_label.startswith(prefix):
            return cls

    raise ValueError(f"Unknown label: {raw_label!r}")
