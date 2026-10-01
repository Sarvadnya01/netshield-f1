"""Tests for netshield.common.labels."""

import pytest

from netshield.common.labels import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    map_label,
)


def test_class_names_count():
    assert NUM_CLASSES == 8
    assert len(CLASS_NAMES) == 8


def test_class_names_order():
    expected = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
    assert CLASS_NAMES == expected


def test_class_to_idx_roundtrip():
    for name in CLASS_NAMES:
        idx = CLASS_TO_IDX[name]
        assert IDX_TO_CLASS[idx] == name


def test_map_label_benign():
    assert map_label("BenignTraffic") == "Benign"


def test_map_label_ddos_variants():
    assert map_label("DDoS-SYN_Flood") == "DDoS"
    assert map_label("DDoS-SlowLoris") == "DDoS"
    assert map_label("DDoS-HTTP_Flood") == "DDoS"


def test_map_label_dos():
    assert map_label("DoS-UDP_Flood") == "DoS"
    assert map_label("DoS-TCP_Flood") == "DoS"


def test_map_label_mirai():
    assert map_label("Mirai-greeth_flood") == "Mirai"
    assert map_label("Mirai-udpplain") == "Mirai"


def test_map_label_recon():
    assert map_label("Recon-PingSweep") == "Recon"
    assert map_label("Recon-OSScan") == "Recon"
    assert map_label("VulnerabilityScan") == "Recon"


def test_map_label_spoofing():
    assert map_label("MITM-ArpSpoofing") == "Spoofing"
    assert map_label("DNS_Spoofing") == "Spoofing"


def test_map_label_web():
    assert map_label("SqlInjection") == "Web"
    assert map_label("XSS") == "Web"
    assert map_label("CommandInjection") == "Web"
    assert map_label("Backdoor_Malware") == "Web"
    assert map_label("Uploading_Attack") == "Web"
    assert map_label("BrowserHijacking") == "Web"


def test_map_label_bruteforce():
    assert map_label("DictionaryBruteForce") == "BruteForce"


def test_map_label_prefix_fallback():
    # Unknown DDoS variant should still map via prefix
    assert map_label("DDoS-NewVariant") == "DDoS"


def test_map_label_unknown_raises():
    with pytest.raises(ValueError, match="Unknown label"):
        map_label("CompletelyUnknownAttack")
