from pathlib import Path


PAGE = Path(__file__).parents[2] / "docs" / "gpu_tierlist.html"


def test_gpu_tierlist_contains_sortable_data_and_required_rows() -> None:
    content = PAGE.read_text(encoding="utf-8")

    assert "const GPU_ROWS =" in content
    assert "function sortRows(key)" in content
    assert "RTX 6000 Ada" in content
    assert "A40" in content
    assert "B300" in content
    assert "$0.77/hr" in content
    assert "$0.44/hr" in content
    assert "Blocked here" in content
    assert "Gate required" in content


def test_gpu_tierlist_preserves_every_user_provided_price() -> None:
    content = PAGE.read_text(encoding="utf-8")

    supplied_prices = {
        "B300": "$7.39/hr",
        "B200": "$5.89/hr",
        "H200 SXM": "$4.39/hr",
        "H200 NVL": "$3.79/hr",
        "H100 NVL": "$3.19/hr",
        "H100 SXM": "$2.99/hr",
        "H100 PCIe": "$2.89/hr",
        "RTX PRO 6000": "$1.99/hr",
        "RTX PRO 6000 WK": "$1.89/hr",
        "A100 SXM": "$1.49/hr",
        "A100 PCIe": "$1.39/hr",
        "RTX 5090": "$0.99/hr",
        "L40S": "$0.99/hr",
        "L40": "$0.82/hr",
        "RTX 6000 Ada": "$0.77/hr",
        "RTX PRO 4500": "$0.74/hr",
        "B300 MIG 34GB": "$0.50/hr",
        "RTX A6000": "$0.49/hr",
        "A40": "$0.44/hr",
    }
    for gpu, price in supplied_prices.items():
        assert gpu in content
        assert price in content
