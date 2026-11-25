"""Strict support checks for the prepared partition experiment; no inference."""
PARTS = {
    "full": (tuple(range(22)), 1024),
    "layers19_20": ((19, 20), 132),
    "complement": (tuple(l for l in range(22) if l not in (19,20)), 892),
}
def expected_support(case):
    assert case["treatment"] in ("mlp","random")
    assert case["partition"] in PARTS
    assert case["alpha"] in (-2,0,2) and type(case["alpha"]) in (int,float)
    return PARTS[case["partition"]]

def validate_support(report, case, contract):
    """contract must be built from the hash-verified frozen offset artifact."""
    layers,count=expected_support(case)
    raw=contract[case["treatment"]+"__"+case["partition"]]
    expected={int(l):v for l,v in raw.items()}
    assert len(expected)==len(raw)
    assert tuple(sorted(expected))==layers
    assert sum(v["selected_neurons"] for v in expected.values())==count
    assert report["passed"] is True
    assert report["neurons"]==count and report["alpha"]==case["alpha"]
    assert tuple(report["layers"])==layers
    assert report["positions_start"]==3 and report["prompt_calls"]==1
    assert report["decode_hook_calls"]==0 and report["hooks_removed_at_end_of_prefill"] is True
    records={int(l):v for l,v in report["records"].items()}
    assert len(records)==len(report["records"]) and tuple(sorted(records))==layers
    for layer,v in records.items():
        e=expected[layer]
        assert v["layer"]==layer and v["selected_neurons"]==e["selected_neurons"]
        assert v["alpha"]==case["alpha"]
        assert v["coefficient_fingerprints"]==e["coefficient_fingerprints"]
        assert v["input_unmodified"] is True and v["unselected_coordinates_exact"] is True
        assert v["earlier_positions_exact"] is True
        assert v["positions"]==list(range(3,case["input_length"]))
        if case["alpha"]==0:
            assert v["zero_is_object_identity"] is True
            assert v["changed_elements"]==v["realized_l2"]==v["rounding_l2"]==v["requested_l2"]==0
            assert v["before"]==v["after"]
    return True
