"""PHASE_13_82_DF S7/M14 typed categorical identity acceptance oracles.

Thirty cases are RED on committed P1 source cd86ab4e, GREEN with S7 repair.
The contract requires homogeneous normalized JSON scalar types per axis;
mixed bool/int/float domains fail before deduplication or payload acceptance.
Scientific contract and production changes require independent review.
"""

import copy

import numpy as np
import pytest

from dfextensions.dfdraw import statistical_state as ss


EDGES = [0.0, 1.0]


def _build(kind, values, *, partition_id='m14-a', weights=None):
    kwargs = dict(
        edges=EDGES, name='qa/m14', partition_id=partition_id,
        categories={'flag': list(values)}, weights=weights,
    )
    x = [0.25] * len(values)
    if kind == 'histogram':
        return ss.build_histogram_state(x, **kwargs)
    return ss.build_profile_state(x, [float(i + 2) for i in range(len(values))], **kwargs)


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
@pytest.mark.parametrize('values', [
    (0, False), (False, 0), (1, True), (True, 1),
    (1, 1.0), (1.0, 1), (0.0, False), (False, 0.0),
], ids=[
    'int0-boolFalse', 'boolFalse-int0', 'int1-boolTrue', 'boolTrue-int1',
    'int1-float1', 'float1-int1', 'float0-boolFalse', 'boolFalse-float0',
])
def test_m14_red_01_builder_rejects_mixed_scalar_type_before_dedup(kind, values):
    # A single physical partition must not silently combine distinct JSON
    # scalar types into one statistical cell, regardless of source-row order.
    with pytest.raises(ValueError, match='mixed category types'):
        _build(kind, values)


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
@pytest.mark.parametrize('values', [(0, False), (False, 0), (1, 1.0), (1.0, 1)],
                         ids=['int-bool', 'bool-int', 'int-float', 'float-int'])
def test_m14_red_02_weighted_builder_refuses_mixed_types(kind, values):
    with pytest.raises(ValueError, match='mixed category types'):
        _build(kind, values, weights=[1.25, 2.75])


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
@pytest.mark.parametrize('values', [(0, False), (1, True), (1, 1.0)],
                         ids=['int-bool', 'int-true', 'int-float'])
def test_m14_red_03_restored_axis_rejects_mixed_types_even_if_unoccupied(kind, values):
    # The type-invalid extra category need not occur in state records.
    payload = copy.deepcopy(_build(kind, [values[0]]).to_payload())
    payload['axes']['flag']['values'] = list(values)
    with pytest.raises(ValueError, match='mixed category types'):
        ss.StatisticalNode.from_payload(payload)


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
def test_m14_green_01_homogeneous_integer_domain_roundtrip(kind):
    node = _build(kind, [0, np.int64(1), 1])
    restored = ss.StatisticalNode.from_payload(node.to_payload())
    assert restored.axes['flag']['values'] == [0, 1]
    assert int(restored.state['entries'].sum()) == 3


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
def test_m14_green_02_homogeneous_boolean_domain_roundtrip(kind):
    node = _build(kind, [False, np.bool_(True), True])
    restored = ss.StatisticalNode.from_payload(node.to_payload())
    assert restored.axes['flag']['values'] == [False, True]
    assert int(restored.state['entries'].sum()) == 3


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
def test_m14_green_03_homogeneous_float_domain_roundtrip(kind):
    node = _build(kind, [0.0, np.float64(1.0), 1.0])
    restored = ss.StatisticalNode.from_payload(node.to_payload())
    assert restored.axes['flag']['values'] == [0.0, 1.0]
    assert int(restored.state['entries'].sum()) == 3


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
@pytest.mark.parametrize('family,labels', [
    ('boolean', [False, True, False, True]),
    ('integer', [0, 1, 0, 1]),
    ('float', [0.0, 1.0, 0.0, 1.0]),
])
def test_m14_green_04_same_type_exact_merge_equals_direct_pooled_truth(kind, family, labels):
    """An accepted homogeneous domain retains exact weighted raw-data moments."""
    y = [2.0, 3.0, 4.0, 5.0]
    weights = [1.25, 0.5, 0.75, 2.0]

    def build(values, yvalues, wvalues, partition_id):
        kwargs = dict(edges=EDGES, name='qa/m14', partition_id=partition_id,
                      weights=wvalues, categories={'flag': values})
        if kind == 'histogram':
            return ss.build_histogram_state([0.25] * len(values), **kwargs)
        return ss.build_profile_state([0.25] * len(values), yvalues, **kwargs)

    left = build(labels[:2], y[:2], weights[:2], 'A')
    right = build(labels[2:], y[2:], weights[2:], 'B')
    merged = ss.merge_exact([left, right], partition_id='Pooled')
    truth = build(labels, y, weights, 'Raw')

    assert merged.axes['flag'] == truth.axes['flag']
    assert {type(value) for value in merged.axes['flag']['values']} == {type(labels[0])}
    assert merged.to_payload()['state'] == truth.to_payload()['state']
    assert merged.provenance['contributing_partition_ids'] == ['A', 'B']
