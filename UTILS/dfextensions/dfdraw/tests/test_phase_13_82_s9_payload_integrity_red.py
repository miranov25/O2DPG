"""PHASE_13_82_DF S9 slice 1 - check stored retained-state data on load.

Plain-language background
-------------------------
S9 slice 0 made ``merge_exact`` refuse two leaves whose analyst-declared
category orders disagree.  Its review found two ways in which a stored
(payload) file is still trusted too easily when it is loaded:

* P-A - a leaf's order declaration is not checked.  The builders only ever
  write one of two forms on a category axis: no order keys at all
  (undeclared order), or ``ordered: True`` together with
  ``order_origin: 'explicit'`` (analyst-declared order).  A stored leaf with
  ``ordered: 1``, ``'yes'``, ``False`` or a missing/foreign ``order_origin``
  loads anyway; with ``ordered: 1`` it then slips past the slice-0 guard.
* P-B - ``PartitionedStatisticalState.from_payload`` compares the stored
  top-level axes with the rebuilt ones using Python equality, which treats
  ``False == 0``, ``True == 1`` and ``0.0 == 0``.  A stored file that changes
  the *type* of category labels, edges or order flags is accepted silently.

What this module asserts
------------------------
RED on the reference (commit 4f5ec79d bytes): S9P_01, S9P_02, S9P_03, S9P_05,
S9P_06 - each must be refused with ``ValueError`` and currently is not.
Controls that pass before and after the fix: S9P_04 (valid leaves still
round-trip) and S9P_07 (an untampered preserved state still round-trips, and a
stored ``-0.0`` label is still read as ``0.0``, the S8 normalisation rule).

Entry points: ``StatisticalNode.from_payload``, direct ``StatisticalNode``
construction, ``merge_exact`` and ``PartitionedStatisticalState.from_payload``
- the functions an analyst uses to reload and combine stored states.

Seat: Fable5_1:draw-coder, 2026-11-07.  Architect scope: option (a) - P-A and
P-B together, with the S9_03 tolerance tightening.
"""

import copy
import dataclasses

import pytest

from dfextensions.dfdraw import statistical_state as ss
from dfextensions.dfdraw import partitioned_state as ps

DECLARED = ['IROC', 'OROC1']
EDGES = [0., 1., 2.]
LEAF_REFUSAL = 'invalid categorical order declaration'
OUTER_REFUSAL = 'S8 payload axes inconsistent with source partitions'


def leaf(pid, x, regions, *, order=None):
    """One retained histogram leaf; ``order`` is the analyst-declared order."""
    kwargs = dict(name='s9/h', partition_id=pid, edges=EDGES,
                  categories={'region': regions}, metadata={'unit': 'arb'})
    if order is not None:
        kwargs['category_orders'] = {'region': list(order)}
    return ss.build_histogram_state(list(x), **kwargs)


def with_order_keys(payload, **keys):
    """Return a copy of a leaf payload with the region-axis order keys replaced.

    A key given as ``None`` is removed; any other value is written as is.
    """
    out = copy.deepcopy(payload)
    axis = out['axes']['region']
    axis.pop('ordered', None)
    axis.pop('order_origin', None)
    for key, value in keys.items():
        if value is not None:
            axis[key] = value
    return out


# Every order declaration the builders never write.
BAD_DECLARATIONS = [
    pytest.param(dict(ordered=1, order_origin='explicit'), id='ordered-int-1'),
    pytest.param(dict(ordered='yes', order_origin='explicit'), id='ordered-string'),
    pytest.param(dict(ordered=True), id='ordered-without-origin'),
    pytest.param(dict(ordered=True, order_origin='canonical'), id='ordered-canonical-origin'),
    pytest.param(dict(order_origin='explicit'), id='origin-without-ordered'),
    pytest.param(dict(ordered=False), id='ordered-false'),
    pytest.param(dict(ordered=False, order_origin='canonical'), id='ordered-false-canonical'),
]


@pytest.mark.parametrize('keys', BAD_DECLARATIONS)
def test_S9P_01_leaf_payload_with_invalid_order_declaration_is_refused(keys):
    """P-A: loading a stored leaf refuses any order declaration the builders never write."""
    payload = leaf('A', [0.2], ['IROC'], order=DECLARED).to_payload()
    with pytest.raises(ValueError, match=LEAF_REFUSAL):
        ss.StatisticalNode.from_payload(with_order_keys(payload, **keys))


def test_S9P_02_direct_construction_with_invalid_order_declaration_is_refused():
    """P-A: the same rule holds for a node constructed directly, not only for payloads."""
    node = leaf('A', [0.2], ['IROC'], order=DECLARED)
    axes = copy.deepcopy(node.axes)
    axes['region']['ordered'] = 1
    with pytest.raises(ValueError, match=LEAF_REFUSAL):
        dataclasses.replace(node, axes=axes)


def test_S9P_03_stored_ordered_1_no_longer_bypasses_merge_exact_guard():
    """P-A end to end: the bypass found in the slice-0 review is closed at load time.

    Two leaves declare opposite orders.  Stored with ``ordered: 1`` they used to
    load and merge with the first order winning; now the load itself is refused.
    """
    a = leaf('A', [0.2], ['IROC'], order=DECLARED).to_payload()
    b = leaf('B', [0.2], ['OROC1'], order=list(reversed(DECLARED))).to_payload()
    a1 = with_order_keys(a, ordered=1, order_origin='explicit')
    b1 = with_order_keys(b, ordered=1, order_origin='explicit')
    with pytest.raises(ValueError, match=LEAF_REFUSAL):
        ss.merge_exact([ss.StatisticalNode.from_payload(a1),
                        ss.StatisticalNode.from_payload(b1)], partition_id='AB')


@pytest.mark.parametrize('order', [None, DECLARED], ids=['undeclared', 'declared'])
def test_S9P_04_valid_leaf_payloads_still_round_trip(order):
    """Control: both forms the builders write still load and re-emit unchanged."""
    node = leaf('A', [0.2, 1.3], ['OROC1', 'IROC'], order=order)
    payload = node.to_payload()
    again = ss.StatisticalNode.from_payload(payload)
    assert again.to_payload() == payload
    if order is None:
        assert 'ordered' not in again.axes['region']
        assert 'order_origin' not in again.axes['region']
    else:
        assert again.axes['region']['ordered'] is True
        assert again.axes['region']['order_origin'] == 'explicit'
        assert again.axes['region']['values'] == DECLARED


def _preserved_numeric():
    return ps.preserve_partitions([leaf('A', [0.2], [0]), leaf('B', [1.2], [1])])


def _preserved_declared():
    return ps.preserve_partitions([leaf('A', [0.2], ['IROC'], order=DECLARED),
                                   leaf('B', [1.2], ['OROC1'], order=DECLARED)])


@pytest.mark.parametrize('values', [[False, True], [0.0, 1.0]], ids=['bool', 'float'])
def test_S9P_05_outer_axis_label_type_change_is_refused(values):
    """P-B: stored outer category labels of another type than the leaves are refused."""
    payload = _preserved_numeric().to_payload()
    assert payload['axes']['region']['values'] == [0, 1]
    payload['axes']['region']['values'] = values
    with pytest.raises(ValueError, match=OUTER_REFUSAL):
        ps.PartitionedStatisticalState.from_payload(payload)


@pytest.mark.parametrize('dim, key, value', [
    pytest.param('region', 'ordered', 1, id='declared-axis-ordered-1'),
    pytest.param('partition_id', 'ordered', 0, id='partition-axis-ordered-0'),
    pytest.param('x_bin', 'edges', [0, 1, 2], id='x-edges-as-integers'),
])
def test_S9P_06_outer_axis_flag_or_edge_type_change_is_refused(dim, key, value):
    """P-B: the type-sensitive check covers the whole stored axis, not only the labels."""
    payload = _preserved_declared().to_payload()
    payload['axes'][dim][key] = value
    with pytest.raises(ValueError, match=OUTER_REFUSAL):
        ps.PartitionedStatisticalState.from_payload(payload)


def test_S9P_07_untampered_preserved_state_round_trips_and_signed_zero_is_normalised():
    """Control: valid stored states still load; a stored -0.0 label reads as 0.0 (S8 rule)."""
    for state in (_preserved_numeric(), _preserved_declared()):
        payload = state.to_payload()
        again = ps.PartitionedStatisticalState.from_payload(payload)
        assert again.to_payload() == payload

    zero = ps.preserve_partitions([leaf('A', [0.2], [0.0]), leaf('B', [0.2], [1.0])])
    payload = zero.to_payload()
    payload['axes']['region']['values'] = [-0.0, 1.0]
    again = ps.PartitionedStatisticalState.from_payload(payload)
    assert again.axes['region']['values'] == [0.0, 1.0]
