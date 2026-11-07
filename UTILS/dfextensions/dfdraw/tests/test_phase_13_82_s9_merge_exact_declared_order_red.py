"""PHASE_13_82_DF S9 slice 0 — ``merge_exact`` must refuse conflicting declared category orders.

Plain-language background
-------------------------
A retained statistical state ("leaf") can carry an analyst-declared order for a
category axis, given through ``category_orders=`` or an ordered pandas
categorical.  The axis then records ``ordered: True`` and
``order_origin: 'explicit'``.  Step S8 made ``preserve_partitions`` refuse two
leaves whose declared orders disagree.  The exact-pooling function
``merge_exact`` did not get the same guard (review condition S8 C-1): it unions
the labels in input order and still labels the result as analyst-declared.  For
two partial declared orders the result even carries an order that nobody
declared.  Downstream collapse and rendering would trust that label.

What this module asserts
------------------------
* S9_01, S9_02, S9_05 — RED on the reference (commit cb22dd58 bytes):
  conflicting declared orders must be refused by ``merge_exact`` with a
  ``ValueError`` naming the axis, exactly as ``preserve_partitions`` already does.
* S9_03, S9_04 — controls that pass before and after the fix: an identical
  declared order is kept (with the pooled sums untouched), and leaves without a
  declared order keep the legacy input-order union.

Entry point: every test calls the public ``merge_exact`` directly, the function
an analyst calls when pooling partitions.

Seat: Fable5_1:draw-coder, 2026-11-07.
"""

import numpy as np
import pytest

from dfextensions.dfdraw import statistical_state as ss
from dfextensions.dfdraw import partitioned_state as ps

DECLARED = ['IROC', 'OROC1', 'OROC2', 'OROC3']
EDGES = (0., 1., 2.)
REFUSAL = 'conflicting explicit categorical order for exact merge'


def leaf(pid, x, regions, *, weights=None, order=None):
    """One retained histogram leaf with a region category per entry.

    ``order`` is the analyst-declared category order (None = undeclared).
    """
    kwargs = dict(name='s9/h', partition_id=pid, edges=EDGES, x_source='pt',
                  weights_source='w', weights=weights,
                  categories={'region': regions}, metadata={'unit': 'arb'})
    if order is not None:
        kwargs['category_orders'] = {'region': list(order)}
    return ss.build_histogram_state(list(x), **kwargs)


def test_S9_01_merge_exact_refuses_reversed_declared_orders():
    """Two leaves declare the same labels in opposite order: refuse, name the axis."""
    a = leaf('A', [0.2], ['IROC'], order=DECLARED)
    b = leaf('B', [0.2], ['OROC1'], order=list(reversed(DECLARED)))
    with pytest.raises(ValueError, match=REFUSAL + ': region'):
        ss.merge_exact([a, b], partition_id='AB')
    # Input order must not matter for the refusal.
    with pytest.raises(ValueError, match=REFUSAL + ': region'):
        ss.merge_exact([b, a], partition_id='BA')


def test_S9_02_merge_exact_refuses_partial_declared_orders():
    """Two partial declared orders have no unique union; refuse instead of inventing one."""
    a = leaf('A', [0.2], ['IROC'], order=['IROC', 'OROC2'])
    b = leaf('B', [0.2], ['OROC1'], order=['OROC1', 'OROC3'])
    with pytest.raises(ValueError, match=REFUSAL + ': region'):
        ss.merge_exact([a, b], partition_id='AB')


def test_S9_03_merge_exact_keeps_identical_declared_order_and_sums():
    """Control: the same declared order on both leaves is kept, and the pooled sums are exact."""
    xa, wa = [0.2, 1.2, 1.3], [1.0, 2.0, 0.5]
    xb, wb = [0.4, 1.1], [3.0, -1.0]
    a = leaf('A', xa, ['OROC3', 'IROC', 'OROC3'], weights=wa, order=DECLARED)
    b = leaf('B', xb, ['OROC1', 'IROC'], weights=wb, order=DECLARED)
    # Inputs deliberately given in the opposite order to the declared one.
    m = ss.merge_exact([b, a], partition_id='AB')

    axis = m.axes['region']
    assert axis['values'] == DECLARED
    assert axis['ordered'] is True and axis['order_origin'] == 'explicit'
    # Empty declared categories stay in the axis; rows are only the occupied cells.
    assert 'OROC2' in axis['values']
    assert 'OROC2' not in set(m.state['region'])

    # Raw-array truth: Σw per (x_bin, region) over both leaves.
    x = np.array(xa + xb)
    w = np.array(wa + wb)
    r = np.array(['OROC3', 'IROC', 'OROC3', 'OROC1', 'IROC'])
    xbin = np.digitize(x, EDGES) - 1
    for row in m.state.itertuples(index=False):
        sel = (xbin == row.x_bin) & (r == row.region)
        assert row.entries == int(sel.sum())
        assert row.sum_weights == pytest.approx(w[sel].sum(), abs=0.0)
        assert row.sum_weights_squared == pytest.approx((w[sel] ** 2).sum(), abs=0.0)
    # Row order follows the declared order inside each physical bin.
    rank = {v: i for i, v in enumerate(DECLARED)}
    keys = [(int(t.x_bin), rank[t.region]) for t in m.state.itertuples(index=False)]
    assert keys == sorted(keys)


def test_S9_04_merge_exact_undeclared_order_is_unchanged_legacy_union():
    """Control: without a declared order the legacy input-order union is unchanged."""
    a = leaf('A', [0.2], ['Z'])
    b = leaf('B', [0.2], ['A'])
    assert ss.merge_exact([a, b], partition_id='AB').axes['region']['values'] == ['Z', 'A']
    assert ss.merge_exact([b, a], partition_id='BA').axes['region']['values'] == ['A', 'Z']
    assert ss.merge_exact([a, b], partition_id='AB2').axes['region'].get('ordered') is not True


def test_S9_05_merge_exact_and_preserve_partitions_agree_on_refusal():
    """The pooled and the preserved paths refuse the same conflicting input."""
    a = leaf('A', [0.2], ['OROC2'], order=['IROC', 'OROC2', 'OROC3'])
    b = leaf('B', [0.2], ['IROC'], order=['OROC2', 'IROC', 'OROC3'])
    with pytest.raises(ValueError, match='conflicting explicit scientific orders'):
        ps.preserve_partitions([a, b])
    with pytest.raises(ValueError, match=REFUSAL + ': region'):
        ss.merge_exact([a, b], partition_id='AB')
