"""PHASE_13_82_DF S9 slice 1 (extension) - stored record label types; caller objects untouched.

Plain-language background
-------------------------
Two small gaps left over from the S8 review, added to slice 1 by the
Architect (2026-11-07, "add A and B"):

* A - a stored data row whose category label has the wrong number type is
  silently converted.  With a category axis of decimals ``[0.5, 1.0]``, a
  stored row labelled ``1`` (an integer) loads and quietly becomes ``1.0``.
  Cause: pandas merges a column holding both ``0.5`` and ``1`` into decimals
  before the existing label check runs.  Every other wrongly typed label
  (bool, string, all-integer column) is already refused; this is the one
  remaining case.  Rule: a stored row label must match a declared axis label
  in value *and* JSON type, after the builders' normalisation (``-0.0`` reads
  as ``0.0``).
* B - building a ``StatisticalNode`` directly rewrites the caller's own
  objects.  If the caller passes a ``-0.0`` label, their axes dictionary and
  their state table are changed to ``0.0`` in place.  Rule: the node
  normalises its own copies; the caller's objects stay as they were.

What this module asserts
------------------------
RED on the current slice-1 candidate (``statistical_state.py`` MD5
``e7a74b83...``): S9R_01 (three cases) and S9R_03 (two cases).
Controls that pass before and after: S9R_02 (labels that are already refused
stay refused; a stored ``-0.0`` row still reads as ``0.0``) and S9R_04 (the
node's own values are still normalised; ordinary builds are unchanged).

Entry points: ``StatisticalNode.from_payload`` and direct ``StatisticalNode``
construction (here through ``dataclasses.replace``, which calls the
constructor).

Seat: Fable5_1:draw-coder, 2026-11-07.
"""

import copy
import dataclasses

import pytest

from dfextensions.dfdraw import statistical_state as ss

REFUSAL = 'undeclared category value'


def stored(labels, axis_values):
    """Payload of one histogram leaf whose region axis is ``axis_values``,
    with the stored row labels then replaced by ``labels`` (one row each)."""
    node = ss.build_histogram_state([0.2] * len(axis_values), edges=[0., 1.], name='s9/h',
                                    partition_id='A', categories={'region': list(axis_values)})
    payload = node.to_payload()
    assert len(payload['state']) == len(labels)
    for record, label in zip(payload['state'], labels):
        record['region'] = label
    return payload


@pytest.mark.parametrize('axis_values, labels', [
    pytest.param([0.5, 1.0], [0.5, 1], id='int-after-float'),
    pytest.param([1.0, 2.5], [1, 2.5], id='int-before-float'),
    pytest.param([0.5, 1.0, 2.0], [0.5, 1, 2.0], id='int-among-floats'),
])
def test_S9R_01_integer_row_label_under_decimal_axis_is_refused(axis_values, labels):
    """A: an integer row label among decimals is refused, not converted to a decimal."""
    with pytest.raises(ValueError, match=REFUSAL):
        ss.StatisticalNode.from_payload(stored(labels, axis_values))


@pytest.mark.parametrize('axis_values, labels', [
    pytest.param([1.0, 2.0], [1, 2], id='all-int-under-decimal'),
    pytest.param([0, 1], [0, 1.0], id='decimal-under-int'),
    pytest.param([0.5, 1.0], [0.5, True], id='bool-under-decimal'),
    pytest.param([0, 1], [False, True], id='bool-under-int'),
    pytest.param(['0', '1'], ['0', 1], id='int-under-string'),
])
def test_S9R_02_labels_already_refused_stay_refused(axis_values, labels):
    """Control: wrongly typed labels that are refused today stay refused."""
    with pytest.raises(ValueError, match=REFUSAL):
        ss.StatisticalNode.from_payload(stored(labels, axis_values))


def test_S9R_02b_stored_negative_zero_row_label_still_reads_as_zero():
    """Control: the S8 normalisation rule is kept - a stored -0.0 row label is 0.0."""
    node = ss.StatisticalNode.from_payload(stored([-0.0, 1.0], [0.0, 1.0]))
    assert list(node.state['region']) == [0.0, 1.0]
    assert node.axes['region']['values'] == [0.0, 1.0]


def _caller_objects():
    """A valid leaf plus caller-owned axes/state that carry a -0.0 label."""
    base = ss.build_histogram_state([0.2, 0.3], edges=[0., 1.], name='s9/h',
                                    partition_id='A', categories={'region': [0.0, 1.0]})
    axes = copy.deepcopy(base.axes)
    axes['region']['values'] = [-0.0, 1.0]
    state = base.state.copy()
    state.loc[state['region'] == 0.0, 'region'] = -0.0
    return base, axes, state


def _signs(values):
    """Text form keeps the sign of zero: '-0.0' vs '0.0'."""
    return [repr(float(v)) for v in values]


@pytest.mark.parametrize('part', ['axes', 'state'])
def test_S9R_03_direct_construction_leaves_caller_objects_unchanged(part):
    """B: constructing a node must not rewrite the caller's axes dict or state table."""
    base, axes, state = _caller_objects()
    axes_before = _signs(axes['region']['values'])
    state_before = _signs(state['region'])
    dataclasses.replace(base, axes=axes, state=state)
    if part == 'axes':
        assert _signs(axes['region']['values']) == axes_before == ['-0.0', '1.0']
    else:
        assert _signs(state['region']) == state_before
        assert '-0.0' in state_before


def test_S9R_04_node_still_normalises_its_own_copies():
    """Control: the constructed node itself carries 0.0, as before; ordinary builds unchanged."""
    base, axes, state = _caller_objects()
    node = dataclasses.replace(base, axes=axes, state=state)
    assert _signs(node.axes['region']['values']) == ['0.0', '1.0']
    assert '-0.0' not in _signs(node.state['region'])
    plain = ss.build_histogram_state([0.2, 0.3], edges=[0., 1.], name='s9/h',
                                     partition_id='A', categories={'region': ['x', 'y']})
    assert plain.axes['region']['values'] == ['x', 'y']
    assert ss.StatisticalNode.from_payload(plain.to_payload()).to_payload() == plain.to_payload()
