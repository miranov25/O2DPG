"""S8: additive partition-preserving Hₙ retained state, M14 completion.

The output has a partition coordinate but never pools its contributors.
S9 explicit information-reducing collapse is intentionally NOT tested here.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import statistical_state as ss
from dfextensions.dfdraw import partitioned_state as ps


def hist(pid, labels=None, *, weights=None, edges=(0., 1., 2.),
         categories=None, category_orders=None, metadata=None):
    if labels is None:
        labels = [0.2, 1.2, 1.3]
    kwargs = dict(name='s8/h', partition_id=pid, edges=edges,
                  x_source='pt', weights_source='w', weights=weights,
                  metadata={'unit': 'arb'} if metadata is None else metadata)
    if categories is not None:
        kwargs['categories'] = {'region': categories}
    if category_orders is not None:
        kwargs['category_orders'] = {'region': category_orders}
    return ss.build_histogram_state(labels, **kwargs)


def prof(pid, *, x=(0.2, 0.4, 1.2), y=(2.0, 3.0, 5.0),
         weights=(1.0, -1.0, 2.0), categories=None, category_orders=None):
    kw = dict(edges=(0., 1., 2.), name='s8/p', partition_id=pid,
              x_source='pt', y_source='bias', weights_source='w',
              weights=weights, metadata={'unit': 'arb'})
    if categories is not None:
        kw['categories'] = {'region': categories}
    if category_orders is not None:
        kw['category_orders'] = {'region': category_orders}
    return ss.build_profile_state(x, y, **kw)


def assert_leaf_equal(a, b):
    assert a.axes == b.axes
    assert a.provenance == b.provenance
    pd.testing.assert_frame_equal(a.state, b.state, check_exact=True)


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
def test_s8_01_independent_weighted_moments_from_raw_observations(kind):
    raw = [
        ('A', np.array([0.2, 0.6, 1.2]), np.array([1.0, -1.0, 0.0]), np.array([3., 4., 2.])),
        ('B', np.array([0.3, 1.5]), np.array([4.0, -2.0]), np.array([6., -3.])),
        ('C', np.array([1.1, 1.2]), np.array([0.5, 1.5]), np.array([1., 8.])),
    ]
    nodes = []
    for pid, x, weights, y in raw:
        if kind == 'histogram':
            node = hist(pid, x, weights=weights)
        else:
            node = prof(pid, x=x, y=y, weights=weights)
        nodes.append(node)
    out = ps.preserve_partitions(nodes)
    assert out.dimensions == ['partition_id', 'x_bin']
    assert out.partition_ids == ('A', 'B', 'C')
    assert len(out.state) == 5  # occupied bins: A two, B two, C one
    for pid, x, weights, y in raw:
        node = out.get_partition(pid)
        for row in node.state.itertuples(index=False):
            selection = (x >= [0., 1.][row.x_bin]) & (x < [1., 2.][row.x_bin])
            expected_w = weights[selection]
            assert row.entries == len(expected_w)
            np.testing.assert_allclose(row.sum_weights, np.sum(expected_w), atol=1e-12)
            np.testing.assert_allclose(row.sum_weights_squared, np.sum(expected_w ** 2), atol=1e-12)
            if kind == 'profile':
                expected_y = y[selection]
                np.testing.assert_allclose(row.sum_weight_times_value,
                                           np.sum(expected_y * expected_w), atol=1e-12)
                np.testing.assert_allclose(row.sum_weight_times_value_squared,
                                           np.sum(expected_y**2 * expected_w), atol=1e-12)


@pytest.mark.parametrize('kind', ['histogram', 'profile'])
def test_s8_02_nested_preservation_is_order_independent(kind):
    nodes = [hist(p) if kind == 'histogram' else prof(p) for p in ('C','A','B')]
    flat = ps.preserve_partitions(nodes)
    nested = ps.preserve_partitions([ps.preserve_partitions(nodes[:2]), nodes[2]])
    assert flat.to_payload() == nested.to_payload()
    assert flat.to_payload() == ps.preserve_partitions(list(reversed(nodes))).to_payload()
    assert flat.get_partition('C').provenance['partition_id'] == 'C'


def test_s8_03_duplicate_nested_source_refused():
    a, b, c = (hist(p) for p in 'ABC')
    ab = ps.preserve_partitions([a, b])
    with pytest.raises(ValueError, match='duplicate source partition_id'):
        ps.preserve_partitions([ab, a])
    with pytest.raises(ValueError, match='duplicate source partition_id'):
        ps.preserve_partitions([ab, ps.preserve_partitions([b, c])])
    with pytest.raises(ValueError, match='duplicate source partition_id'):
        ps.preserve_partitions([a, hist('A')])


def test_s8_04_already_collapsed_merge_refused():
    a, b, c = (hist(p) for p in 'ABC')
    ab = ss.merge_exact([a, b], partition_id='AB')
    with pytest.raises(ValueError, match='collapsed'):
        ps.preserve_partitions([ab, c])
    with pytest.raises(ValueError, match='collapsed'):
        ps.preserve_partitions([c, ab])


def test_s8_05_source_and_schema_compatibility_checks():
    a = hist('A')
    with pytest.raises(ValueError, match='metadata'):
        ps.preserve_partitions([a, hist('B', metadata={'unit': 'different'})])
    with pytest.raises(ValueError, match='x_bin'):
        ps.preserve_partitions([a, hist('B', edges=(0., 0.5, 2.))])
    with pytest.raises(ValueError, match='name'):
        ps.preserve_partitions([a, ss.build_histogram_state([0.3], edges=(0,1,2),
                                     name='another', partition_id='B')])


@pytest.mark.parametrize('kind', ['histogram','profile'])
@pytest.mark.parametrize('label_order', [[0.0, -0.0], [-0.0, 0.0]])
def test_s8_06_signed_zero_builder_and_roundtrip(kind, label_order):
    n = (hist('A', [0.2,0.3], categories=label_order) if kind == 'histogram' else
         prof('A', x=(0.2,0.3), y=(2., 3.), weights=(1.,1.), categories=label_order))
    assert n.axes['region']['values'] == [0.0]
    assert json.dumps(n.axes['region']['values']) == '[0.0]'
    assert list(n.state['region']) == [0.0]
    assert json.dumps(n.to_payload()['state'][0]['region']) == '0.0'
    r = ss.StatisticalNode.from_payload(n.to_payload())
    assert_leaf_equal(n, r)


@pytest.mark.parametrize('kind', ['histogram','profile'])
def test_s8_07_signed_zero_restore_and_ambiguous_domain_refusal(kind):
    n = hist('A', [0.2], categories=[0.0]) if kind == 'histogram' else \
        prof('A', x=(0.2,), y=(1.,), weights=(1.,), categories=[0.0])
    payload = n.to_payload()
    payload['axes']['region']['values'] = [-0.0]
    payload['state'][0]['region'] = -0.0
    r = ss.StatisticalNode.from_payload(payload)
    assert json.dumps(r.axes['region']['values']) == '[0.0]'
    assert json.dumps(r.to_payload()['state'][0]['region']) == '0.0'
    malformed = n.to_payload()
    malformed['axes']['region']['values'] = [0.0, -0.0]
    with pytest.raises(ValueError, match='ambiguous signed-zero'):
        ss.StatisticalNode.from_payload(malformed)


def test_s8_08_signed_zero_merge_is_nonduplicating():
    a = hist('A', [0.2], categories=[0.0])
    b = hist('B', [0.2], categories=[-0.0])
    preserved = ps.preserve_partitions([b, a])
    assert json.dumps(preserved.axes['region']['values']) == '[0.0]'
    assert len(preserved.state) == 2
    assert preserved.state.entries.sum() == 2


@pytest.mark.parametrize('kind', ['histogram','profile'])
def test_s8_09_declared_order_kept_including_unoccupied_labels(kind):
    labels = pd.Series(pd.Categorical(['OROC2','IROC'],
             categories=['IROC', 'OROC1', 'OROC2', 'OROC3'], ordered=True))
    a = hist('A', [0.2,0.3], categories=labels) if kind == 'histogram' else \
        prof('A', x=(0.2,0.3), y=(1.,2.), weights=(1.,1.), categories=labels)
    b = hist('B', [0.3], categories=['OROC1']) if kind == 'histogram' else \
        prof('B', x=(0.3,), y=(1.,), weights=(1.,), categories=['OROC1'])
    assert a.axes['region']['values'] == ['IROC','OROC1','OROC2','OROC3']
    assert a.axes['region']['ordered'] is True
    out = ps.preserve_partitions([b,a])
    assert out.axes['region']['values'] == ['IROC','OROC1','OROC2','OROC3']
    assert out.axes['region']['ordered'] is True
    assert out.axes['region']['order_origin'] == 'explicit'
    restored = ps.PartitionedStatisticalState.from_payload(out.to_payload())
    assert restored.axes == out.axes
    assert restored.get_partition('A').axes == a.axes


def test_s8_10_explicit_mapping_and_conflicting_orders():
    a = hist('A', [0.2], categories=['OROC2'], category_orders=['IROC','OROC2','OROC3'])
    b = hist('B', [0.2], categories=['IROC'], category_orders=['OROC2','IROC','OROC3'])
    with pytest.raises(ValueError, match='conflicting explicit scientific orders'):
        ps.preserve_partitions([a, b])
    c = hist('C', [0.2], categories=['IROC'], category_orders=['IROC','OROC2','OROC3'])
    assert ps.preserve_partitions([a, c]).axes['region']['values'] == ['IROC','OROC2','OROC3']
    with pytest.raises(ValueError, match='absent from declared order'):
        hist('D', [0.2], categories=['INVALID'], category_orders=['IROC'])
    with pytest.raises(ValueError, match='outside declared scientific order'):
        ps.preserve_partitions([a, hist('E', [0.2], categories=['UNKNOWN'])])


@pytest.mark.parametrize('labels', [[3,1,2], ['Z','A','B'], [2.0,0.0,1.0], [True,False]])
def test_s8_11_unordered_canonical_union_permutation_invariant(labels):
    a = hist('A', [0.2,0.3], categories=labels[:2])
    b = hist('B', [0.2], categories=labels[2:]) if len(labels)>2 else hist('B', [0.2], categories=labels[1:])
    ab = ps.preserve_partitions([a,b])
    ba = ps.preserve_partitions([b,a])
    assert ab.to_payload() == ba.to_payload()
    assert ab.axes['region']['values'] == sorted(set(labels))
    assert ab.get_partition('A').axes['region']['values'] == labels[:2]


def test_s8_12_payload_tampering_detected_and_no_source_loss():
    a,b,c=(hist(p) for p in 'ABC')
    original=ps.preserve_partitions([a,b,c])
    payload = original.to_payload()
    assert ps.PartitionedStatisticalState.from_payload(payload).to_payload() == payload
    broken=copy.deepcopy(payload)
    broken['axes']['partition_id']['values'].pop()
    with pytest.raises(ValueError, match='axes inconsistent'):
        ps.PartitionedStatisticalState.from_payload(broken)
    broken=copy.deepcopy(payload)
    broken['partitions'][1]['provenance']['partition_id'] = 'A'
    with pytest.raises(ValueError, match='duplicate source'):
        ps.PartitionedStatisticalState.from_payload(broken)


def test_s8_13_source_mutation_of_copy_does_not_modify_aggregate():
    out=ps.preserve_partitions([hist('A'),hist('B')])
    a=out.get_partition('A')
    a.state.loc[0,'sum_weights']=12345.
    a.provenance['partition_id']='forged'
    assert out.get_partition('A').state.iloc[0].sum_weights == 1.0
    assert out.partition_ids == ('A','B')
    refs=out.partitions
    refs[0].axes['x_bin']['edges'][0]=-999.
    assert out.axes['x_bin']['edges'][0]==0.


def test_s8_14_invalid_partition_inputs_and_scientific_domain():
    with pytest.raises(ValueError, match='at least two'):
        ps.preserve_partitions([hist('A')])
    with pytest.raises(TypeError, match='accepts'):
        ps.preserve_partitions([hist('A'), 4])
    a=hist('A', [0.2], categories=[1]);b=hist('B', [0.2], categories=[True])
    with pytest.raises(ValueError, match='incompatible categorical scalar types'):
        ps.preserve_partitions([a,b])


@pytest.mark.parametrize('kind', ['histogram','profile'])
def test_s8_15_signed_weight_cancellation_stays_occupied(kind):
    if kind == 'histogram':
        a=hist('A', [0.2,0.3], weights=[2.,-2.])
        b=hist('B', [0.2], weights=[0.])
    else:
        a=prof('A', x=(0.2,0.3), y=(1.,2.), weights=(2.,-2.))
        b=prof('B', x=(0.2,), y=(3.,), weights=(0.,))
    out=ps.preserve_partitions([a,b])
    assert len(out.state)==2
    assert out.state.entries.sum()==3
    assert out.get_partition('A').state.iloc[0].sum_weights==0.
    assert out.get_partition('B').state.iloc[0].entries==1


def test_s8_16_backward_compatible_legacy_merge_exact_order():
    a=hist('A', [0.2], categories=['Z'])
    b=hist('B', [0.2], categories=['A'])
    assert ss.merge_exact([a,b],partition_id='AB').axes['region']['values']==['Z','A']
    assert ss.merge_exact([b,a],partition_id='BA').axes['region']['values']==['A','Z']
    assert ps.preserve_partitions([b,a]).axes['region']['values']==['A','Z']


def test_s8_17_unordered_numeric_values_are_numeric_sorted():
    a=hist('A',[0.2,0.3],categories=[10,2]);b=hist('B',[0.2],categories=[5])
    assert ps.preserve_partitions([a,b]).axes['region']['values']==[2,5,10]


def test_s8_18_partition_id_is_reserved_as_new_s8_coordinate():
    a=ss.build_histogram_state([0.2],edges=[0.,1.],name='q',partition_id='A',categories={'partition_id':['x']})
    b=ss.build_histogram_state([0.3],edges=[0.,1.],name='q',partition_id='B',categories={'partition_id':['y']})
    with pytest.raises(ValueError, match='reserved'):
        ps.preserve_partitions([a,b])
