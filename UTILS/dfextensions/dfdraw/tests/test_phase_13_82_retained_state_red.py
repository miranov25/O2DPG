"""Independent first-slice RED/GREEN oracles for generic retained statistical state.

This is deliberately independent of dfdraw's rendering/statistical executor.
First-slice name and signatures are internal candidates, not public API locks.
"""
import importlib
import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def state_api():
    return importlib.import_module('dfextensions.dfdraw.statistical_state')


def test_t1_weighted_histogram_against_numpy(state_api):
    x = np.array([-2., 0., 0.2, 0.8, 1., 1.2, 2., 3., np.nan])
    w = np.array([9., 2., -1., 3., 4., 5., -2., 8., 7.])
    edges = [0., 1., 2.]
    node = state_api.build_histogram_state(x, edges=edges, weights=w,
        name='qa/weighted', partition_id='p1', x_source='x', weights_source='w')
    mask = np.isfinite(x) & np.isfinite(w)
    expected_n = np.histogram(x[mask], edges)[0]
    expected_w = np.histogram(x[mask], edges, weights=w[mask])[0]
    expected_w2 = np.histogram(x[mask], edges, weights=w[mask] ** 2)[0]
    got = node.state.set_index('x_bin').sort_index()
    assert list(got.index) == [0, 1]
    np.testing.assert_array_equal(got.entries.to_numpy(), expected_n)
    np.testing.assert_allclose(got.sum_weights.to_numpy(), expected_w, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.sum_weights_squared.to_numpy(), expected_w2, rtol=0, atol=1e-12)
    assert str(got.entries.dtype) == 'int64'
    assert str(got.sum_weights.dtype) == 'float64'


def test_t2_cancelled_weight_is_not_empty(state_api):
    node = state_api.build_histogram_state([0.1, 0.5], edges=[0, 1, 2],
        weights=[2., -2.], name='qa/signed', partition_id='p1')
    assert len(node.state) == 1
    r = node.state.iloc[0]
    assert r['entries'] == 2
    assert r['sum_weights'] == 0.
    assert r['sum_weights_squared'] == 8.


def test_t3_weighted_profile_against_independent_raw_sums(state_api):
    x = np.array([0.1, 0.3, 0.9, 1.2, 1.4, 2.])
    y = np.array([3., -2., 8., 6., -1., 5.])
    w = np.array([1., 2., -1., 1., 4., -3.])
    edges = [0., 1., 2.]
    node = state_api.build_profile_state(x, y, edges=edges, weights=w,
        name='qa/profile', partition_id='p1', x_source='x', y_source='y')
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    expected = [np.histogram(x[keep], edges)[0].astype('int64')]
    for z in [w, w*w, w*y, w*y*y]:
        expected.append(np.histogram(x[keep], edges, weights=z[keep])[0])
    cols = ['entries','sum_weights','sum_weights_squared',
            'sum_weight_times_value','sum_weight_times_value_squared']
    got = node.state.set_index('x_bin').sort_index()
    for col, arr in zip(cols, expected):
        np.testing.assert_allclose(got[col].to_numpy(), arr, rtol=0, atol=1e-12)


def test_t4_categorical_coordinate_is_semantic_dimension(state_api):
    x = [0.1, 0.5, 1.5, 0.2, 1.7]
    cat = ['A', 'B', 'A', 'A', 'B']
    node = state_api.build_histogram_state(x, edges=[0,1,2],
        categories={'group':cat}, category_sources={'group':'run_class'},
        name='qa/group', partition_id='p1')
    assert node.dimensions == ['x_bin', 'group']
    assert node.axes['group']['source'] == 'run_class'
    expected = {(0,'A'):2,(0,'B'):1,(1,'A'):1,(1,'B'):1}
    actual = {(r.x_bin, r.group): r.entries for r in node.state.itertuples(index=False)}
    assert actual == expected
    assert node.axes['group']['values'] == ['A', 'B']


def test_t5_schema_name_and_partition_contract(state_api):
    with pytest.raises(ValueError, match='name'):
        state_api.build_histogram_state([0.2], edges=[0,1], name='', partition_id='p1')
    with pytest.raises(ValueError, match='partition_id'):
        state_api.build_histogram_state([0.2], edges=[0,1], name='qa/test', partition_id='')
    node = state_api.build_histogram_state([0.2], edges=[0,1],
        name='qa/test', partition_id='p1', provenance={'dataset':'data.root'})
    assert node.schema_version == 1
    assert node.provenance['partition_id'] == 'p1'
    assert node.provenance['dataset'] == 'data.root'
    assert node.name == 'qa/test'


def test_t6_sparse_underflow_overflow_nan_and_last_edge(state_api):
    node = state_api.build_histogram_state([-1, 0, 2, 2.1, np.nan],
        edges=[0,1,2,3,4], name='qa/domain', partition_id='p1')
    got = node.state.set_index('x_bin').entries.to_dict()
    assert got == {0:1, 2:2}  # upper edge 2 belongs to [2,3)
    terminal = state_api.build_histogram_state([0.,4.,4.1], edges=[0.,1.,2.,3.,4.],
        name='qa/terminal', partition_id='p1')
    assert terminal.state.set_index('x_bin').entries.to_dict() == {0:1,3:1}


def test_t7_one_node_class_different_sufficient_state(state_api):
    kwargs = dict(edges=[0,1], name='qa/test', partition_id='p1')
    hist = state_api.build_histogram_state([0.5], **kwargs)
    prof = state_api.build_profile_state([0.5], [2.], **kwargs)
    assert type(hist) is type(prof) is state_api.StatisticalNode
    assert hist.dimensions == prof.dimensions
    assert len(prof.state_schema) == len(hist.state_schema) + 2
    assert 'sum_weight_times_value' in prof.state_schema


def test_t8_logical_payload_round_trip_with_metadata(state_api):
    node = state_api.build_profile_state([0.2,0.5,1.1], [1.,2.,3.],
        edges=[0.,1.,2.], weights=[1.,2.,-1.],
        categories={'sector':['A','A','B']},
        name='physics/pt_profile', partition_id='run523001',
        x_source='pt', y_source='dcar', weights_source='corr_w',
        semantic_description={'expression':'dcar:pt','type':'profile','weights':'corr_w'},
        metadata={'x_title':'Transverse momentum', 'unit':'GeV/c'},
        provenance={'source_sha256':'abcdef'})
    encoded = json.loads(json.dumps(node.to_payload(), allow_nan=False))
    restored = state_api.StatisticalNode.from_payload(encoded)
    assert restored.to_payload() == encoded
    pd.testing.assert_frame_equal(restored.state, node.state, check_dtype=True)
    assert restored.provenance['source_sha256'] == 'abcdef'
    with pytest.raises(ValueError, match='schema_version'):
        state_api.StatisticalNode.from_payload({**encoded,'schema_version':999})
    with pytest.raises(ValueError, match='coordinate|duplicate'):
        bad = dict(encoded)
        bad['state'] = encoded['state'] + encoded['state'][:1]
        state_api.StatisticalNode.from_payload(bad)


def test_t9_invalid_resolved_edges_refused(state_api):
    for edges in ([0,0,1], [2,1,0], [0,float('nan'),2], [0]):
        with pytest.raises(ValueError, match='edges'):
            state_api.build_histogram_state([0.2], edges=edges,
                name='qa/bad', partition_id='p1')


def test_t10_nonfinite_weight_and_profile_value_not_accumulated(state_api):
    node = state_api.build_profile_state([0.1,0.2,0.3,0.4],
        [1.0,np.nan,3.,4.], edges=[0,1], weights=[1.,1.,np.inf,2.],
        name='qa/nan', partition_id='p1')
    got = node.state.iloc[0]
    assert got.entries == 2
    assert got.sum_weights == 3.
    assert got.sum_weight_times_value == 9.


def test_t11_numeric_categories_preserve_identity(state_api):
    node = state_api.build_histogram_state([0.2, 0.3, 0.4], edges=[0, 1],
        categories={'side': np.array([0, 1, 0], dtype='int64')},
        name='qa/side', partition_id='p1')
    assert node.axes['side']['values'] == [0, 1]
    assert {(r.side, r.entries) for r in node.state.itertuples()} == {(0, 2), (1, 1)}
    assert state_api.StatisticalNode.from_payload(node.to_payload()).to_payload() == node.to_payload()


def test_t12_corrupt_payload_fractional_entries_refused(state_api):
    node = state_api.build_histogram_state([0.1], edges=[0, 1],
        name='qa/payload', partition_id='p1')
    payload = node.to_payload()
    payload['state'][0]['entries'] = 1.5
    with pytest.raises(ValueError, match='entries'):
        state_api.StatisticalNode.from_payload(payload)
    payload = node.to_payload()
    payload['state'][0]['x_bin'] = 0.5
    with pytest.raises(ValueError, match='coordinate'):
        state_api.StatisticalNode.from_payload(payload)


def test_t13_two_named_categorical_dimensions_form_sparse_product(state_api):
    # Independent row-by-row truth; only occupied x_bin×group×side tuples appear.
    x = [0.1, 0.3, 1.5, 1.2]
    group = ['A', 'A', 'B', 'A']
    side = [0, 1, 0, 1]
    node = state_api.build_histogram_state(x, edges=[0,1,2],
        categories={'group':group, 'side':side},
        name='qa/two_category_axes', partition_id='p2')
    assert node.dimensions == ['x_bin','group','side']
    truth = {(0,'A',0):1, (0,'A',1):1, (1,'B',0):1, (1,'A',1):1}
    actual = {(r.x_bin,r.group,r.side):r.entries
              for r in node.state.itertuples(index=False)}
    assert actual == truth
    assert len(node.state) == 4
