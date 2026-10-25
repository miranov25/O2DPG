"""PHASE_13_82 exact retained-state merge RED/GREEN oracles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import statistical_state as ss


def _assert_same_state(actual, expected):
    assert actual.name == expected.name
    assert actual.statistic_kind == expected.statistic_kind
    assert actual.dimensions == expected.dimensions
    assert actual.axes == expected.axes
    assert actual.semantic_roles == expected.semantic_roles
    assert actual.state_schema == expected.state_schema
    assert actual.semantic_description == expected.semantic_description
    assert actual.metadata == expected.metadata
    pd.testing.assert_frame_equal(
        actual.state.reset_index(drop=True), expected.state.reset_index(drop=True),
        check_dtype=True, check_exact=False, rtol=0, atol=1e-12)


def test_m1_weighted_histogram_merge_equals_direct_pooled_state():
    edges = [0.0, 1.0, 2.0, 3.0]
    xa = np.array([0.1, 0.4, 1.2, 2.8, np.nan])
    wa = np.array([1.0, 2.0, -1.0, 3.0, 9.0])
    xb = np.array([0.2, 1.4, 1.8, 3.0, -2.0])
    wb = np.array([-1.0, 4.0, 2.0, -3.0, 7.0])
    a = ss.build_histogram_state(xa, edges=edges, weights=wa,
        name='qa/hist', partition_id='part-A', x_source='x', weights_source='w',
        metadata={'unit':'arb'}, provenance={'source':'A.root'})
    b = ss.build_histogram_state(xb, edges=edges, weights=wb,
        name='qa/hist', partition_id='part-B', x_source='x', weights_source='w',
        metadata={'unit':'arb'}, provenance={'source':'B.root'})
    merged = ss.merge_exact([a,b], partition_id='pooled-AB')
    pooled = ss.build_histogram_state(np.concatenate([xa,xb]), edges=edges,
        weights=np.concatenate([wa,wb]), name='qa/hist', partition_id='ref',
        x_source='x', weights_source='w', metadata={'unit':'arb'})
    _assert_same_state(merged, pooled)
    assert merged.provenance['partition_id'] == 'pooled-AB'
    assert merged.provenance['contributing_partition_ids'] == ['part-A','part-B']
    assert [p['source'] for p in merged.provenance['contributing_provenance']] == ['A.root','B.root']


def test_m2_weighted_profile_merge_equals_direct_pooled_state():
    edges=[0.,1.,2.]
    xa=np.array([0.1,0.7,1.1,1.9]); ya=np.array([3.,-2.,4.,7.]); wa=np.array([1.,2.,-1.,3.])
    xb=np.array([0.2,0.8,1.2,2.0]); yb=np.array([5.,6.,-3.,8.]); wb=np.array([-2.,1.,4.,-1.])
    kwargs=dict(edges=edges,name='qa/profile',x_source='x',y_source='y',weights_source='w',metadata={'x_title':'x','y_title':'y'})
    a=ss.build_profile_state(xa,ya,weights=wa,partition_id='A',**kwargs)
    b=ss.build_profile_state(xb,yb,weights=wb,partition_id='B',**kwargs)
    merged=ss.merge_exact([a,b],partition_id='AB')
    pooled=ss.build_profile_state(np.concatenate([xa,xb]),np.concatenate([ya,yb]),weights=np.concatenate([wa,wb]),partition_id='ref',**kwargs)
    _assert_same_state(merged, pooled)


def test_m3_signed_weight_cancellation_across_partitions_stays_occupied():
    a=ss.build_histogram_state([0.2],edges=[0,1],weights=[2.],name='qa/cancel',partition_id='A')
    b=ss.build_histogram_state([0.4],edges=[0,1],weights=[-2.],name='qa/cancel',partition_id='B')
    merged=ss.merge_exact([a,b],partition_id='AB'); row=merged.state.iloc[0]
    assert len(merged.state)==1 and row.entries==2 and row.sum_weights==0. and row.sum_weights_squared==8.


def test_m4_sparse_categorical_union_matches_direct_pooled_state():
    edges=[0,1,2]
    a=ss.build_histogram_state([0.1,1.2,1.4],edges=edges,categories={'sector':['A','A','B']},category_sources={'sector':'sector'},name='qa/categories',partition_id='A')
    b=ss.build_histogram_state([0.2,0.4,1.8],edges=edges,categories={'sector':['C','B','C']},category_sources={'sector':'sector'},name='qa/categories',partition_id='B')
    merged=ss.merge_exact([a,b],partition_id='AB')
    pooled=ss.build_histogram_state([0.1,1.2,1.4,0.2,0.4,1.8],edges=edges,categories={'sector':['A','A','B','C','B','C']},category_sources={'sector':'sector'},name='qa/categories',partition_id='ref')
    assert merged.axes['sector']['values']==['A','B','C']
    _assert_same_state(merged, pooled)


def test_m5_duplicate_partition_refused_including_nested_contributors():
    a=ss.build_histogram_state([0.1],edges=[0,1],name='qa/dup',partition_id='A')
    b=ss.build_histogram_state([0.2],edges=[0,1],name='qa/dup',partition_id='B')
    with pytest.raises(ValueError,match='duplicate partition'):
        ss.merge_exact([a,a],partition_id='bad')
    ab=ss.merge_exact([a,b],partition_id='AB')
    c=ss.build_histogram_state([0.3],edges=[0,1],name='qa/dup',partition_id='A')
    with pytest.raises(ValueError,match='duplicate partition'):
        ss.merge_exact([ab,c],partition_id='bad2')


def test_m6_different_name_refused():
    a=ss.build_histogram_state([0.1],edges=[0,1],name='qa/a',partition_id='A')
    b=ss.build_histogram_state([0.2],edges=[0,1],name='qa/b',partition_id='B')
    with pytest.raises(ValueError,match='name'):
        ss.merge_exact([a,b],partition_id='AB')


def test_m7_different_binned_axis_refused_without_rebinning():
    a=ss.build_histogram_state([0.1],edges=[0,1,2],name='qa/axis',partition_id='A')
    b=ss.build_histogram_state([0.2],edges=[0,0.5,1,2],name='qa/axis',partition_id='B')
    with pytest.raises(ValueError,match='axis|edges|compatible'):
        ss.merge_exact([a,b],partition_id='AB')


def test_m8_statistic_kind_and_state_contract_mismatch_refused():
    h=ss.build_histogram_state([0.1],edges=[0,1],name='qa/kind',partition_id='A')
    p=ss.build_profile_state([0.2],[1.],edges=[0,1],name='qa/kind',partition_id='B')
    with pytest.raises(ValueError,match='statistic_kind|state_schema|compatible'):
        ss.merge_exact([h,p],partition_id='AB')


def test_m9_semantic_source_definition_mismatch_refused():
    a=ss.build_profile_state([0.1],[1.],edges=[0,1],name='qa/source',partition_id='A',x_source='pt',y_source='residual')
    b=ss.build_profile_state([0.2],[2.],edges=[0,1],name='qa/source',partition_id='B',x_source='p',y_source='residual')
    with pytest.raises(ValueError,match='semantic|axis|source|compatible'):
        ss.merge_exact([a,b],partition_id='AB')


def test_m10_metadata_mismatch_refused_for_first_exact_merge_slice():
    a=ss.build_histogram_state([0.1],edges=[0,1],name='qa/meta',partition_id='A',metadata={'unit':'GeV'})
    b=ss.build_histogram_state([0.2],edges=[0,1],name='qa/meta',partition_id='B',metadata={'unit':'MeV'})
    with pytest.raises(ValueError,match='metadata|compatible'):
        ss.merge_exact([a,b],partition_id='AB')


def test_m11_requires_at_least_two_nodes_and_explicit_output_partition():
    a=ss.build_histogram_state([0.1],edges=[0,1],name='qa/input',partition_id='A')
    with pytest.raises(ValueError,match='at least two'):
        ss.merge_exact([a],partition_id='only')
    with pytest.raises(ValueError,match='partition_id'):
        ss.merge_exact([a,a],partition_id='')


def _finalize_profile_row(row):
    """Independent reference finalizer matching current weighted profile contract."""
    n = int(row.entries)
    sw = float(row.sum_weights)
    sw2 = float(row.sum_weights_squared)
    swy = float(row.sum_weight_times_value)
    swy2 = float(row.sum_weight_times_value_squared)
    if sw <= 0:
        return np.nan, np.nan, np.nan
    mean = swy / sw
    if n <= 1:
        return mean, np.nan, np.nan
    variance = swy2 / sw - mean * mean
    # Protect only against roundoff-level negative values in the reference.
    variance = max(variance, 0.0)
    std = np.sqrt(variance)
    n_eff = sw * sw / sw2 if sw2 > 0 else 1.0
    sem = std / np.sqrt(n_eff) if n_eff > 0 else np.nan
    return mean, std, sem


def test_m12_merged_profile_finalization_matches_direct_pooled_raw_data():
    edges = np.array([0.0, 1.0, 2.0])
    xa = np.array([0.1, 0.4, 1.1])
    ya = np.array([2.0, 4.0, 5.0])
    wa = np.array([1.0, 2.0, 1.5])
    xb = np.array([0.2, 0.9, 1.4, 1.8])
    yb = np.array([8.0, 1.0, -2.0, 6.0])
    wb = np.array([3.0, 1.0, 2.0, 4.0])

    a = ss.build_profile_state(xa, ya, edges=edges, weights=wa,
        name='qa/finalize', partition_id='A', x_source='x', y_source='y')
    b = ss.build_profile_state(xb, yb, edges=edges, weights=wb,
        name='qa/finalize', partition_id='B', x_source='x', y_source='y')
    merged = ss.merge_exact([a, b], partition_id='AB')

    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    w = np.concatenate([wa, wb])
    for row in merged.state.itertuples(index=False):
        i = int(row.x_bin)
        in_bin = ((x >= edges[i]) &
                  ((x < edges[i + 1]) if i < len(edges) - 2 else (x <= edges[i + 1])))
        yy = y[in_bin]
        ww = w[in_bin]
        mean = np.sum(ww * yy) / np.sum(ww)
        variance = np.sum(ww * (yy - mean) ** 2) / np.sum(ww)
        std = np.sqrt(variance)
        n_eff = np.sum(ww) ** 2 / np.sum(ww ** 2)
        sem = std / np.sqrt(n_eff)
        got = _finalize_profile_row(row)
        np.testing.assert_allclose(got, (mean, std, sem), rtol=0, atol=1e-12)


def test_m13_payload_roundtrip_states_merge_to_direct_pooled_truth():
    edges = np.array([0.0, 1.0, 2.0])
    xa = np.array([0.1, 0.8, 1.2])
    ya = np.array([1.0, 3.0, 5.0])
    wa = np.array([1.0, 2.0, 1.5])
    xb = np.array([0.2, 1.1, 1.9])
    yb = np.array([7.0, -1.0, 4.0])
    wb = np.array([3.0, 2.0, 4.0])

    a = ss.build_profile_state(
        xa, ya, edges=edges, weights=wa,
        name='qa/saved-merge', partition_id='A', x_source='x', y_source='y'
    )
    b = ss.build_profile_state(
        xb, yb, edges=edges, weights=wb,
        name='qa/saved-merge', partition_id='B', x_source='x', y_source='y'
    )

    # The first persistence slice freezes backend-neutral payload semantics.
    # Exact merge must therefore work on states restored from those payloads,
    # not only on objects still resident from their original builders.
    restored_a = ss.StatisticalNode.from_payload(a.to_payload())
    restored_b = ss.StatisticalNode.from_payload(b.to_payload())
    merged = ss.merge_exact([restored_a, restored_b], partition_id='AB')

    direct = ss.build_profile_state(
        np.concatenate([xa, xb]), np.concatenate([ya, yb]),
        edges=edges, weights=np.concatenate([wa, wb]),
        name='qa/saved-merge', partition_id='DIRECT', x_source='x', y_source='y'
    )

    _assert_same_state(merged, direct)
    assert merged.provenance['contributing_partition_ids'] == ['A', 'B']
