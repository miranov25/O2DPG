"""PHASE_13_82 P1 adversarial source-partition/provenance RED oracles.

These do not assert cryptographic authenticity of a wholly rewritten file:
they enforce system-owned builder IDs and internally coherent logical payloads.
"""
import copy
import pytest
from dfextensions.dfdraw import statistical_state as ss


def _hist(pid, **kwargs):
    return ss.build_histogram_state([0.3], edges=[0., 1.], name='qa/p1',
                                    partition_id=pid, **kwargs)


def test_p1_01_leaf_builder_rejects_reserved_contributor_keys():
    for reserved in (
        {'contributing_partition_ids': ['FALSE_A']},
        {'contributing_provenance': [{'partition_id': 'FALSE_A'}]},
        {'contributing_partition_ids': ['FALSE_A'],
         'contributing_provenance': [{'partition_id': 'FALSE_A'}]},
    ):
        with pytest.raises(ValueError, match='reserved|contributing'):
            _hist('TRUE_A', provenance={'operator': 'qa', **reserved})


def test_p1_02_direct_merge_cannot_double_count_spoofed_true_partition():
    with pytest.raises(ValueError, match='reserved|contributing'):
        forged = _hist('TRUE_A', provenance={
            'contributing_partition_ids': ['FAKE_A'],
            'contributing_provenance': [{'partition_id':'FAKE_A'}]})
        ss.merge_exact([forged, _hist('TRUE_A')], partition_id='OUT')


def test_p1_03_payload_rejects_inconsistent_contributor_identifiers():
    a,b = _hist('A'),_hist('B')
    merged = ss.merge_exact([a,b], partition_id='AB')
    for bad in ('mismatch', 'length', 'duplicate', 'nested'):
        payload = copy.deepcopy(merged.to_payload())
        prov = payload['provenance']
        if bad == 'mismatch':
            prov['contributing_partition_ids'][0] = 'INVENTED'
        elif bad == 'length':
            prov['contributing_provenance'].pop()
        elif bad == 'duplicate':
            prov['contributing_provenance'][1]['partition_id'] = 'A'
        else:
            prov['contributing_provenance'][0]['contributing_partition_ids'] = ['A']
        with pytest.raises(ValueError, match='contribut|partition|provenance'):
            ss.StatisticalNode.from_payload(payload)


def test_p1_04_payload_rejects_leaf_with_contributor_spoof():
    payload = _hist('TRUE_A').to_payload()
    payload['provenance']['contributing_partition_ids']=['FAKE_A']
    payload['provenance']['contributing_provenance']=[{'partition_id':'FAKE_A'}]
    with pytest.raises(ValueError, match='contribut|partition|provenance'):
        ss.StatisticalNode.from_payload(payload)


def test_p1_05_output_identity_cannot_reuse_direct_merged_input_identity():
    a,b,c = _hist('A'),_hist('B'),_hist('C')
    ab=ss.merge_exact([a,b],partition_id='AB')
    with pytest.raises(ValueError, match='output partition_id|partition_id'):
        ss.merge_exact([ab,c], partition_id='AB')


def test_p1_06_nested_roundtrip_preserves_truth_and_refuses_duplicate():
    a,b,c = _hist('A'),_hist('B'),_hist('C')
    ab=ss.merge_exact([a,b],partition_id='AB')
    restored=ss.StatisticalNode.from_payload(ab.to_payload())
    abc=ss.merge_exact([restored,c],partition_id='ABC')
    assert abc.provenance['contributing_partition_ids'] == ['A','B','C']
    assert [p['partition_id'] for p in abc.provenance['contributing_provenance']] == ['A','B','C']
    with pytest.raises(ValueError, match='duplicate partition'):
        ss.merge_exact([abc,ss.StatisticalNode.from_payload(a.to_payload())], partition_id='BAD')


def test_p1_07_duplicate_direct_input_identity_rejected_when_not_leaf_contributor():
    a,b,c,d=_hist('A'),_hist('B'),_hist('C'),_hist('D')
    ab=ss.merge_exact([a,b],partition_id='POOL')
    cd=ss.merge_exact([c,d],partition_id='POOL')
    with pytest.raises(ValueError, match='duplicate.*partition|partition.*duplicate'):
        ss.merge_exact([ab,cd],partition_id='FINAL')


@pytest.mark.parametrize('integer,boolean', [(0,False),(1,True)])
def test_m14_integer_boolean_category_identity_is_not_silently_collapsed(integer, boolean):
    """Diagnostic for separately flagged M14; do not merge int and bool identities."""
    a = ss.build_histogram_state([0.1], edges=[0,1], name='qa/m14',
                                 partition_id='A', categories={'flag':[integer]})
    b = ss.build_histogram_state([0.2], edges=[0,1], name='qa/m14',
                                 partition_id='B', categories={'flag':[boolean]})
    with pytest.raises(ValueError, match='category value types'):
        ss.merge_exact([a,b], partition_id='AB')
