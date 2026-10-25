"""Experimental internal retained-state node for dfdraw N-dimensional statistics.

First additive slice only: weighted 1-D histogram/profile with optional named
categorical coordinates. No executor integration, merge or physical storage.
The logical payload is a *candidate* schema awaiting phase-owner review; it
must not yet be treated as a stable public or on-disk serialization format.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
_COMMON_STATE = {
    'entries': 'int64',
    'sum_weights': 'float64',
    'sum_weights_squared': 'float64',
}
_PROFILE_EXTRA = {
    'sum_weight_times_value': 'float64',
    'sum_weight_times_value_squared': 'float64',
}


def _json_copy(value: Any, label: str) -> Any:
    """Reject nonportable/callable payload objects rather than serializing code."""
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be JSON-safe and finite: {exc}') from exc


def _validate_edges(edges: Any) -> list[float]:
    try:
        arr = np.asarray(edges, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('edges must be finite, strictly increasing numbers') from exc
    if (arr.ndim != 1 or len(arr) < 2 or not np.all(np.isfinite(arr))
            or not np.all(arr[1:] > arr[:-1])):
        raise ValueError('edges must be finite, strictly increasing numbers')
    return arr.tolist()


def _columns(dimensions: list[str], state_schema: Mapping[str, str]):
    return dimensions + list(state_schema)


def _typed_frame(rows, dimensions, state_schema):
    frame = pd.DataFrame(rows, columns=_columns(dimensions, state_schema))
    # Logical payload integer coordinates and counts must never silently truncate.
    for row in rows:
        if isinstance(row, dict):
            if set(row) != set(_columns(dimensions, state_schema)):
                raise ValueError('state record has missing or unknown coordinates/fields')
            xbin, entry = row['x_bin'], row['entries']
        else:
            xbin, entry = row[0], row[len(dimensions)]
        if type(xbin) is not int:
            raise ValueError('state x_bin coordinate must be an exact integer')
        if type(entry) is not int:
            raise ValueError('entries must be an exact integer')
    frame['x_bin'] = frame['x_bin'].astype('int64')
    for field, dtype in state_schema.items():
        frame[field] = frame[field].astype(dtype)
    return frame


def _validate_node(node: 'StatisticalNode') -> None:
    if node.schema_version != SCHEMA_VERSION:
        raise ValueError(f'unsupported schema_version: {node.schema_version!r}')
    if not isinstance(node.name, str) or not node.name.strip():
        raise ValueError('name must be a nonempty durable string')
    if node.statistic_kind not in ('histogram', 'profile'):
        raise ValueError('statistic_kind must be histogram or profile')
    if not isinstance(node.provenance, dict):
        raise ValueError('provenance must be a dictionary with partition_id')
    partition_id = node.provenance.get('partition_id')
    if not isinstance(partition_id, str) or not partition_id.strip():
        raise ValueError('partition_id must be a nonempty stable string')
    if (not isinstance(node.dimensions, list) or not node.dimensions
            or node.dimensions[0] != 'x_bin' or len(set(node.dimensions)) != len(node.dimensions)):
        raise ValueError('dimensions must begin with unique x_bin')
    if set(node.axes) != set(node.dimensions):
        raise ValueError('axes must define exactly the declared dimensions')
    edges = _validate_edges(node.axes['x_bin']['edges'])
    if node.axes['x_bin'].get('kind') != 'binned':
        raise ValueError('x_bin must have kind=binned')
    for dim in node.dimensions[1:]:
        axis = node.axes[dim]
        values = axis.get('values')
        if axis.get('kind') != 'categorical' or not isinstance(values, list):
            raise ValueError(f'coordinate {dim}: categorical axis and values required')
        if len(values) != len({json.dumps(v, sort_keys=True) for v in values}):
            raise ValueError(f'coordinate {dim}: duplicate declared values')
    expected_schema = dict(_COMMON_STATE)
    if node.statistic_kind == 'profile':
        expected_schema.update(_PROFILE_EXTRA)
    if node.state_schema != expected_schema:
        raise ValueError('incompatible state_schema for statistic_kind')
    if list(node.state.columns) != _columns(node.dimensions, expected_schema):
        raise ValueError('state coordinates/fields do not match schema')
    for field, dtype in {'x_bin': 'int64', **expected_schema}.items():
        if str(node.state[field].dtype) != dtype:
            raise ValueError(f'state field {field}: expected dtype {dtype}')
    if len(node.state):
        idx = node.state['x_bin'].to_numpy(dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= len(edges) - 1):
            raise ValueError('state x_bin coordinate outside resolved edges')
        if (node.state['entries'] <= 0).any():
            raise ValueError('state must contain only occupied coordinates (entries > 0)')
        for dim in node.dimensions[1:]:
            allowed = {json.dumps(v, sort_keys=True) for v in node.axes[dim]['values']}
            observed = {json.dumps(v, sort_keys=True) for v in node.state[dim]}
            if not observed <= allowed:
                raise ValueError(f'coordinate {dim}: undeclared category value')
        if node.state.duplicated(subset=node.dimensions).any():
            raise ValueError('duplicate state coordinate tuple')
        for col in expected_schema:
            if not np.all(np.isfinite(node.state[col].to_numpy(dtype=np.float64))):
                raise ValueError(f'state field {col}: nonfinite values not supported')
    for field in ('axes', 'semantic_roles', 'state_schema', 'semantic_description', 'metadata', 'provenance'):
        _json_copy(getattr(node, field), field)


@dataclass
class StatisticalNode:
    """Sparse, named retained statistic; no rendering or disk ownership."""
    schema_version: int
    name: str
    statistic_kind: str
    dimensions: list[str]
    axes: dict[str, Any]
    semantic_roles: dict[str, str]
    state_schema: dict[str, str]
    state: pd.DataFrame
    semantic_description: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]

    def __post_init__(self):
        _validate_node(self)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe *logical* payload, not a durable disk format."""
        _validate_node(self)
        records = []
        for row in self.state.to_dict('records'):
            record = {}
            for key, value in row.items():
                if isinstance(value, np.generic):
                    value = value.item()
                record[key] = value
            records.append(record)
        payload = {
            'schema_version': self.schema_version,
            'name': self.name,
            'statistic_kind': self.statistic_kind,
            'dimensions': self.dimensions,
            'axes': self.axes,
            'semantic_roles': self.semantic_roles,
            'state_schema': self.state_schema,
            'state': records,
            'semantic_description': self.semantic_description,
            'metadata': self.metadata,
            'provenance': self.provenance,
        }
        return _json_copy(payload, 'statistical payload')

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> 'StatisticalNode':
        if not isinstance(payload, Mapping) or payload.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('unsupported schema_version in statistical payload')
        fields = ['schema_version','name','statistic_kind','dimensions','axes',
                  'semantic_roles','state_schema','state','semantic_description',
                  'metadata','provenance']
        if set(payload) != set(fields):
            raise ValueError('statistical payload has missing or unknown fields')
        value = _json_copy(dict(payload), 'statistical payload')
        if not isinstance(value['state'], list):
            raise ValueError('statistical payload state must be a list of records')
        if not isinstance(value['dimensions'], list) or not isinstance(value['state_schema'], dict):
            raise ValueError('dimensions/state_schema invalid')
        frame = _typed_frame(value.pop('state'), value['dimensions'], value['state_schema'])
        return cls(**value, state=frame)



def _partition_ids_for_merge(node: 'StatisticalNode') -> list[str]:
    """Return original contributing partitions, flattening prior exact merges."""
    contributed = node.provenance.get('contributing_partition_ids')
    if contributed is None:
        return [node.provenance['partition_id']]
    if (not isinstance(contributed, list) or not contributed
            or any(not isinstance(v, str) or not v.strip() for v in contributed)
            or len(set(contributed)) != len(contributed)):
        raise ValueError('invalid contributing_partition_ids in merge provenance')
    return list(contributed)


def _contributing_provenance_for_merge(node: 'StatisticalNode') -> list[dict[str, Any]]:
    """Preserve source-level provenance without nesting merge wrappers."""
    contributed = node.provenance.get('contributing_provenance')
    if contributed is None:
        return [copy.deepcopy(node.provenance)]
    if not isinstance(contributed, list) or any(not isinstance(v, dict) for v in contributed):
        raise ValueError('invalid contributing_provenance in merge provenance')
    return copy.deepcopy(contributed)


def _axes_for_exact_merge(nodes: Sequence['StatisticalNode']) -> dict[str, Any]:
    """Validate axis identity and union categorical domains in input order."""
    first = nodes[0]
    merged_axes = copy.deepcopy(first.axes)
    for node in nodes[1:]:
        # Physical binned axes are exact-merge compatible only when identical.
        if node.axes['x_bin'] != first.axes['x_bin']:
            raise ValueError('incompatible x_bin axis/source/edges for exact merge')
        for dim in first.dimensions[1:]:
            left = {k: v for k, v in first.axes[dim].items() if k != 'values'}
            right = {k: v for k, v in node.axes[dim].items() if k != 'values'}
            if left != right:
                raise ValueError(f'incompatible categorical axis/source for exact merge: {dim}')
            values = merged_axes[dim]['values']
            seen = {json.dumps(v, sort_keys=True) for v in values}
            for value in node.axes[dim]['values']:
                key = json.dumps(value, sort_keys=True)
                if key not in seen:
                    values.append(copy.deepcopy(value))
                    seen.add(key)
            if len({type(v) for v in values}) > 1:
                raise ValueError(f'incompatible category value types for exact merge: {dim}')
    return merged_axes


def _validate_exact_merge_compatibility(nodes: Sequence['StatisticalNode']) -> dict[str, Any]:
    first = nodes[0]
    for node in nodes:
        if not isinstance(node, StatisticalNode):
            raise TypeError('exact merge accepts StatisticalNode inputs only')
        _validate_node(node)
    for node in nodes[1:]:
        if node.schema_version != first.schema_version:
            raise ValueError('incompatible schema_version for exact merge')
        if node.name != first.name:
            raise ValueError('incompatible name for exact merge')
        if node.statistic_kind != first.statistic_kind:
            raise ValueError('incompatible statistic_kind for exact merge')
        if node.dimensions != first.dimensions:
            raise ValueError('incompatible dimensions for exact merge')
        if node.state_schema != first.state_schema:
            raise ValueError('incompatible state_schema for exact merge')
        if node.semantic_roles != first.semantic_roles:
            raise ValueError('incompatible semantic_roles for exact merge')
        if node.semantic_description != first.semantic_description:
            raise ValueError('incompatible semantic_description for exact merge')
        # First bounded slice is deliberately strict.  Metadata relaxation needs
        # an explicit policy because metadata can carry interpretation semantics.
        if node.metadata != first.metadata:
            raise ValueError('incompatible metadata for exact merge')
    return _axes_for_exact_merge(nodes)


def merge_exact(nodes: Sequence['StatisticalNode'], *, partition_id: str) -> 'StatisticalNode':
    """Exactly pool compatible retained states by additive sufficient fields.

    This is the first collapsed-partition merge slice.  It does not implement
    partition-preserving mode, rebinning, derived aggregation, persistence, or
    rendering.  The caller supplies a new stable ``partition_id`` for the
    pooled object; all original partition identities/provenance remain recorded.
    """
    nodes = list(nodes)
    if len(nodes) < 2:
        raise ValueError('exact merge requires at least two StatisticalNode inputs')
    if not isinstance(partition_id, str) or not partition_id.strip():
        raise ValueError('partition_id must be a nonempty stable string')

    merged_axes = _validate_exact_merge_compatibility(nodes)

    contributing_ids: list[str] = []
    contributing_provenance: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in nodes:
        ids = _partition_ids_for_merge(node)
        overlap = seen_ids.intersection(ids)
        if overlap:
            raise ValueError(f'duplicate partition_id in exact merge: {sorted(overlap)!r}')
        seen_ids.update(ids)
        contributing_ids.extend(ids)
        contributing_provenance.extend(_contributing_provenance_for_merge(node))
    if partition_id in seen_ids:
        raise ValueError('output partition_id must differ from contributing partition_id values')

    first = nodes[0]
    ncoord = len(first.dimensions)
    accumulated: dict[tuple[Any, ...], list[Any]] = {}
    for node in nodes:
        for row in node.state.itertuples(index=False, name=None):
            key = tuple(row[:ncoord])
            values = row[ncoord:]
            slot = accumulated.get(key)
            if slot is None:
                slot = [0] + [0.0] * (len(first.state_schema) - 1)
                accumulated[key] = slot
            slot[0] += int(values[0])
            for i, value in enumerate(values[1:], start=1):
                slot[i] += float(value)

    # Deterministic sparse-row order follows physical bin then declared
    # categorical-domain order, exactly as the builders do.
    lookup = {
        dim: {json.dumps(v, sort_keys=True): i
              for i, v in enumerate(merged_axes[dim]['values'])}
        for dim in first.dimensions[1:]
    }
    def _sort_key(key):
        return (int(key[0]),) + tuple(
            lookup[dim][json.dumps(key[i + 1], sort_keys=True)]
            for i, dim in enumerate(first.dimensions[1:]))

    rows = [list(key) + accumulated[key] for key in sorted(accumulated, key=_sort_key)]
    frame = _typed_frame(rows, first.dimensions, first.state_schema)
    provenance = {
        'partition_id': partition_id,
        'contributing_partition_ids': contributing_ids,
        'contributing_provenance': contributing_provenance,
    }
    return StatisticalNode(
        schema_version=first.schema_version,
        name=first.name,
        statistic_kind=first.statistic_kind,
        dimensions=copy.deepcopy(first.dimensions),
        axes=merged_axes,
        semantic_roles=copy.deepcopy(first.semantic_roles),
        state_schema=copy.deepcopy(first.state_schema),
        state=frame,
        semantic_description=copy.deepcopy(first.semantic_description),
        metadata=copy.deepcopy(first.metadata),
        provenance=provenance,
    )

def _build(kind, x, *, edges, name, partition_id, y=None, weights=None,
           categories=None, category_sources=None, x_source='x', y_source='y',
           weights_source=None, semantic_description=None, semantic_roles=None,
           metadata=None, provenance=None):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('name must be a nonempty durable string')
    if not isinstance(partition_id, str) or not partition_id.strip():
        raise ValueError('partition_id must be a nonempty stable string')
    resolved_edges = _validate_edges(edges)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError('x must be one-dimensional')
    n = len(x_arr)
    w_arr = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w_arr.ndim != 1 or len(w_arr) != n:
        raise ValueError('weights length must equal x length')
    if kind == 'profile':
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1 or len(y_arr) != n:
            raise ValueError('profile y length must equal x length')
    else:
        y_arr = None
    categories = {} if categories is None else dict(categories)
    category_sources = {} if category_sources is None else dict(category_sources)
    if ('x_bin' in categories or any(not isinstance(k, str) or not k or k in _COMMON_STATE
                                     or k in _PROFILE_EXTRA for k in categories)):
        raise ValueError('categorical coordinate names collide with state fields')
    dimensions = ['x_bin'] + list(categories)
    cat_arrays = {}
    for dim, values in categories.items():
        a = np.asarray(values, dtype=object)
        if a.ndim != 1 or len(a) != n:
            raise ValueError(f'coordinate {dim}: input length must equal x length')
        cat_arrays[dim] = a
    edge_arr = np.asarray(resolved_edges)
    accepted = np.isfinite(x_arr) & np.isfinite(w_arr)
    if y_arr is not None:
        accepted &= np.isfinite(y_arr)
    # Use NumPy bin convention: [low, high) except the final upper edge.
    binidx = np.searchsorted(edge_arr, x_arr, side='right') - 1
    binidx[x_arr == edge_arr[-1]] = len(edge_arr) - 2
    accepted &= (x_arr >= edge_arr[0]) & (x_arr <= edge_arr[-1])
    for dim, arr in cat_arrays.items():
        accepted &= ~pd.isna(arr)
    axes = {
        'x_bin': {
            'kind': 'binned', 'source': x_source, 'edges': resolved_edges,
            'row_membership': 'partition', 'underflow':'exclude',
            'overflow':'exclude', 'nan_policy':'omit',
        }
    }
    for dim, arr in cat_arrays.items():
        # Category order determined from accepted rows, never from layout position.
        values = [v.item() if isinstance(v, np.generic) else v
                  for v in pd.unique(arr[accepted])]
        if len({type(v) for v in values}) > 1:
            raise ValueError(f'coordinate {dim}: mixed category types are unsupported')
        axes[dim] = {'kind':'categorical','source':category_sources.get(dim,dim),
                     'values':_json_copy(values, f'category {dim}'),
                     'row_membership':'partition'}
    schema = dict(_COMMON_STATE)
    if kind == 'profile':
        schema.update(_PROFILE_EXTRA)
    accumulated: dict[tuple, list] = {}
    for i in np.nonzero(accepted)[0]:
        key = (int(binidx[i]),) + tuple(cat_arrays[dim][i] for dim in categories)
        if key not in accumulated:
            accumulated[key] = [0, 0., 0.] + ([0., 0.] if kind == 'profile' else [])
        slot = accumulated[key]
        wi = float(w_arr[i])
        slot[0] += 1
        slot[1] += wi
        slot[2] += wi * wi
        if kind == 'profile':
            yi = float(y_arr[i])
            slot[3] += wi * yi
            slot[4] += wi * yi * yi
    # Sort by explicit coordinate-domain ordinal, not lexical category value.
    lookup = {dim: {v:i for i,v in enumerate(axes[dim]['values'])} for dim in categories}
    keys = sorted(accumulated, key=lambda key: (key[0],) + tuple(
        lookup[dim][key[j+1]] for j, dim in enumerate(categories)))
    rows = [list(key) + accumulated[key] for key in keys]
    frame = _typed_frame(rows, dimensions, schema)
    prov = {} if provenance is None else dict(provenance)
    if 'partition_id' in prov and prov['partition_id'] != partition_id:
        raise ValueError('partition_id in provenance conflicts with explicit argument')
    prov['partition_id'] = partition_id
    default_desc = {'expression': f'{y_source}:{x_source}' if y_arr is not None else x_source,
                    'type': 'profile' if kind == 'profile' else 'hist',
                    'weights': weights_source}
    return StatisticalNode(
        schema_version=SCHEMA_VERSION, name=name, statistic_kind=kind,
        dimensions=dimensions, axes=axes,
        semantic_roles={} if semantic_roles is None else copy.deepcopy(dict(semantic_roles)),
        state_schema=schema, state=frame,
        semantic_description=default_desc if semantic_description is None
            else copy.deepcopy(dict(semantic_description)),
        metadata={} if metadata is None else copy.deepcopy(dict(metadata)),
        provenance=prov)


def build_histogram_state(x, *, edges, name, partition_id, weights=None,
                          categories=None, category_sources=None, x_source='x',
                          weights_source=None, semantic_description=None,
                          semantic_roles=None, metadata=None, provenance=None):
    """Accumulate retained weighted histogram state without rendering."""
    return _build('histogram', x, edges=edges, name=name, partition_id=partition_id,
                  weights=weights, categories=categories, category_sources=category_sources,
                  x_source=x_source, weights_source=weights_source,
                  semantic_description=semantic_description, semantic_roles=semantic_roles,
                  metadata=metadata, provenance=provenance)


def build_profile_state(x, y, *, edges, name, partition_id, weights=None,
                        categories=None, category_sources=None, x_source='x',
                        y_source='y', weights_source=None, semantic_description=None,
                        semantic_roles=None, metadata=None, provenance=None):
    """Accumulate sufficient weighted profile moments without rendering."""
    return _build('profile', x, y=y, edges=edges, name=name, partition_id=partition_id,
                  weights=weights, categories=categories, category_sources=category_sources,
                  x_source=x_source, y_source=y_source, weights_source=weights_source,
                  semantic_description=semantic_description, semantic_roles=semantic_roles,
                  metadata=metadata, provenance=provenance)


__all__ = ['StatisticalNode', 'build_histogram_state', 'build_profile_state', 'merge_exact']
