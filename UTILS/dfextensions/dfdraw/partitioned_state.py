"""S8 experimental partition-preserving statistical state (no collapse).

A partition is a user-defined scientific/computational source identity, not a
run/file/job by default.  Each original StatisticalNode retains its complete
sufficient state and source-level provenance.  The combined coordinate view
is deterministic; explicit analyst-defined category order takes precedence.

This module is additive: StatisticalNode schema v1 and merge_exact are not
replaced.  S9 will provide a separately reviewed, explicit collapse operation.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

import pandas as pd

from .statistical_state import (
    StatisticalNode, _canonical_category_sort_key, _category_key,
    _json_copy, _validate_node,
)

PARTITIONED_SCHEMA_VERSION = 1


def _canonical_axes(nodes: Sequence[StatisticalNode]) -> dict[str, Any]:
    first = nodes[0]
    if 'partition_id' in first.dimensions:
        raise ValueError('S8 partition_id is a reserved partition coordinate')
    for node in nodes:
        _validate_node(node)
        if 'contributing_partition_ids' in node.provenance:
            raise ValueError('S8 requires unpooled leaf partitions; collapsed contributor state cannot be recovered')
        for field in ('schema_version', 'name', 'statistic_kind', 'dimensions',
                      'state_schema', 'semantic_roles', 'semantic_description', 'metadata'):
            if getattr(node, field) != getattr(first, field):
                raise ValueError(f'S8 incompatible {field} between partitions')
        if node.axes['x_bin'] != first.axes['x_bin']:
            raise ValueError('S8 incompatible physical x_bin axes; no rebinning')

    merged = {'x_bin': copy.deepcopy(first.axes['x_bin'])}
    for dim in first.dimensions[1:]:
        reference = {k: v for k, v in first.axes[dim].items()
                     if k not in ('values', 'ordered', 'order_origin')}
        domains = []
        explicit = []
        for node in nodes:
            axis = node.axes[dim]
            base = {k: v for k, v in axis.items()
                    if k not in ('values', 'ordered', 'order_origin')}
            if base != reference:
                raise ValueError(f'S8 incompatible categorical axis semantics: {dim}')
            values = axis['values']
            domains.append(values)
            if axis.get('ordered') is True:
                if axis.get('order_origin') != 'explicit':
                    raise ValueError(f'S8 {dim}: ordered axis requires explicit order provenance')
                explicit.append(values)
            elif axis.get('ordered') or axis.get('order_origin') == 'explicit':
                raise ValueError(f'S8 {dim}: inconsistent categorical order declaration')

        all_values = {}
        for values in domains:
            for value in values:
                all_values[_category_key(value)] = value
        # Do not interpret two differently typed values with Python equality.
        if len({type(v) for v in all_values.values()}) > 1:
            raise ValueError(f'S8 coordinate {dim}: incompatible categorical scalar types')
        # Canonical unordered ordering never depends on input partition order.
        if explicit:
            declared = explicit[0]
            if any([_category_key(v) for v in other] !=
                   [_category_key(v) for v in declared] for other in explicit[1:]):
                raise ValueError(f'S8 coordinate {dim}: conflicting explicit scientific orders')
            if set(all_values) != {_category_key(v) for v in declared}:
                raise ValueError(f'S8 coordinate {dim}: labels outside declared scientific order')
            values = copy.deepcopy(declared)
            origin = 'explicit'
        else:
            values = sorted(all_values.values(), key=_canonical_category_sort_key)
            origin = 'canonical'
        merged[dim] = copy.deepcopy(reference)
        merged[dim]['values'] = values
        merged[dim]['ordered'] = origin == 'explicit'
        merged[dim]['order_origin'] = origin
    return merged


def _axes_identity(axes: Mapping[str, Any]) -> str:
    """Type-sensitive identity text of an axes mapping.

    Python equality treats ``False == 0``, ``True == 1`` and ``0.0 == 0``,
    so two axes mappings that differ in JSON scalar type compare equal.
    Category labels are normalised exactly as the builders do (``-0.0`` ->
    ``0.0``, NumPy scalars unwrapped, via ``_category_key``); every other
    entry is compared by its JSON text, which keeps the type.
    """
    normalised = copy.deepcopy(dict(axes))
    for axis in normalised.values():
        if (isinstance(axis, dict) and axis.get('kind') == 'categorical'
                and isinstance(axis.get('values'), list)):
            axis['values'] = [_category_key(v) for v in axis['values']]
    return json.dumps(normalised, sort_keys=True, allow_nan=False)


class PartitionedStatisticalState:
    """Immutable-by-access composition of independent unpooled state leaves.

    `state` is only a computed table with partition_id as first coordinate;
    the source of truth is the distinct StatisticalNodes.  Every read returns
    copies to avoid accidental corruption of provenance or retained moments.
    """

    def __init__(self, nodes: Sequence[StatisticalNode]):
        items = list(nodes)
        if len(items) < 2:
            raise ValueError('S8 preserving partitions requires at least two independent leaves')
        if any(not isinstance(node, StatisticalNode) for node in items):
            raise TypeError('S8 partitioned state accepts StatisticalNode leaves only')
        ids = [node.provenance['partition_id'] for node in items]
        if len(set(ids)) != len(ids):
            raise ValueError('S8 duplicate source partition_id in preserved state')
        axes = _canonical_axes(items)  # refuse before copying or exposing state
        self._nodes = tuple(copy.deepcopy(node) for node in sorted(
            items, key=lambda node: node.provenance['partition_id']))
        self._axes = axes

    @property
    def name(self):
        return self._nodes[0].name

    @property
    def statistic_kind(self):
        return self._nodes[0].statistic_kind

    @property
    def dimensions(self):
        return ['partition_id'] + list(self._nodes[0].dimensions)

    @property
    def axes(self):
        return {'partition_id': {'kind': 'categorical', 'source': 'partition_id',
                                  'values': list(self.partition_ids),
                                  'row_membership': 'partition',
                                  'ordered': False, 'order_origin': 'canonical'},
                **copy.deepcopy(self._axes)}

    @property
    def state_schema(self):
        return copy.deepcopy(self._nodes[0].state_schema)

    @property
    def partition_ids(self):
        return tuple(node.provenance['partition_id'] for node in self._nodes)

    @property
    def partitions(self):
        return tuple(copy.deepcopy(node) for node in self._nodes)

    def get_partition(self, partition_id: str) -> StatisticalNode:
        for node in self._nodes:
            if node.provenance['partition_id'] == partition_id:
                return copy.deepcopy(node)
        raise KeyError(partition_id)

    @property
    def state(self) -> pd.DataFrame:
        """Sparse coordinate view, sorted by partition, x-bin and category ordinals."""
        first = self._nodes[0]
        dimensions = list(first.dimensions)
        cols = ['partition_id'] + dimensions + list(first.state_schema)
        row_tuples = []
        for node in self._nodes:
            for row in node.state.itertuples(index=False, name=None):
                row_tuples.append((node.provenance['partition_id'], *row))
        ordinals = {dim: {_category_key(v): i for i, v in enumerate(self._axes[dim]['values'])}
                    for dim in dimensions[1:]}
        def sort_key(row):
            return (row[0], int(row[1])) + tuple(
                ordinals[dim][_category_key(row[i + 2])]
                for i, dim in enumerate(dimensions[1:]))
        return pd.DataFrame(sorted(row_tuples, key=sort_key), columns=cols)

    def to_payload(self) -> dict[str, Any]:
        """Experimental logical payload; distinct from StatisticalNode schema v1."""
        payload = {
            'partitioned_schema_version': PARTITIONED_SCHEMA_VERSION,
            'kind': 'partitioned_statistical_state',
            'axes': self.axes,
            'partitions': [node.to_payload() for node in self._nodes],
        }
        return _json_copy(payload, 'S8 partition-preserving payload')

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> 'PartitionedStatisticalState':
        """Rebuild a preserved state from its payload (the output of ``to_payload``).

        The state is rebuilt from the stored leaf payloads; the stored
        top-level ``axes`` must then describe exactly the rebuilt state,
        compared type-sensitively.  Category labels follow the builders'
        normalisation (``-0.0`` reads as ``0.0``); any other difference in
        value or JSON type (``False`` vs ``0``, ``0.0`` vs ``0``, ``1`` vs
        ``True``) is refused.

        Raises
        ------
        ValueError
            Invalid schema or kind, an invalid leaf payload, or stored axes
            inconsistent with the source partitions.
        """
        if (not isinstance(payload, Mapping) or
                set(payload) != {'partitioned_schema_version', 'kind', 'axes', 'partitions'} or
                payload.get('partitioned_schema_version') != PARTITIONED_SCHEMA_VERSION or
                payload.get('kind') != 'partitioned_statistical_state' or
                not isinstance(payload.get('partitions'), list)):
            raise ValueError('invalid S8 partition-preserving payload/schema')
        safe = _json_copy(dict(payload), 'S8 partition-preserving payload')
        nodes = [StatisticalNode.from_payload(entry) for entry in safe['partitions']]
        obj = cls(nodes)
        # Compare by JSON type as well as value: Python equality would accept
        # False for 0, True for 1 or 0.0 for 0 (PHASE_13_82_DF S9 slice 1).
        if _axes_identity(safe['axes']) != _axes_identity(obj.axes):
            raise ValueError('S8 payload axes inconsistent with source partitions')
        return obj


def preserve_partitions(inputs: Sequence[StatisticalNode | PartitionedStatisticalState]
                        ) -> PartitionedStatisticalState:
    """Preserve source statistics without pooling their moments or identities.

    Flatten nested preserved states. Reject already pooled/merged source nodes;
    `merge_exact` is information reducing and its original leaves cannot be
    reconstructed from the pooled sufficient statistics.
    """
    leaves = []
    for item in inputs:
        if isinstance(item, PartitionedStatisticalState):
            leaves.extend(item.partitions)
        elif isinstance(item, StatisticalNode):
            leaves.append(item)
        else:
            raise TypeError('S8 accepts StatisticalNode or PartitionedStatisticalState')
    return PartitionedStatisticalState(leaves)


__all__ = ['PartitionedStatisticalState', 'preserve_partitions',
           'PARTITIONED_SCHEMA_VERSION']
