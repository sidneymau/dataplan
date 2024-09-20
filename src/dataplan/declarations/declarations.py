# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import ast
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import acero


# https://github.com/apache/arrow/blob/main/python/pyarrow/_dataset.pyx#L3395-L3440
def _parse_columns(columns):
    expressions = []
    names = None
    if columns is not None:
        if isinstance(columns, dict):
            names = []
            for name, expr in columns.items():
                if not isinstance(expr, pc.Expression):
                    raise TypeError(
                        "Expected an Expression for a 'column' dictionary "
                        "value, got {} instead".format(type(expr))
                    )
                expressions.append(expr)
                names.append(name)

        elif isinstance(columns, list):
            for column in columns:
                expr = pc.field(column)
                expressions.append(expr)
        else:
            raise ValueError(
                "Expected a list or a dict for 'columns', "
                "got {} instead.".format(type(columns))
            )

    return expressions, names


# https://github.com/apache/arrow/blob/main/python/pyarrow/parquet/core.py#L135-L199
def _parse_filters(filters):
    if isinstance(filters, pc.Expression):
        return filters

    def _parse_token(token):
        try:
            val = ast.literal_eval(token)
        except:
            val = token
        return val

    # see pq.core._DNF_filter_doc
    _operators = ["==", "=", "!=", "<=", ">=", "<", ">", "in", "not in"]
    _pattern = "(" + "|".join(_operators) + ")"
    pattern = re.compile(_pattern)
    disjunction = []
    for _conjunction in filters:
        conjunction = []
        for filters_string in _conjunction:
            filters_split = pattern.split(filters_string)
            filters_tuple = tuple(
                map(
                    _parse_token,
                    map(
                        str.strip,
                        filters_split,
                    ),
                ),
            )
            conjunction.append(filters_tuple)
        disjunction.append(conjunction)

    return pq.filters_to_expression(disjunction)


def declare_table(table):
    decl = acero.Declaration(
        "table_source",
        acero.TableSourceNodeOptions(
            table,
        ),
    )
    return decl


# https://github.com/apache/arrow/blob/main/python/pyarrow/acero.py#L59-L77
def declare_dataset(dataset, columns=None, filter=None):
    if columns is None:
        columns = dataset.schema.names

    filter_expression = None
    if filter is not None:
        # filter_expression = pq.filters_to_expression(filter)
        filter_expression = _parse_filters(filter)

    decl = acero.Declaration(
        "scan",
        acero.ScanNodeOptions(
            dataset,
            columns=columns,
            filter=filter_expression,
        ),
    )

    # While the dataset scan can apply pushdown projections and filters to
    # minimize which data are read, it does not construct the associated nodes
    # that perform these operations. Therefore, we explicitly do so here.
    if filter_expression is not None:
        decl = declare_filter(decl, filter_expression)

    decl = declare_project(decl, columns)

    return decl


# https://github.com/apache/arrow/blob/main/python/pyarrow/acero.py#L80-L254
def declare_join(
    left_source,
    right_source,
    join_type,
    left_keys,
    right_keys,
    left_output=None,
    right_output=None,
    left_suffix=None,
    right_suffix=None,
):
    join_opts = acero.HashJoinNodeOptions(
        join_type,
        left_keys,
        right_keys,
        left_output=left_output,
        right_output=right_output,
        output_suffix_for_left=left_suffix or "",
        output_suffix_for_right=right_suffix or "",
    )
    join_node = acero.Declaration(
        "hashjoin", options=join_opts, inputs=[left_source, right_source],
    )

    return join_node


# https://github.com/apache/arrow/blob/main/python/pyarrow/acero.py#L257-L340
def declare_join_asof(
    left_source,
    right_source,
    left_on,
    left_by,
    right_on,
    right_by,
    tolerance,
):
    if not isinstance(left_by, (tuple, list)):
        left_by = [left_by]
    if not isinstance(right_by, (tuple, list)):
        right_by = [right_by]

    join_opts = acero.AsofJoinNodeOptions(
        left_on, left_by, right_on, right_by, tolerance,
    )
    join_node = acero.Declaration(
        "asofjoin", options=join_opts, inputs=[left_source, right_source],
    )

    return join_node


def declare_project(source, columns):
    expressions, names = _parse_columns(columns)

    project_node = acero.Declaration(
        "project",
        acero.ProjectNodeOptions(expressions, names=names),
    )
    decl = acero.Declaration.from_sequence(
        [
            source,
            project_node,
        ],
    )

    return decl


def declare_filter(source, filters):
    filter_expression = None
    if filters is not None:
        # filter_expression = pq.filters_to_expression(filters)
        filter_expression = _parse_filters(filters)

    filter_node = acero.Declaration(
        "filter",
        acero.FilterNodeOptions(filter_expression),
    )
    decl = acero.Declaration.from_sequence(
        [
            source,
            filter_node,
        ],
    )

    return decl


def declare_order_by(source, sort_keys, **kwargs):

    order_by = acero.Declaration(
        "order_by",
        acero.OrderByNodeOptions(sort_keys, **kwargs),
    )

    decl = acero.Declaration.from_sequence([source, order_by])

    return decl


# https://github.com/apache/arrow/blob/main/python/pyarrow/table.pxi#L6396-L6511
def declare_aggregate(source, aggregations, keys=None):

    if keys is None:
        keys = []

    group_by_aggrs = []
    for aggr in aggregations:
        # Set opt to None if not specified
        if len(aggr) == 2:
            target, func = aggr
            opt = None
        else:
            target, func, opt = aggr
        # Ensure target is a list
        if not isinstance(target, (list, tuple)):
            target = [target]
        # Ensure aggregate function is hash_ if needed
        if len(keys) > 0 and not func.startswith("hash_"):
            func = "hash_" + func
        if len(keys) == 0 and func.startswith("hash_"):
            func = func[5:]
        # Determine output field name
        func_nohash = func if not func.startswith("hash_") else func[5:]
        if len(target) == 0:
            aggr_name = func_nohash
        else:
            aggr_name = "_".join(target) + "_" + func_nohash
        group_by_aggrs.append((target, func, opt, aggr_name))

    aggregate_node = acero.Declaration(
        "aggregate",
        acero.AggregateNodeOptions(group_by_aggrs, keys=keys),
    )

    decl = acero.Declaration.from_sequence([source, aggregate_node])

    return decl
