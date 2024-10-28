import functools
import operator

import numpy as np
import pyarrow as pa

from .declarations import (
    declare_table,
    declare_dataset,
    declare_join,
    declare_join_asof,
    declare_project,
    declare_filter,
    declare_order_by,
    declare_aggregate,
)


class DataPlan:
    """
    A plan to process data
    """

    def __init__(self, execplan, source=None):
        """
        Construct a DataPlan from an exec plan
        """
        self._execplan = execplan
        if isinstance(source, list):
            self._source = source
        else:
            self._source = [source]

    def __repr__(self):
        execplan_repr = str(self.execplan)
        if self.source is not None:
            source_repr = "\n  ".join([
                f"{type(source)} at {hex(id(source))}"
                for source in self.source
            ])
        else:
            source_repr = None
        return (
            f"DataPlan\n"
            f"--------------------------------------------------------------------------------\n"
            f"{execplan_repr}\n"
            f"--------------------------------------------------------------------------------\n"
            f"Sources:\n  {source_repr}\n"
            f"--------------------------------------------------------------------------------"
        )

    @property
    def execplan(self):
        """
        The current plan to be executed.
        """
        return self._execplan

    @classmethod
    def from_dict(cls, mapping, **kwargs):
        """
        Construct a DataPlan from a dictionary.
        """
        table = pa.Table.from_pydict(mapping, **kwargs)
        _plan = declare_table(table)
        return cls(_plan, source=mapping)

    @classmethod
    def from_list(cls, mapping, **kwargs):
        """
        Construct a DataPlan from a list.
        """
        table = pa.Table.from_pylist(mapping, **kwargs)
        _plan = declare_table(table)
        return cls(_plan, source=mapping)

    @classmethod
    def from_dataframe(cls, df, **kwrags):
        """
        Construct a DataPlan from a pandas DataFrame.
        """
        table = pa.Table.from_pandas(df, **kwargs)
        _plan = declare_table(table)
        return cls(_plan, source=df)

    @classmethod
    def from_table(cls, table):
        """
        Construct a DataPlan from a pyarrow Table.
        """
        _plan = declare_table(table)
        return cls(_plan, source=table)

    @classmethod
    def from_dataset(cls, dataset, columns=None, filter=None):
        """
        Construct a DataPlan from a pyarrow Dataset.
        """
        _plan = declare_dataset(dataset, columns=columns, filter=filter)
        return cls(_plan, source=dataset)

    @property
    def source(self):
        """
        The source over which the plan will operate.
        """
        return self._source

    def project(self, *args, **kwargs):
        """
        Apply a projection (e.g., select columns, rename columns, construct new
        columns, etc.).
        """
        _project = declare_project(self.execplan, *args, **kwargs)
        return self.__class__(_project, source=self.source)

    def filter(self, *args, **kwargs):
        """
        Apply a filter.
        """
        _filter = declare_filter(self.execplan, *args, **kwargs)
        return self.__class__(_filter, source=self.source)

    def join(self, other, *args, **kwargs):
        """
        Perform a join with another DataPlan.
        """
        _join = declare_join(self.execplan, other.execplan, *args, **kwargs)
        _source = [self.source, other.source]
        source = list(
            functools.reduce(
                operator.concat,
                _source,
                [],
            ),
        )
        return self.__class__(_join, source=source)

    def join_asof(self, other, *args, **kwargs):
        """
        Perform an asof join with another DataPlan.
        """
        _join = declare_join_asof(self.execplan, other.execplan, *args, **kwargs)
        _source = [self.source, other.source]
        source = list(
            functools.reduce(
                operator.concat,
                _source,
                [],
            ),
        )
        return self.__class__(_join, source=source)

    def order_by(self, *args, **kwargs):
        """
        Apply an ordering.
        """
        _order_by = declare_order_by(self.execplan, *args, **kwargs)
        return self.__class__(_order_by, source=self.source)

    def aggregate(self, *args, **kwargs):
        """
        Perform a series of aggregations.
        """
        _aggregate = declare_aggregate(self.execplan, *args, **kwargs)
        return self.__class__(_aggregate, source=self.source)

    def _to_reader(self, use_threads=True):
        """
        Execute the plan lazily as a reader.
        """
        return self.execplan.to_reader(use_threads=use_threads)

    def _to_table(self, use_threads=True):
        """
        Execute the plan and return a pyarrow Table.
        """
        return self.execplan.to_table(use_threads=use_threads)

    def to_reader(self, use_threads=True):
        """
        Execute the plan lazily as a reader.
        """
        return self._to_reader(use_threads=use_threads)

    def to_table(self, use_threads=True):
        """
        Execute the plan and return a pyarrow Table.
        """
        return self._to_table(use_threads=use_threads)

    def to_batches(self, use_threads=True):
        """
        Execute the plan and return a list of pyarrow RecordBatches.
        """
        table = self._to_table(use_threads=use_threads)
        return table.to_batches()

    def to_dataframe(self, use_threads=True):
        """
        Execute the plan and return a pandas DataFrame.
        """
        table = self._to_table(use_threads=use_threads)
        return table.to_pandas()

    def to_dict(self, use_threads=True):
        """
        Execute the plan and return a dictionary of columns.
        """
        table = self._to_table(use_threads=use_threads)
        return table.to_pydict()

    def to_list(self, use_threads=True):
        """
        Execute the plan and return a list of rows.
        """
        table = self._to_table(use_threads=use_threads)
        return table.to_pylist()

    def to_records(self, use_threads=True):
        """
        Execute the plan and return a numpy record array.
        """
        table = self._to_table(use_threads=use_threads)
        return np.rec.fromarrays(table.columns, names=table.schema.names)

    def to_array(self, use_threads=True):
        """
        Execute the plan and return a numpy array.
        """
        table = self._to_table(use_threads=use_threads)
        data = [col.to_numpy() for col in table.itercolumns()]
        dtype = []
        for field in table.schema:
            try:
                np_dtype = field.type.to_pandas_dtype()
            except NotImplementedError:
                # Fall back to generic python object dtypes when necessary
                np_dtype = np.object_
            _dtype = (field.name, np_dtype)
            dtype.append(_dtype)
        return np.array(list(zip(*data)), dtype=dtype)

    def group_by(self, keys):
        return DataPlanGroupBy(self, keys)


class DataPlanGroupBy:
    def __init__(self, dataplan, keys):
        if isinstance(keys, str):
            keys = [keys]

        self._dataplan = dataplan
        self.keys = keys

    def __repr__(self):
        dataplan_repr = f"{type(self.dataplan)} at {hex(id(self.dataplan))}"
        if isinstance(self.keys, list):
            key_repr = "\n  ".join(self.keys)
        else:
            key_repr = self.keys
        return (
            f"DataPlanGroupBy\n"
            f"--------------------------------------------------------------------------------\n"
            f"{dataplan_repr}\n"
            f"--------------------------------------------------------------------------------\n"
            f"keys:\n  {key_repr}\n"
            f"--------------------------------------------------------------------------------\n"
        )

    @property
    def dataplan(self):
        """
        The dataplan to group.
        """
        return self._dataplan

    def aggregate(self, aggregations):
        """
        Perform a series of aggregations over the groups.
        """
        return self.dataplan.aggregate(
            aggregations,
            keys=self.keys,
        )
