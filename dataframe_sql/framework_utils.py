from typing import TypeVar

import dask.dataframe
from dask.dataframe import DataFrame as DaskDataFrame, Series as DaskSeries
import pandas
from pandas import DataFrame as PandasDataFrame, Series as PandasSeries

PANDAS = "pandas"
DASK = "dask"
OPTIONS = {"backend": PANDAS}

dataframe_type = {PANDAS: pandas.DataFrame, DASK: dask.dataframe.DataFrame}
read_csv_type = {PANDAS: pandas.read_csv, DASK: dask.dataframe.read_csv}
series_type = {PANDAS: pandas.Series, DASK: dask.dataframe.Series}


SeriesType = TypeVar('SeriesType', DaskSeries, PandasSeries)
DataFrameType = TypeVar('DataFrameType', PandasDataFrame, DaskDataFrame)
