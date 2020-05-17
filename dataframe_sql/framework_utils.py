import pandas
import dask.dataframe

PANDAS = "pandas"
DASK = "dask"
OPTIONS = {"backend": PANDAS}

dataframe_type = {PANDAS: pandas.DataFrame, DASK: dask.dataframe.DataFrame}
read_csv_type = {PANDAS: pandas.read_csv, DASK: dask.dataframe.read_csv}
