"""
Shared functions among the tests like setting up test environment
"""
from pathlib import Path

from dataframe_sql import register_temp_table, remove_temp_table
from dataframe_sql.framework_utils import read_csv_type

DATA_PATH = Path(__file__).parent.parent / "data"

DATA_FILES = {
    "FOREST_FIRES": "forestfires.csv",
    "DIGIMON_MON_LIST": "DigiDB_digimonlist.csv",
    "DIGIMON_MOVE_LIST": "DigiDB_movelist.csv",
    "DIGIMON_SUPPORT_LIST": "DigiDB_supportlist.csv",
    "AVOCADO": "avocado.csv",
}


def yield_test_dataframes(backend: str):
    for datafile in DATA_FILES:
        dataframe = read_csv_type[backend](DATA_PATH / DATA_FILES[datafile])

        # Name change is for name interference
        if datafile == "DIGIMON_MON_LIST":
            dataframe["mon_attribute"] = dataframe["Attribute"]
        if datafile == "DIGIMON_MOVE_LIST":
            dataframe["move_attribute"] = dataframe["Attribute"]

        yield datafile, dataframe


def register_env_tables(backend: str):
    """
    Returns all globals but in lower case
    :return:
    """
    for name, dataframe in yield_test_dataframes(backend):
        register_temp_table(frame=dataframe, table_name=name)


def remove_env_tables():
    """
    Remove all env tables
    :return:
    """
    for table_name in DATA_FILES:
        remove_temp_table(table_name=table_name)
