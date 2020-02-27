from distutils.core import setup
import os
from pathlib import Path

from versioneer import get_cmdclass, get_version

CODE_DIRECTORY = Path(__file__).parent


def read_file(filename):
    """Source the contents of a file"""
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()


setup(
    name="dataframe_sql",
    version=get_version(),
    cmdclass=get_cmdclass(),
    long_description=read_file("README.md"),
    maintainer="Zach Brookler",
    maintainer_email="zachb1996@yahoo.com",
    description="A package for querying dataframes using SQL",
    python_requires=">=3.6.1",
    install_requires=["lark-parser == 0.8.1", "pandas == 1.0.1"],
    project_urls={
        "Source": "https://github.com/zbrookle/dataframe_sql",
        "Documentation": "",
    },
    url="https://github.com/zbrookle/dataframe_sql",
    download_url="https://github.com/zbrookle/dataframe_sql/archive/master.zip",
    keywords=["pandas", "data", "dataframe", "sql"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
