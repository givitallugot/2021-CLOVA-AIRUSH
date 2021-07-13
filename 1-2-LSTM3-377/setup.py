#nsml: tensorflow/tensorflow:2.2.0
from distutils.core import setup

setup(
    name='airush-2021-pubtrans',
    version='1.0',
    install_requires=[
        'pandas',
        'pyarrow',
        'tensorflow >= 2.2.0',
        'haversine'
    ],
    python_requires='>=3.6',
)