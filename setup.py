from setuptools import setup, find_packages


setup(
    name='radiocarbon',
    version='0.5.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'radiocarbon': ['calcurves/*.14c']
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'
    ],
)
