from setuptools import setup, find_packages

setup(
    name='pyHopperVGG',
    version='0.0.1',
    author="francisco-rai",
    author_email="francisco.mendes.pv@renesas.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={'pyHopperVGG.pre_trained_models': ['*']
                  },
    include_package_data=True,
    install_requires=[
        'numpy==1.24.3',
        'resampy',
        'tensorflow==2.13.0',
        'tf_slim',
        'six',
        'soundfile',
        'pandas==1.5.3'
    ],
    entry_points={
        "console_scripts": [
            "pyHopperVGG = pyHopper.inference:main",
        ]
    }
)
