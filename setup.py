import os
from setuptools import setup, find_packages
from setuptools.command.install import install


# Define a custom installation class that extends setuptools' install command
class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        # Run your custom code here
        download_files()


# Function to download the necessary files
def download_files():
    # Get the directory of setup.py
    setup_dir = os.path.dirname(os.path.abspath(__file__))

    # Create directory if it does not exist
    model_dir = os.path.join(setup_dir, "pre_trained_models", "vggish")
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading VGGish files...{model_dir}")
    # Download vggish_model.ckpt to pre_trained_models/vggish/
    os.system(
        f"curl -o {os.path.join(model_dir, 'vggish_model.ckpt')} https://storage.googleapis.com/audioset/vggish_model.ckpt")
    # Download vggish_pca_params.npz to pre_trained_models/vggish/
    os.system(
        f"curl -o {os.path.join(model_dir, 'vggish_pca_params.npz')} https://storage.googleapis.com/audioset/vggish_pca_params.npz")


# Setup function with custom install command
setup(
    name='pyHopperVGG',
    version='0.0.1',
    author="francisco-rai",
    author_email="francisco.mendes.pv@renesas.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
            "pyHopperVGG = pyHopperVGG.vggish_smoke_test:main",
        ]
    },
    cmdclass={
        'install': CustomInstallCommand,
    },
)
