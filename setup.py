import os
from setuptools import setup, find_packages
from setuptools.command.install import install


# Define a custom installation class that extends setuptools' install command
class CustomInstallCommand(install):
    def run(self):
        # Run your custom code here
        install.run(self)
        download_files()

# Function to download the necessary files
# def download_files():
#
#
#     # Get the directory of setup.py
#     setup_dir = os.path.dirname(os.path.abspath(__file__))
#
#     # Create directory if it does not exist
#     model_dir = os.path.join(setup_dir, "pre_trained_models", "vggish")
#     os.makedirs(model_dir, exist_ok=True)
#
#     print(f"Downloading VGGish files...{model_dir}")
#     # Download vggish_model.ckpt to pre_trained_models/vggish/
#     os.system(
#         f"curl -o {os.path.join(model_dir, 'vggish_model.ckpt')} https://storage.googleapis.com/audioset/vggish_model.ckpt")
#     # Download vggish_pca_params.npz to pre_trained_models/vggish/
#     os.system(
#         f"curl -o {os.path.join(model_dir, 'vggish_pca_params.npz')} https://storage.googleapis.com/audioset/vggish_pca_params.npz")

def download_files():
    import os
    import requests

    # Get the directory where the current Python script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the URLs for the files to download
    url_vggish_model = 'https://storage.googleapis.com/audioset/vggish_model.ckpt'
    url_vggish_pca_params = 'https://storage.googleapis.com/audioset/vggish_pca_params.npz'

    # Define the local directory paths relative to the script directory
    local_dir_vggish = os.path.join(script_dir, 'pre_trained_models', 'vggish')

    # Ensure the directory exists, create it if necessary
    os.makedirs(local_dir_vggish, exist_ok=True)

    # Download vggish_model.ckpt
    local_path_vggish_model = os.path.join(local_dir_vggish, 'vggish_model.ckpt')
    response_model = requests.get(url_vggish_model)
    if response_model.status_code == 200:
        with open(local_path_vggish_model, 'wb') as f:
            f.write(response_model.content)
        print('vggish_model.ckpt downloaded successfully!')
    else:
        print('Failed to download vggish_model.ckpt:', response_model.status_code)

    # Download vggish_pca_params.npz
    local_path_vggish_pca_params = os.path.join(local_dir_vggish, 'vggish_pca_params.npz')
    response_params = requests.get(url_vggish_pca_params)
    if response_params.status_code == 200:
        with open(local_path_vggish_pca_params, 'wb') as f:
            f.write(response_params.content)
        print('vggish_pca_params.npz downloaded successfully!')
    else:
        print('Failed to download vggish_pca_params.npz:', response_params.status_code)


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
