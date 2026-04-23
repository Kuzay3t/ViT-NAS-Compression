# setup.py
from setuptools import setup, find_packages

setup(
    name="vit-nas-compression",
    version="0.1.0",
    description="Hardware-aware NAS framework for efficient Vision Transformers",
    author="Kuzay3t",
    author_email="your-email@example.com",
    url="https://github.com/Kuzay3t/ViT-NAS-Compression",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3",
        "tqdm>=4.60",
        "tensorboard>=2.12",
        "onnx>=1.14",
        "onnxruntime>=1.16",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
            "isort>=5.12",
        ],
        "nas": [
            "optuna>=3.0",
            "ray>=2.0",
            "deap>=1.4",
        ],
    },
)
