# setup.py

from setuptools import setup, find_packages

setup(
    name="perception-system",
    version="0.1.0",
    description="Multi-task perception system for object detection, tracking, and depth estimation",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/perception-system",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pyyaml>=5.1",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
        "scipy>=1.5.0",
        "filterpy>=1.4.5",
        "yolov5>=6.0.0",
        "timm>=0.4.12",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    entry_points={
        "console_scripts": [
            "perception=example_perception:main",
        ],
    },
)