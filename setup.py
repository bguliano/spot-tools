from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_full.txt") as f:
    requirements_full = f.read().splitlines()

setup(
    name="spot-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'full': requirements_full
    }
)
