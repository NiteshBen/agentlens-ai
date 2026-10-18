from setuptools import setup, find_packages

setup(
    name="agentlens-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "agentlens = agentlens.cli:main",
        ],
    },
)
