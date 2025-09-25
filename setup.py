from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="study-query-llm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Panel application starter template",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/study-query-llm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "panel>=1.3.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "hvplot>=0.9.0",
        "jupyter",
        "jupyterlab>=3.0",
        "param>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "study-query-llm=panel_app.app:main",
        ],
    },
)