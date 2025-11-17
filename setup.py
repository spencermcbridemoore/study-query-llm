from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="study-query-llm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="LLM inference experimentation and analytics framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spencermcbridemoore/study-query-llm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "panel>=1.3.0",
        "python-dotenv>=1.0.0",
        "openai>=1.0.0",  # Azure OpenAI and OpenAI SDK
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter",
        ]
    },
    entry_points={
        "console_scripts": [
            "study-query-llm=panel_app.app:main",
        ],
    },
)
