from setuptools import find_packages, setup

setup(
    name="chesslm",
    version="0.1.1",
    author="Yash Bonde",
    author_email="bonde.yash97@gmail.com",
    description="LMs playing chess",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yashbonde/chess_lm",
    package_dir={"": "chess_lm"},
    packages=find_packages("chess_lm"),
)
