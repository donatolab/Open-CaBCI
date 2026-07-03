from setuptools import setup

setup(
    name="openbmi",
    version="0.1",
    description="open source [ca] bmi imaging tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/donatolab/Open-CaBCI",
    author="Donato Lab",
    author_email="flavio.donato@unibas.ch",
    packages=["openbmi"],
    install_requires=[
        'numpy==1.24.4',
        'matplotlib==3.5.2',
        'parmap==1.6.0',
        'tqdm==4.64.1',
        'scipy==1.9.1',
        'opencv-python==4.7.0.72',
        'scikit-learn==1.6.1',
        'scikit-image==0.19.2',
        'pandas==1.4.4',
        'networkx==2.8.4',
        'openpyxl==3.0.10',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
