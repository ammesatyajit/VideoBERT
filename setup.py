import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VideoBERT", # Replace with your own username
    version="0.0.1",
    author="Satyajit Kumar",
    author_email="ammesatyajit@gmail.com",
    description="Reproducing the results of VideoBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ammesatyajit/VideoBERT",
    packages=setuptools.find_packages(),
    install_requires=[
        "transformers",
        "tensorboardX",
        "torch",
        "tensorflow_hub",
        "opencv-python"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "unlicensed",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)