from setuptools import setup, find_packages

# Read the contents of the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="dadrl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
     extras_require={
        "tensorboard": ["tensorboard~=2.19.0"], 
    },   
    # Add more metadata as needed
    author="Francesco Frattolillo",
    description="discrete abstraction for sparse deep reinforcement learning",
    url="https://github.com/yourusername/your_project",
)
