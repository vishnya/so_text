from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = '0.1'

setup(
    name='so_text',
    version=version,
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/vishnya/so_text/',
    description='Repo to analyze stack overflow text')
