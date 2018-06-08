from setuptools import setup

setup(
    name='pokemon-recognition',
    version='0.1',
    description='module for pokemon recognition',
    author='Lion Pierau',
    author_email='lion.pierau@gmail.com',
    packages=['networks', 'scripts'],
    install_requires=['keras', 'sklearn', 'imutils', 'opencv-python', 'matplotlib'],
)
