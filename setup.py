from setuptools import setup

setup(
    name='cosmo_learn',
    version='0.0.0',
    description='Cosmological simulations and learning in one',
    url='https://github.com/reggiebernardo/cosmo_learn',
    author='Reginald Christian Bernardo',
    author_email='reginaldchristianbernardo@gmail.com',
    license='MIT',
    packages=['cosmo_learn'],
    install_requires=[
        'numpy',
        'scipy==1.13.1',
        'scikit-learn==1.6.1',
        'astropy',
        'numdifftools',
        'multiprocess',
        'corner',
        'emcee',
        'tqdm',
        'geneticalgorithm',
        'torch',
        'torchvision',
        'torchaudio',
        'refann'
    ],
    python_requires='==3.10.*',
)
