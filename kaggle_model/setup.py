from setuptools import setup

setup(name='kaggle_model',
      version='0.1',
      description='Code for architecture and training script',
      author='Julien Horwood',
      packages=['kaggle_model'],
      install_requires=[
          'numpy', 'sklearn', 'torch','matplotlib','skorch'
      ],
      zip_safe=False)