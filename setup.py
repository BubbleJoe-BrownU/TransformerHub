from setuptools import setup, find_packages

setup(
  name = 'transformerhub',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'A comprehensive model hub for Transformers',
  long_description_content_type = 'text/markdown',
  author = 'Jiayu Zheng',
  author_email = 'jiayuzheng99@gmail.com',
  url = 'https://github.com/BubbleJoe-BrownU/TransformerHub',
  keywords = ['transformer', 'bert', 'gpt'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.9',
  ],
)
