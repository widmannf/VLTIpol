from setuptools import setup, find_packages

setup(name='VLTIpol',
      version='1.0',
      author='Felix Widmann',
      description='Python package to get the instrumental polarization of the VLTI',
      url='https://github.com/widmannf/VLTIpol',
      python_requires='>=3.7',
      packages=['VLTIpol'],
      package_dir={'':'src'},
      package_data={'VLTIpol': ['Models/*']},
      include_package_data=True,
      install_requires=[
        'matplotlib',
        'numpy',
        'math'
        'pkg_resources',
        'astropy',
    ]
)
