from setuptools import setup, find_packages

setup(
    name='annadca',
    version='0.2.1',
    author='Lorenzo Rosset, Aurélien Decelle, Beatriz Seoane, Francesco Zamponi, Martin Weigt',
    maintainer='Lorenzo Rosset',
    author_email='rosset.lorenzo@gmail.com',
    description='Annotation Assisted Direct Coupling Analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rossetl/annaDCA',
    packages=find_packages(include=['annadca', 'annadca.*']),
    include_package_data=True,
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'annadca=annadca.cli:main',
        ],
    },
    install_requires=[
        'adabmDCA>=0.7.0'
    ],
)
