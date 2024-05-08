from setuptools import setup, find_packages

setup(
    name='FCN_KAN',
    version='0.1.0',
    packages=find_packages(),
    description='Kolmogorov–Arnold Networks with modified activation (using fully connected network to represent the activation)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yanbo Zhang',
    author_email='yzhang84@tufts.edu',
    url='https://github.com/Zhangyanbo/FCN-KAN',
    install_requires=[
        'torch>=1.7.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='KAN, FCN-KAN, PyTorch',
)
