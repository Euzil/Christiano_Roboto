from setuptools import setup, find_packages

setup(
    name='simulator',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        # 写依赖，比如 pybullet 等
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='Simulator package with pybullet wrapper',
    zip_safe=False,
)

