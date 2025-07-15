from setuptools import find_packages, setup

package_name = 'whole_body_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch.py']),  

    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='devel',
    maintainer_email='ge93zof@mytum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            't4_standing = whole_body_control.t4_standing:main',
            '02_one_leg_stand = whole_body_control.02_one_leg_stand:main',
            '03_squating = whole_body_control.03_squating:main',
            't51 = whole_body_control.t51:main',
            't52 = whole_body_control.t52:main',
        ],
    },
)