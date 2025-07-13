from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'bullet_sims'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS 2 visuals package',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'foot_trajectory = bullet_sims.foot_trajectory:main',
            'footstep_planner = bullet_sims.footstep_planner:main',
            'talos = bullet_sims.talos:main',
            'test = bullet_sims.test:main',
            'walking = bullet_sims.walking:main',
            'lip_mpc = bullet_sims.lip_mpc:main',
            
        ],
    },
)