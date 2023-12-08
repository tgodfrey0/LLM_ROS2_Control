import os
from setuptools import setup, find_packages
from glob import glob

package_name = 'llm_controller'

setup(
    name=package_name,
    version='0.0.1',
    # packages=[package_name],
    packages=find_packages(exclude=["test"]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tg',
    maintainer_email='toby.godfrey2003@gmail.com',
    description='Interacts with an LLM and parse the response into topic messages.',
    license='GPT-3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'llm_node = llm_controller.llm_node:main',
            'camera_info_publisher = llm_controller.camera_info_publisher:main',
            'hardware_protection = llm_controller.hardware_protection:main'
        ],
    },
)
