from setuptools import setup

package_name = 'save_can_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krri',
    maintainer_email='aaaa@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
                'avante_can = save_can_interface.avante_interface:main',
                'avante2_can = save_can_interface.avante2_interface:main',
                'v2x_interface = save_can_interface.v2x_interface:main',
        ],
    },
)
