from setuptools import setup

package_name = 'tracker'

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
    maintainer='sdu',
    maintainer_email='sungju29@g.skku.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracker_node = tracker.tracker_node:main',
            'multi_tracker_node = tracker.multi_tracker_node:main',
            'fake_object_node = tracker.fake_object_publisher:main',
            'static_tracker_node = tracker.static_tracker_node:main',
            'bbox_tracker_node = tracker.bbox_tracker_node:main',
        ],
    },
)
