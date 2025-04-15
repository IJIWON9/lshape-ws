from setuptools import setup
from glob import glob
package_name = 'lshape_classification'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'datasets', 'models'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/weights', ['weights/svm_model_500.pkl']),
        (f'share/{package_name}/weights', ['weights/svm_model_1000.pkl']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amlab',
    maintainer_email='wldnjs946429@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'contour_processor = lshape_classification.contour_processor:main',
            'contour_infer_node = lshape_classification.contour_infer_node:main',
            'infer_cnn1d = lshape_classification.infer_cnn1d:main',
            
        ],
    },
)
