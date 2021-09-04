from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['ma_soft_actor_critic'],
    package_dir={'': 'src'}
)

setup(**d)
