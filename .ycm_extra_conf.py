import os
from rospkg import RosPack

def Settings(**kwargs):

    flags = [
        '-Wall',
        #  '-Wextra',
        '-fexceptions',
        '-ferror-limit=10000',
        '-DNDEBUG',
        '-std=c++11',
        '-xc++',
        '-I/usr/lib/',
        '-I/usr/include/'
        '-I/usr/local/lib'
        '-I/usr/local/include',

        '-I/usr/local/include/opencv4',
    ]

    rospack = RosPack()

    for pkg in rospack.list():
        pkg_path = rospack.get_path(pkg)
        for subdir_path in os.walk(pkg_path):
            subpath = subdir_path[0]
            if subpath.split('/')[-1] == 'include':
                flags.append('-I' + subpath)

    return {'flags': flags}
