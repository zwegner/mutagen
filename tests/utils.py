import os
import sys

def import_path(path):
    path = os.path.abspath(path)
    path, file = os.path.split(path)
    file, ext = os.path.splitext(file)
    sys.path.append(path)
    module = __import__(file)
    sys.path.pop()
    return module

