import os
from os.path import dirname, abspath, join

base_path = dirname(abspath(__file__))
temp_path = join(base_path, 'temp')
pycache_path = join(base_path, '__pycache__')

def delete_uwaw(path):
    for data in os.listdir(path):
        if data != '.gitkeep':
            try:
                os.remove(join(path, data))
            except Exception as a:
                print(a)
                
    print("Success removing temp file...")

delete_uwaw(temp_path)
delete_uwaw(pycache_path)