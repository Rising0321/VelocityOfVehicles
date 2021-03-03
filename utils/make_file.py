import os

def make_output_dir(name):
    if not os.path.exists('output'):
        os.makedirs('output')

    if not os.path.exists('output/' + name):
        os.mkdir('output/' + name)