"""
A set of helper functions. You may add your own to make your codebase cleaner.
"""

def read_file(file):
    with open(file, 'r', encoding="utf-8-sig") as f:
        return f.readlines()