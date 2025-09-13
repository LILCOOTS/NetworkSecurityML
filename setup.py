from setuptools import setup, find_packages
from typing import List

def get_requirements():
    # this will return the list of requirements
    requirements_list: List[str] = []
    try:
        with open("requirements.txt") as file_obj:
            lines = file_obj.readlines()
            for line in lines:
                req = line.strip()
                if req and req != "-e .":
                    requirements_list.append(req)
        return requirements_list
    except FileNotFoundError:
        print("requirements.txt file not found.")
        return []
    
setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Om Asanani",
    author_email="asanniom27@gmil.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
    
