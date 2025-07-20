from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> list[str]: 
    """
    This function reads the requirements from a file and returns them as a list.
    """
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        # Remove new line characters from each line
        requirements = [req.replace("\n", "") for req in requirements]

    # Remove any -e . line if it exists
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='student_performance_indicator',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[get_requirements('requirements.txt')],
    author='Gemechis W.',
    author_email='game.worku@gmail.com'

)