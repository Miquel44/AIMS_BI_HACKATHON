# -*- coding: utf-8 -*- noqa

import time

import copy
import datetime
import importlib
import json
import logging
import os
import pathlib
import pickle
import sys
import requests
# Time execution
TIME_EXECUTION = datetime.datetime.now(datetime.timezone.utc)
time_format = ''

# Environment files
ENVIRONTMENT_FILE_PATH = pathlib.Path(os.path.abspath('env/.env'))
TEMPORAL_ENVIRONTMENT_FILE_PATH = pathlib.Path(
    os.path.abspath('env/temp.env'),
)

# Python
PYTHON_VERSION = '{0}.{1}.{2}'.format(
    *sys.version_info.__getnewargs__()[0][:3]
)

# Platform
PLATFORM = sys.platform.lower()

# User
USER = ''

if PLATFORM == 'win32':  # Windows
    USER = os.getenv('USERNAME')
else:  # Unix-like platforms
    USER = os.getenv('USER')

# Context
loaded = False


def load_env():
    """
    Load environment from file "./env/.env".

    Raises
    ------
    SystemError
        Python is not correct version.
    ModuleNotFoundError
        Module version is not correct.

    Returns
    -------
    bool
        If environment was loaded succesfully or not.

    """
    global log_level_name, log_level
    global time_format
    global paths, variables
    global loaded

    if loaded:
        return False

    if not ENVIRONTMENT_FILE_PATH.exists():
        return False

    if not ENVIRONTMENT_FILE_PATH.is_file():
        return False

    with open(ENVIRONTMENT_FILE_PATH,  'r') as file:
        env = json.load(file)

    # Python comprobatioon
    if PYTHON_VERSION != env['python_version']:
        ERROR = f'Python {env["python_version"]} not found ({PYTHON_VERSION}).'
        # logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise SystemError(ERROR)

    # Paths
    class Paths:
        pass

    paths = Paths()

    paths.code = pathlib.Path(os.path.abspath(os.path.split(__file__)[1]))

    for path_name, path_value in env['paths'].items():
        path_value = pathlib.Path(os.path.abspath(path_value))
        env['paths'][path_name] = path_value
        os.makedirs(path_value, exist_ok=True)
        setattr(paths, path_name, path_value)

    del path_name, path_value

    # Varaibles
    class Variables:
        pass
    variables = Variables()

    for variable_name, variable_value in env['variables'].items():
        setattr(variables, variable_name, copy.deepcopy(variable_value))

    del variable_name, variable_value

    # Time format
    time_format = env['time_format']

    # Logging
    log_level_name = copy.deepcopy(env['log_level'])
    log_level = logging.getLevelNamesMapping()[log_level_name]

    logging.basicConfig(
        filename=os.path.join(
            paths.logs,
            f'{TIME_EXECUTION.strftime(time_format)}--{PLATFORM}--{USER}--'
            + f'{log_level_name}.log',
        ),
        filemode='w',
        level=log_level,
        force=True,
        format=(
            '[%(asctime)s] %(levelname)s:\n\tModule: "%(module)s"\n\t'
            + 'Function: "%(funcName)s"\n\tLine: %(lineno)d\n\tLog:\n\t\t'
            + '%(message)s\n'
        ),
    )

    logging.info('Start logger.')

    # Load external modules
    for module_name, module_version in env['external_libraries'].items():
        globals()[module_name] = importlib.import_module(module_name)

        if globals()[module_name].__version__ != module_version:
            ERROR = f'Module {module_name} {module_version} not found'
            + f' (found {globals()[module_name].__version__}).'
            logging.critical(ERROR.replace('\n', '\n\t\t'))
            raise ModuleNotFoundError(ERROR)

    del module_name, module_version

    # Load environ
    for environ_name, environ_value in env['environ'].items():
        os.environ[environ_name] = environ_value

    # Save information for unload
    with open(TEMPORAL_ENVIRONTMENT_FILE_PATH, 'wb') as file:
        pickle.dump(env, file)

    del env

    loaded = True

    return True


def unload_env():
    """
    Unload environment.

    Returns
    -------
    bool
        If environment was unloaded succesfully or not.

    """
    global loaded

    if not loaded:
        return False

    if not TEMPORAL_ENVIRONTMENT_FILE_PATH.exists():
        return False

    if not TEMPORAL_ENVIRONTMENT_FILE_PATH.is_file():
        return False

    with open(TEMPORAL_ENVIRONTMENT_FILE_PATH, 'rb') as file:
        env = pickle.load(file)

    # Paths
    del globals()['paths']

    # Variables
    del globals()['variables']

    # Time format
    del globals()['time_format']

    # Logging
    logging.shutdown()
    del globals()['log_level']
    del globals()['log_level_name']

    # Unload modules
    for module_name in env['external_libraries'].keys():
        del globals()[module_name]
        del sys.modules[module_name]

    # Unload environ
    for environ_name in env['environ'].keys():
        del os.environ[environ_name]

    # Delted information from load
    TEMPORAL_ENVIRONTMENT_FILE_PATH.unlink()

    loaded = False

    return True


def reload_env():
    """
    Reload environment from file "./env/.env".

    Returns
    -------
    bool
        If environment was reloaded succesfully or not.

    """
    unload_success = unload_env()
    load_success = load_env()

    return unload_success and load_success


def get_environment_information():
    """
    Return the information about the environment in the moment of load of it.

    Returns
    -------
    information : str
        String with the information of the loaded environent in the momenet of load.

    """
    information = ''

    information += '\033[4mGeneral information\033[0m\n'
    information += f'Environment: "{ENVIRONTMENT_FILE_PATH}"\n'
    information += f'Python version: {globals()["PYTHON_VERSION"]}\n'
    information += f'Platform: {globals()["PLATFORM"]}\n'
    information += f'User: {globals()["USER"]}\n'
    information += 'Time execution: '
    information += TIME_EXECUTION.strftime(time_format) or str(TIME_EXECUTION)
    information += bool(time_format) * f' ({time_format})' + '\n'
    information += f'Loaded: {globals()["loaded"]}\n'

    if globals()['loaded']:
        information += 'Loaded environment: '
        information += f'"{TEMPORAL_ENVIRONTMENT_FILE_PATH}"\n'
    else:
        return information

    if not TEMPORAL_ENVIRONTMENT_FILE_PATH.exists():
        information += 'Saved environment not found.\n'
        return information

    if not TEMPORAL_ENVIRONTMENT_FILE_PATH.is_file():
        information += 'Saved environtment is not on a proper format.\n'
        return information

    with open(TEMPORAL_ENVIRONTMENT_FILE_PATH, 'rb') as file:
        env = pickle.load(file)

    # Logging
    information += f'Log mode: {log_level_name} (log_level)\n'

    # Paths
    information += '\033[4mPaths\033[0m\n'
    information += f'Code path: "{globals()["paths"].code}"\n'

    for path_name, path_value in env['paths'].items():
        information += f'{path_name.title()} path: "{path_value}"\n'

    # Variables
    information += '\033[4mVariables\033[0m\n'

    for variable_name, variable_value in env['variables'].items():
        information += f'{variable_name} = {variable_value}\n'

    # Modules
    information += '\033[4mModules\033[0m\n'

    for module_name, module_version in env['external_libraries'].items():
        information += f'{module_name} ({module_version})\n'

    # Environ
    information += '\033[4mEnviron\033[0m\n'

    for environ_name, environ_value in env['environ'].items():
        information += f'"{environ_name}" = "{environ_value}"'

    return information


def check_environment():
    """
    Check is the state of the loaded environemnt maches with the saved state of the environment.

    Returns
    -------
    bool
        If the loaded environemnt maches with the information temporallly saved.

    """
    if not loaded:
        return False

    if not TEMPORAL_ENVIRONTMENT_FILE_PATH.exists():
        return False

    if not TEMPORAL_ENVIRONTMENT_FILE_PATH.is_file():
        return False

    with open(TEMPORAL_ENVIRONTMENT_FILE_PATH, 'rb') as file:
        env = pickle.load(file)

    if PYTHON_VERSION != env['python_version']:
        return False

    if time_format != env['time_format']:
        return False

    if log_level_name != env['log_level']:
        return False

    for path_name, path_value in env['paths'].items():
        if getattr(paths, path_name, False) != path_value:
            return False

    for variable_name, variable_value in env['variables'].items():
        if getattr(variables, variable_name, None) != variable_value:
            return False

    for module_name, module_version in env['external_libraries'].items():
        if globals()[module_name].__version__ != module_version:
            return False

    for environ_name, environ_value in env['environ'].items():
        if os.environ.get(environ_name, False) != environ_value:
            return False

    return True


# # Prints
# try:
#     TERMINAL_WIDTH = os.get_terminal_size().columns
# except OSError:
#     TERMINAL_WIDTH = 80
# SECTION = '='
# SECTION_LINE = SECTION * TERMINAL_WIDTH
# SEPARATOR = '-'
# SEPARATOR_LINE = SEPARATOR * TERMINAL_WIDTH
# MARKER = '*'
if __name__ == "__main__":
    load_env()
    print(get_environment_information())
    print(f'CHECK: {check_environment()}')
    time.sleep(20)  # Simulate some processing time
    unload_env()
    pass