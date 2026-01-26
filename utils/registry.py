import inspect
import importlib
import os
import sys
from typing import Dict, Any, Optional

class Registry:
    """
    The registry that provides name -> object mapping, to support simple
    third-party integration.
    """

    def __init__(self, name: str):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any):
        if name in self._obj_map:
            raise ValueError(f"An object named '{name}' was already registered in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj: Any = None, name: str = None):
        """
        Register the given object under the the name `obj.__name__` if name is not provided.
        Can be used as either a decorator or a function call.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                n = name if name else func_or_class.__name__
                self._do_register(n, func_or_class)
                return func_or_class
            return deco

        # used as a function call
        n = name if name else obj.__name__
        self._do_register(n, obj)
        return obj

    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry! Available: {list(self._obj_map.keys())}")
        return ret

    def list_available(self):
        return list(self._obj_map.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map


# Define global registries
MODELS = Registry("models")
DATASETS = Registry("datasets")
LOSSES = Registry("losses")
CALLBACKS = Registry("callbacks")


def import_modules_from_folder(folder: str, package_prefix: str):
    """
    Automatically scan and import python files from a folder to trigger registration.

    Args:
        folder: The file system path to the folder.
        package_prefix: The package prefix to use for importing (e.g., 'models').
    """
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("_"):
                # Construct the module name
                # e.g., models/losses/focal_loss.py -> models.losses.focal_loss

                # Get relative path from the base folder
                rel_path = os.path.relpath(os.path.join(root, filename), os.path.dirname(folder))

                # Convert path to module format
                module_name = rel_path.replace(os.sep, ".").replace(".py", "")

                # Add prefix if needed (though usually folder path matches package structure)
                # If folder is ".../models" and we want "models.resnet",
                # rel_path might be "models/resnet.py" if we passed ".../" as folder?
                # Let's assume folder is absolute path to the package directory.

                try:
                    importlib.import_module(module_name)
                    # print(f"Successfully imported {module_name}")
                except ImportError as e:
                    print(f"Failed to import {module_name}: {e}")
                except Exception as e:
                    print(f"Error importing {module_name}: {e}")
