import warnings
from importlib import import_module


def import_modules_with_relative(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (dict | list | str | None): The given module names or (pairs of name and package) to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    relative_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if isinstance(imports, dict):
        relative_import = True
    imported = []
    if relative_import:
        for module, package in imports.items():
            if not isinstance(package, str):
                raise TypeError(
                    f"{package} is of type {type(package)} and cannot be imported."
                )
            try:
                imported_tmp = import_module(module, package)
            except ImportError:
                if allow_failed_imports:
                    warnings.warn(
                        f"{package} failed to import and is ignored.", UserWarning
                    )
                    imported_tmp = None
                else:
                    raise ImportError
            imported.append(imported_tmp)
    else:
        for imp in imports:
            if not isinstance(imp, str):
                raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
            try:
                imported_tmp = import_module(imp)
            except ImportError:
                if allow_failed_imports:
                    warnings.warn(
                        f"{imp} failed to import and is ignored.", UserWarning
                    )
                    imported_tmp = None
                else:
                    raise ImportError
            imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported
