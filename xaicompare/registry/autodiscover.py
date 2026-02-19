
# xaicompare/registry/autodiscover.py
import importlib
import pkgutil
import threading
import xaicompare.adapters as adapters_pkg

__DISCOVERED = False
__LOCK = threading.Lock()

def autodiscover_adapters():
    """
    Recursively import all modules under xaicompare.adapters.*
    so that @register_model / @register_xai decorators execute.
    Safe to call multiple times.
    """
    global __DISCOVERED
    if __DISCOVERED:
        return
    with __LOCK:
        if __DISCOVERED:
            return

        # DEBUG: where are we scanning?
        # print("[DEBUG] autodiscover in:", adapters_pkg.__name__, "paths:", list(adapters_pkg.__path__))

        # walk_packages is recursive: finds subpackages and their modules
        for finder, modname, ispkg in pkgutil.walk_packages(
            adapters_pkg.__path__, adapters_pkg.__name__ + "."
        ):
            # Optionally, you can restrict to only certain trees:
            # if not (modname.startswith("xaicompare.adapters.models")
            #         or modname.startswith("xaicompare.adapters.explainers")):
            #     continue
            try:
                importlib.import_module(modname)
                # print("[DEBUG] imported:", modname)
            except Exception as e:
                # Don't fail discovery if one module has an optional dependency
                # print(f"[WARN] Failed importing {modname}: {e}")
                pass

        __DISCOVERED = True
