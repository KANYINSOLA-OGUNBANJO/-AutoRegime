import importlib.util, sys, site
spec = importlib.util.find_spec("psutil")
print("spec =", spec)
print("origin =", getattr(spec, "origin", None))
print("locations =", list(getattr(spec, "submodule_search_locations", []) or []))
print("\nsite-packages:", site.getsitepackages())
print("user-site    :", site.getusersitepackages())
print("\nsys.path (first 10):")
for p in sys.path[:10]:
    print("  ", p)