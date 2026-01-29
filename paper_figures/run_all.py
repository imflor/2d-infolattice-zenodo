import importlib

for name in ["fig08", "fig09", "fig10", "fig11", "fig12", "fig13"]:
    print(f"[run_all] running figures/{name}.py")
    importlib.import_module(f"figures.{name}")
