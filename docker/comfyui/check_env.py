"""Build check: the locked packages satisfy ComfyUI's own requirements and torch is the pinned build."""

import sys
from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.requirements import Requirement

problems = []
with open("/opt/ComfyUI/requirements.txt", encoding="utf-8") as f:
    for line in f:
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            problems.append(f"{requirement.name} is missing")
            continue
        if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
            problems.append(f"{requirement.name} {installed} does not satisfy {requirement.specifier}")
if torch.__version__ != sys.argv[1]:
    problems.append(f"torch {torch.__version__}, expected {sys.argv[1]}")
if problems:
    sys.exit("environment check failed: " + "; ".join(problems))
print("environment check passed")
