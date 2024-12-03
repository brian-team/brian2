import re
import os
import pkg_resources
import tomllib
import setuptools_scm
import json

basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

with open(os.path.join(basedir, "pyproject.toml"), "rb") as f:
    pyproject = tomllib.load(f)
with open(os.path.join(basedir, ".codemeta", "codemeta_base.json"), "r", encoding="utf-8") as f:
    codemeta = json.load(f)
with open(os.path.join(basedir, "AUTHORS"), "r", encoding="utf-8") as f:
    authors = f.read().splitlines()[6:]

# Add software requirements from pyproject.toml
parsed_deps = pkg_resources.parse_requirements(pyproject['project']['dependencies'])
codemeta["softwareRequirements"] = []
for dep in parsed_deps:
    version = ",".join(f"{op}{v}" for op,v in dep.specs)
    requirement = {"name": dep.project_name,"@type": "SoftwareApplication", "runtimePlatform": "Python 3"}
    if version:
        requirement["version"] = version
    codemeta["softwareRequirements"].append(requirement)

# Add contributors from AUTHORS
codemeta["contributor"] = []
for author in authors:
    matches = re.match(r"^(\w[\w-]*?) ([\w]+[.]? )??(\w+) \(@(.*)\)$", author)
    if not matches:
        raise ValueError("author not matched:", author)
    given_name, middle_name, family_name, github = matches.groups()

    contributor = {"@type": "Person", "givenName": given_name, "familyName": family_name, "identifier": f"https://github.com/{github}"}
    # FIXME: additionalName does not seem to be recognized by codemeta (validation fails)
    if middle_name:
        contributor["givenName"] += " " + middle_name.strip()
    codemeta["contributor"].append(contributor)

# Add version from setuptools_scm
version = setuptools_scm.get_version(root=basedir)
codemeta["version"] = version
codemeta["softwareVersion"] = version

# Write codemeta.json
with open(os.path.join(basedir, "codemeta.json"), "w", encoding="utf-8") as f:
    json.dump(codemeta, f, indent=2, ensure_ascii=False)
