import json
import os
import re
import sys
import tomllib

from packaging.requirements import Requirement

if __name__ == "__main__":
    if not len(sys.argv) == 2:
        raise ValueError("Usage: python create_codemeta.py <version>")
    basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    with open(os.path.join(basedir, "pyproject.toml"), "rb") as f:
        pyproject = tomllib.load(f)
    with open(
        os.path.join(basedir, ".codemeta", "codemeta_base.json"), encoding="utf-8"
    ) as f:
        codemeta = json.load(f)
    with open(os.path.join(basedir, "CONTRIBUTORS"), encoding="utf-8") as f:
        contributors = f.read().splitlines()[4:]  # Skip comments at the beginning

    # Add software requirements from pyproject.toml
    codemeta["softwareRequirements"] = []
    for dependency in pyproject["project"]["dependencies"]:
        dep = Requirement(dependency)
        version = str(dep.specifier)
        requirement = {
            "name": dep.name,
            "@type": "SoftwareApplication",
            "runtimePlatform": "Python 3",
        }
        if version:
            requirement["version"] = version
        codemeta["softwareRequirements"].append(requirement)

    # Add contributors from AUTHORS
    codemeta["contributor"] = []
    for contributor in contributors:
        matches = re.match(r"^(.*?) \(@([^)]*)\)$", contributor)
        if not matches:
            raise ValueError(f"author not matched: '{contributor}'")
        full_name, github = matches.groups()

        name_parts = full_name.split()
        if len(name_parts) == 1:
            given_name = name_parts[0]
            family_name = None
        else:
            given_name = " ".join(name_parts[:-1])
            family_name = name_parts[-1]

        contributor_dict = {
            "@type": "Person",
            "givenName": given_name,
        }
        if family_name is not None:
            contributor_dict["familyName"] = family_name

        contributor_dict["identifier"] = f"https://github.com/{github}"

        codemeta["contributor"].append(contributor_dict)

    # Add version from setuptools_scm
    version = sys.argv[1]
    codemeta["version"] = version
    codemeta["softwareVersion"] = version

    # Write codemeta.json
    with open(os.path.join(basedir, "codemeta.json"), "w", encoding="utf-8") as f:
        json.dump(codemeta, f, indent=2, ensure_ascii=False)
