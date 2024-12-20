"""
Update CITATION.cff and README.md with the latest Zenodo and Software Heritage records
"""
import os
import re
import subprocess

import pyaml
import requests
import yaml

basedir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

# Get the latest tag from git
tag = (
    subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
    .strip()
    .decode("utf-8")
)

print("Latest tag/release:", tag)
ZENODO_API = "https://zenodo.org/api"
# DOI always linking to the latest version (escaped slash for elasticsearch syntax)
CONCEPT_DOI = r"10.5281\/zenodo.654861"

print("Searching for Zenodo record...")
# Get the latest version via concept doi
# The Zenodo search guide says that one can directly search by "version:...", but it does not work (BAD REQUEST)
r = requests.get(ZENODO_API + "/records", params={"q": f"conceptdoi:{CONCEPT_DOI}"})
if not r.ok:
    raise RuntimeError("Request failed: ", r.reason)
data = r.json()
assert data["hits"]["total"] == 1
latest_zenodo_id = data["hits"]["hits"][0]["id"]

# Get all versions
r = requests.get(ZENODO_API + f"/records/{latest_zenodo_id}/versions")
if not r.ok:
    raise RuntimeError("Request failed: ", r.reason)
data = r.json()
versions = data["hits"]["hits"]
latest_version = [v for v in versions if v["metadata"]["version"] == tag]
if latest_version:
    zenodo_record = latest_version[0]
    print("Found Zenodo record for version", tag)
else:
    raise RuntimeError("No Zenodo record found for version " + tag)

print("Searching for SWH record...")

# Find Software Heritage
resp = requests.get(
    "https://archive.softwareheritage.org/api/1/origin/https://github.com/brian-team/brian2/get/"
)
if not resp.ok:
    raise RuntimeError("Request failed: ", resp.reason)
data = resp.json()
visits_url = data["origin_visits_url"]
resp = requests.get(visits_url)
if not resp.ok:
    raise RuntimeError("Request failed: ", resp.reason)
data = resp.json()
latest_visit = sorted(data, key=lambda x: x["date"], reverse=True)[0]
snapshot_url = latest_visit["snapshot_url"]
resp = requests.get(snapshot_url)
if not resp.ok:
    raise RuntimeError("Request failed: ", resp.reason)
data = resp.json()
swh_record = data["branches"].get(f"refs/tags/{tag}")

if swh_record:
    print("Found SWH record for version", tag)
else:
    swh_record = None
    print("No SWH record found for version", tag)

print("Updating CITATION.cff and README.md...")
with open(os.path.join(basedir, "CITATION.cff"), "r") as f:
    citation_cff = yaml.load(f, Loader=yaml.SafeLoader)
for identifier in citation_cff["identifiers"]:
    if zenodo_record and identifier["description"].startswith(
        "This is the archived snapshot of version"
    ):
        identifier["value"] = zenodo_record["metadata"]["doi"]
        identifier["description"] = (
            f"This is the archived snapshot of version {tag} of Brian 2"
        )
    if swh_record and identifier["type"] == "swh":
        identifier["value"] = f"swh:1:rel:{swh_record['target']}"
        identifier["description"] = (
            f"Software Heritage identifier for version {tag} of Brian 2"
        )

with open(os.path.join(basedir, "CITATION.cff"), "w") as f:
    pyaml.dump(citation_cff, f, sort_keys=False)

with open(os.path.join(basedir, "README.md"), "r") as f:
    readme = f.read()

if zenodo_record:
    # Replace the old DOI with the new one
    readme = re.sub(
        r"\[!\[DOI\]\(https://zenodo.org/badge/DOI/.*\.svg\)\]\(https://zenodo.org/doi/.*\)",
        f"[![DOI](https://zenodo.org/badge/DOI/{zenodo_record['metadata']['doi']}.svg)](https://zenodo.org/doi/{zenodo_record['metadata']['doi']})",
        readme,
    )

if swh_record:
    # Replace the old SWH badge with the new one
    # [![Software Heritage (release)](https://archive.softwareheritage.org/badge/swh:1:rel:2d4c5c8c8a6d2318332889df93ab74aef53e2c61/)](https://archive.softwareheritage.org/swh:1:rel:2d4c5c8c8a6d2318332889df93ab74aef53e2c61;origin=https://github.com/brian-team/brian2;visit=swh:1:snp:a90ab7416901a9c5cf6f56d68b3455c65d322afc)
    readme = re.sub(
        r"\[!\[Software Heritage \(release\)\]\(https://archive.softwareheritage.org/badge/swh:1:rel:.*\)\]\(https://archive.softwareheritage.org/swh:1:rel:.*;origin=.*\)",
        f"[![Software Heritage (release)](https://archive.softwareheritage.org/badge/swh:1:rel:{swh_record['target']}/)](https://archive.softwareheritage.org/swh:1:rel:{swh_record['target']};origin=https://github.com/brian-team/brian2;visit=swh:1:snp:{latest_visit['snapshot']})",
        readme,
    )

with open(os.path.join(basedir, "README.md"), "w") as f:
    f.write(readme)
