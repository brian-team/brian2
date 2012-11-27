#!/bin/bash

pylint --rcfile=dev/jenkins/pylint.rc brian2 > pylint.log || :
