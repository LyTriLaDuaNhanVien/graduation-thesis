#!/bin/bash

bandit -r duc -o bandit-report.json -f json

if [[ $? -eq 0 ]]; then
echo "No security issues found"
else
echo "Security issues found"
exit 1
fi