#!/bin/bash
cd /home/kavia/workspace/code-generation/resume-matcher-301649-301659/ats_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

