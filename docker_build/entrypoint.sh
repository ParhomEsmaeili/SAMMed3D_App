#!/bin/bash
# Fix permissions for the home directory and .vscode-server
chown -R $USER:$USER /home/$USER
chown -R $USER:$USER /workspace
# If no arguments are given, start a shell as $USER
if [ $# -eq 0 ]; then
  exec su - $USER
else
  exec su - $USER -c "$*"
fi
