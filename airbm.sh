#!/bin/bash

# Check if the first positional argument is provided
if [ -z "$1" ]; then
  echo "Error: No command provided. Use 'train'."
  exit 1
fi

# Assign the first positional argument to a variable
COMMAND=$1
shift # Remove the first positional argument, so "$@" now contains only the optional arguments

# Map the command to the corresponding script
case "$COMMAND" in
  train)
    SCRIPT="train.py"
    ;;
  *)
    echo "Error: Invalid command '$COMMAND'. Use 'train'."
    exit 1
    ;;
esac

# Run the corresponding Python script with the remaining optional arguments
python3 scripts/$SCRIPT "$@"