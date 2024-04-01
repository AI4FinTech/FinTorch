#!/bin/bash
# Set up a link to the API key to root's home.
mkdir /home/vscode/.kaggle
ln -s /workspaces/FinTorch/kaggle/kaggle.json /home/vscode/.kaggle/kaggle.json
chmod 600 /home/vscode/.kaggle/kaggle.json
