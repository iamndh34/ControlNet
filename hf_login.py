#!/usr/bin/env python3
"""
Login to HuggingFace.
"""
from huggingface_hub import login

print("Enter your HuggingFace token:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Create a new token (type: Read)")
print("3. Paste the token below\n")

token = input("Token: ").strip()

login(token=token)
print("Login successful!")
