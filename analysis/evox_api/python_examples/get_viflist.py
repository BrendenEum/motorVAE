#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard Libraries
import os

# Third Party Libraries
import requests

# Imported Libraries
from evox.evox_client import EvoxHelper

# Get the EVOX key from your ENV Variables
evox_api_key = os.environ['EVOX_KEY'] 
evox_client = EvoxHelper(api_key=evox_api_key)

# Call the EVOX API to get the latest VIF List
vif_json_output = evox_client.get_viflists()
print(vif_json_output)

# Parse the JSON and get the correct URL
csv_url = vif_json_output['data'][0]
response = requests.get(csv_url)
content = response.content.decode('utf-8', errors='replace')

# Write the VIF List CSV to your File System
utf8_filename = "viflist.csv"
with open(utf8_filename, "w", encoding="utf-8") as f:
    f.write(content)


