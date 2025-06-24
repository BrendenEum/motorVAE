#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard Libraries
import os
import tempfile
import zipfile

# Third Party Libraries
import requests

# Imported Libraries
from evox.evox_client import EvoxHelper

# Get the EVOX key from your ENV Variables
evox_api_key = os.environ['EVOX_KEY'] 
evox_client = EvoxHelper(api_key=evox_api_key)

# Using the Product API get all the urls related to this product
json_output = evox_client.get_all_product_urls(
    vif_num=4137, # Toyota Corolla 2007
    product_id=23, # 
    product_type_id=173 # 
)

product_urls = json_output['urls']
print(product_urls)
print(f"There are {len(product_urls)} urls avaliable for this product")

# Download all the Images in the List

if len(product_urls) > 0:
    print("Downloading all the Evox Images")
    zip_path = "evox_batch.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for idx, url in enumerate(product_urls):
            try:
                # Lets Parse the URL for the Original File Name
                img_name = product_urls[idx].split("/")[-1]
                r = requests.get(url, stream=True, timeout=5)
                if r.status_code == 200:
                    zipf.writestr(f"{img_name}.jpg", r.content)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
    print("Finished!!!!")
