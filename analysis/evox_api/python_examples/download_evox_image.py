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

# Using the Product API get all the urls related to this product
json_output = evox_client.get_all_product_urls(
    vif_num=4137, # Toyota Corolla 2007
    product_id=21, # stills_0480 - 480x360 canvas size with copyright on grey stage - JPEG
    product_type_id=162 # stills_0480 - 480x360 canvas size with copyright on grey stage - JPEG
)

product_urls = json_output['urls']
print(product_urls)
print(f"There are only {len(product_urls)} urls avaliable for this product")

# Lets Download One of the Possible Images in the List
if len(product_urls) > 0:

    # Lets Parse the URL for the Original File Name
    img_name = product_urls[0].split("/")[-1]

    img_data = requests.get(product_urls[0]).content
    with open(f'{img_name}', 'wb') as handler:
        handler.write(img_data)
