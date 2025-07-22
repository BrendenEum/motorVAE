#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard Libraries
import os
import tempfile
import zipfile
import pandas as pd
import numpy as np

# Third Party Libraries
import requests

# Imported Libraries
from evox.evox_client import EvoxHelper

# Get the EVOX key from your ENV Variables
evox_api_key = os.environ['EVOX_KEY'] 
evox_client = EvoxHelper(api_key=evox_api_key)

# Load the CSV file
df = pd.read_csv("viflist.csv")

# Extract unique values from the "VIF #" column
unique_vif_values = df["VIF #"].unique()

# List of url counts for averaging later
url_counts = []

# Vehicle images and max index dictionary
max_ind = {}

# Create a zip file with all images
zip_fn = "all_vehicle_images.zip"
with zipfile.ZipFile(zip_fn, 'w', zipfile.ZIP_DEFLATED) as zipf:

    # Loop through vif values
    for vif in unique_vif_values:

        # Using the Product API get all the urls related to this product.
        json_output = evox_client.get_all_product_urls(
            vif_num=int(vif), # Year-Make-Model-Trim
            product_id=2, # color photo
            product_type_id=41 # extra-large front profile PNG
        )

        try:
            door_str = str(int(df.loc[df["VIF #"]==vif,"Drs"].iloc[0])).replace(" ", "")
        except ValueError:
            door_str = "NaN"

        product = (
            str(int(df.loc[df["VIF #"]==vif,"Yr"].iloc[0])).replace(" ", "") + "_" + 
            str(df.loc[df["VIF #"]==vif,"Make"].iloc[0]).replace(" ", "") + "_" + 
            str(df.loc[df["VIF #"]==vif,"Model"].iloc[0]).replace(" ", "") + "_" + 
            str(df.loc[df["VIF #"]==vif,"Trim"].iloc[0]).replace(" ", "") + "_" + 
            str(df.loc[df["VIF #"]==vif,"Body"].iloc[0]).replace(" ", "") + "_" + 
            door_str + "Door" + "_" + 
            str(int(df.loc[df["VIF #"]==vif,"VIF #"].iloc[0])).replace(" ", "") 
        )
        urls = json_output['urls']
        if product not in max_ind:
            max_ind[f"{product}"] = 0

        print(f"There are {len(urls)} urls avaliable for the {product}.")
        
        # Download all the Images in the List
        if len(urls) > 0:
            url_counts.append(len(urls))
            for idx, url in enumerate(urls):
                try:
                    # Lets Parse the URL for the Original File Name
                    real_ind = max_ind[f"{product}"]
                    img_name = f"{product}_{real_ind+1}.png"
                    r = requests.get(url, stream=True, timeout=5)
                    if r.status_code == 200:
                        zipf.writestr(img_name, r.content)
                        max_ind[f"{product}"] += 1
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
            print("Finished!!!")

    url_counts = np.asarray(url_counts)
    print(f"The average number of images per car with at least one image: {np.mean(url_counts)}")