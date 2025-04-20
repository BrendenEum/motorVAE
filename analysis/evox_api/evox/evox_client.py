#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Default Libraries
import json

# Third Party Libraries
import requests

def return_json_file(raw_json, file_name):

    """Returns nicely formated json file as .json. Used for debugging.
    Args:
        param raw_json(dict):    Takes in a json dict.
        param file_name(string): Name of the file name you want to write too.
    Return:
        True (boolean):          Return True if the function finished properly
                                 (for now)
    """

    # Open file with the ability to write to the file
    with open(file_name, "w") as data_file:
        json.dump(raw_json, data_file, indent=4, sort_keys=True)

    return True

class EvoxHelper:

    def __init__(self, api_key):

        if api_key is None:
            raise Exception("No api key was passed in")

        self.api_key = api_key
        self.url = "https://api.evoximages.com/api/v1"
        self.header = {"x-api-key": self.api_key}

    def get_viflists(self):

        """Return a list of all the different downloadable vif nums

        Return:
            raw_json (dict): A JSON of the raw json 
        
        """

        url = self.url + '/viflists'

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_vid(self, model_num):

        """Get the video of the model

        Input:
            model_num (int): The model number of the vehicle that you are trying to get the video of

        """


        url = self.url + '/vids/' + str(model_num)

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_all_makes(self):

        """Get a list of all the different makes listed on Evox
        """

        url = self.url + '/makes'

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_vehicle_data(self, vif_num):

        """ Get all the hotspot data from a specific vif number

        Input: 
            vif_num (Int): vifnum is a unique identifier for a 
                           specific set of media for one vehicle's trim.
                           Ex: 12234
        Return:
            raw_json (dict): The json of the hotspot data

        """

        url = "https://api.evoximages.com/api/v1/vehicles/" + str(vif_num)

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_all_vehicle_data(self):

        """Get all vehicle data. This call doesn't work 
        """

        url = "https://api.evoximages.com/api/v1/vehicles/"
    
        data = {
            'page': 1
        }

        r = requests.get(url, headers=self.header, data=data)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_vif_product_info(self, vif_num, product_id, product_type_id):

        """Get the full information on a product info

        Input: 
            vif_num (Int): The vif number of a vehicle
            product_id (Int): The product id of the certain vehicle
            product_type_id (Int): The product type of id

        Return:
            raw_json (dict): The information as a json

        """


        url = self.url + "/vehicles/" + str(vif_num) + "/products/" + str(product_id) + "/" + str(product_type_id)

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json


    def get_hotspot_data(self, vif_num):

        """ Get all the hotspot data from a specific vif number

        Input: 
            vif_num (Int): vifnum is a unique identifier for a 
                           specific set of media for one vehicle's trim.
                           Ex: 12234
        Return:
            raw_json (dict): The json of the hotspot data

        """

        url = "https://api.evoximages.com/api/v1/vehicles/" + str(vif_num) + "/hotspots"

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_stills(self, vif_num):

        """ Get all the color data from a specific vif number

        Input: 
            vif_num (Int): vifnum is a unique identifier for a 
                           specific set of media for one vehicle's trim.
                           Ex: 12234
        Return:
            raw_json (dict): The json of the color data

        """

        url = "https://api.evoximages.com/api/v1/vehicles/" + str(vif_num) + "/stills"

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json
    
    def get_mappings(self):

        url = "https://api.evoximages.com/api/v1/mappings"

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_specs(self):

        """Get a list of the documentation of all the specs you are allowed to use
        
        """

        url = "https://api.evoximages.com/api/v1/specguides"

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_products(self):

        """Get a list of all the products that the subscription has access too
        """

        url = "https://api.evoximages.com/api/v1/products"

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_product_info(self, product_id):

        """Get the avaliable information of a product by the product id

        Input:
            product_id (int): The product id of the vehicle. This can be found on the documentation in the product table

        Return:
            raw_json (dict): The raw json of the returned infomration

        """

        url = "https://api.evoximages.com/api/v1/products" + "/" + str(product_id)

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

    def get_color_list(self, vif_num):

        """ Get all the color data from a specific vif number

        Input: 
            vif_num (Int): vifnum is a unique identifier for a 
                           specific set of media for one vehicle's trim.
                           Ex: 12234
        Return:
            raw_json (dict): The json of the color data

        """

        url = "https://api.evoximages.com/api/v1/vehicles/" + str(vif_num) + "/colors"

        r = requests.get(url, headers=self.header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json
    
    def get_all_product_urls(self, vif_num, product_id, product_type_id):

        """Get the avaliable information of a product by the product id

        Input:
            vif_num (int): The Unique Identifier of the Vehicle. This can be found in the Viflist.csv
            product_id (int): The product id of the set of images. This can be found on the documentation in the product table
            product_type_id (int): The product_type id of the set of images. This can be found in the README
        Return:
            raw_json (dict): The raw json of the returned infomration

        """

        url = f"https://api.evoximages.com/api/v1/vehicles/{vif_num}/products/{product_id}/{product_type_id}"
        header = {"x-api-key": self.api_key}
        r = requests.get(url, headers=header)
        if r.status_code != 200:
            raise Exception('Unable to Return Request {}'
                            .format(r.status_code))

        raw_json = r.json()
        return raw_json

if __name__ == "__main__":
    client = EvoxHelper(api_key="")
    #raw_json = client.get_products()
    #raw_json = client.get_product_info(product_id=2)
    #return_json_file(raw_json, "all_products.json")
    raw_json = client.get_stills(vif_num="12")
    #raw_json = client.get_all_makes()
    print(raw_json)
    #return_json_file(raw_json, "12_stills_data.json")
