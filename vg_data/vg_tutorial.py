from visual_genome import api
import json
import requests


print("--------get image data by id--------")
image = api.get_image_data(id=61512)
print(image)

# equals to:
image = requests.get("http://visualgenome.org/api/v0/images/61512").json()
print(image)

# get image url
image_url = image['url']
print(image_url)



print("--------get all region descriptions for an image--------")
regions = api.get_region_descriptions_of_image(id=61512)
print(regions[0])

# equals to:
regions = requests.get("http://visualgenome.org/api/v0/images/61512/regions").json()
print(regions[0])

# get phrases of all regions
phrases = [region['phrase'] for region in regions]
print(phrases)








