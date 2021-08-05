import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import matplotlib
# import matplotlib.font_manager
#
# x = [f.name for f in matplotlib.font_manager.fontManager.ttflist if 'New Gulim' in f.name]
# print(x)

subscription_key = 'ad364518d66a4df09b7a510a85e42cbc'
endpoint = 'https://daegu0001.cognitiveservices.azure.com/vision/v2.0/'
analyze_url = endpoint + "ocr"

image_path = "Vehicle/license/licenseLicense000122.png"


image_data = open(image_path, "rb").read()
headers = {'Ocp-Apim-Subscription-Key': subscription_key,
           'Content-Type': 'application/octet-stream'}
params = {'language': 'ko', 'detectOrientation': 'true'}  # params = {'language': 'ko', 'detectOrientation': 'true'}
response = requests.post(
    analyze_url, headers=headers, params=params, data=image_data)
response.raise_for_status()

analysis = response.json()
print(analysis)
line_infos = [region["lines"] for region in analysis["regions"]]
word_infos = []

for line in line_infos:
    for word_metadata in line:
        for word_info in word_metadata["words"]:
            word_infos.append(word_info)

print(word_info)
# plt.figure(figsize=(5, 5))
#
# plt.rcParams["font.family"] = 'Nanum'
# # # 'New Gulim'or 'Nanum'
# plt.rcParams["font.size"] = 10
#
# image = Image.open(BytesIO(requests.get(image_url).content))
# ax = plt.imshow(image, alpha=0.5)
# for word in word_infos:
#     bbox = [int(num) for num in word["boundingBox"].split(",")]
#     text = word["text"]
#     origin = (bbox[0], bbox[1])
#     patch = plt.Rectangle(origin, bbox[2], bbox[3], fill=False, linewidth=2, color='y')
#     ax.axes.add_patch(patch)
#     plt.text(origin[0], origin[1], text, fontsize=20, weight="bold", va="top")
#
# plt.axis("off")
# plt.show()