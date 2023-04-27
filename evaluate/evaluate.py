# This file immplements an automatic evaluation to evaluate the success of S1 as compared to S0

from build_test_data import build_img_path
from PIL import Image as PIL_Image
from generate_captions import generate_captions


# TODO: define a listener L_eval(image|caption)
# (Bayes' rule: L_eval(image|caption) = (P_S0(caption|image) * P(image))/P(caption)?)
# assume that the P(image) and P(caption) is equal for all images/captions, that leads us to 
# L_eval(image|caption) = P_S0(caption|image)
# def L_eval(image, caption):
#     return None

# Test set 1: 10 entry x 10 clusters x 10 most common objects, 1000 entry
# Always choose the first entry (image) in each cluster as the target 
                




# The caption succeeds as referring expression if the target has more probability mass under the distribution L_eval(image|caption) than any distractor
#success = 0
#if max([L_eval(image) for image in cluster]) == L_eval(target):


if __name__ == "__main__":
    #img_path = build_img_path("vg_data/ts1_img/")
    #print(img_path)
    #img = PIL_Image.open(img_path['sign'][0][1])
    #img.show()
    #img.close()
    paths = ['vg_data/ts1_img/man/569_4957241', 'vg_data/ts1_img/man/910_4974287', 'vg_data/ts1_img/man/770_5525114', 'vg_data/ts1_img/man/51_4931359', 'vg_data/ts1_img/man/420_4949802', 'vg_data/ts1_img/man/1080_4982799', 'vg_data/ts1_img/man/305_4944073', 'vg_data/ts1_img/man/911_4974350', 'vg_data/ts1_img/man/598_5523166', 'vg_data/ts1_img/man/953_5527977']
    generate_captions(paths, greedy=False)
    