file_name = [
    "assets/coffee.png",
    "assets/coffee.png",
    "assets/coffee.png",
    "assets/coffee.png",
    "assets/clock.jpg",
    "assets/book.jpg",
    "assets/monalisa.jpg",
    "assets/oranges.jpg",
    "assets/penguin.jpg",
    "assets/vase.jpg",
    "assets/room_corner.jpg",
]

test_instruction = [
    "Make the image look like it's from an ancient Egyptian mural.",
    'get rid of the coffee bean.',
    'remove the cup.',
    "Change it to look like it's in the style of an impasto painting.",
    "Make this photo look like a comic book",
    "Give this the look of a traditional Japanese woodblock print.",
    'delete the woman',
    "Change the image into a watercolor painting.",
    "Make it black and white.",
    "Make it pop art.",
    'the sofa is leather, and the wall is black',
]

import os
id = 0
for x, y in zip(file_name, test_instruction):
    id += 1
    y = '"' + y + '"'
    os.system(f"ASCEND_RT_VISIBLE_DEVICES=7 python scripts/inference.py --image train/{x} \
    --instruction {y} \
    --seed 304897401 --save-name {id}.png --output-dir results")

