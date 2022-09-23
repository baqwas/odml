#!/usr/bin/env python3
import fiftyone as fo

dataset = fo.Dataset("my_dataset")

sample = fo.Sample(filepath="/path/to/image.png")
sample.tags.append("train")
sample["custom_field"] = 51

dataset.add_sample(sample)

view = dataset.match_tags("train").sort_by("custom_field").limit(10)

for sample in view:
    print(sample)
