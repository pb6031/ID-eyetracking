### Eye-Tracking Based Parallel CNN for Intellectual Disability Classification

---

It contains the code and data for the paper: **"A Novel Multidimensional Eye-tracking Assessment to Identify Cognitive Profiles of Intellectual Disability: A Deep Learning Pilot Study"**

**1. Project overview**

The goal is to create a model that can classify children with intellectual disability(ID) by analyzing eye-tracking data.
We used a CNN to process gaze patterns derived from three cognitive tasks: Verbal comprehension, Fluid reasoning, and Working memory.


**2. Data**

The ‘image.zip’ file contains image data generated from raw eye-tracking data.

* Filename format: [name] _ [label] _ [subject id] _ [problem number] _ [task type].png

* Task Type: The code processes images from three tasks.

* Label:

  * 0: Typical developmental Group
  
  * 1: Intellectual Disability Group

The 'all_feature.csv' file has the behavior and gaze feature values calculated from the raw eye-tracking data.

**3. Model**

The ‘model_train.py’ file contains the CNN model and the training process.
