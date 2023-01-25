# FaceReconstruction :no_mouth:

Inpainting project done on https://paperswithcode.com/dataset/ffhq with custom inpainting. ( right now done on 128x128 - for testing purposes, find the thumbnails on: https://drive.google.com/drive/folders/1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv)



### End Goals / Plan
 Table of end points:

| project feature      | points awared | status |
| ----------- | ----------- | ----------- |
| Project type: inpainting      | 3       |  :white_check_mark: |
| Additional problem: optimization of generator and discriminator training  | 1        |  :white_check_mark: |
| Model: pre-trained model on the different problem      | 1       |  :x: |
| Model: non-trivial solution    | 1        |  :white_check_mark: |
| Dataset: > 10000 photos      | 1       |  :white_check_mark: |
| Dataset: own part > 500    | 1        |  :x: |
| Training: hyperparameter tuning      | 1       |  :white_check_mark: |
| Training: architecture tuning (at least 3 architecture)      | 1       |  :white_check_mark: |
| Training: overfitting some examples from the training set      | 1       |  :white_check_mark: |
| Training: data augmentation ( different mask on training)      | 1       |  :white_check_mark: |
| Training: cross-validation      | 1       |  :x: |
| Training: testing a few optimizers      | 1       |  :x: |
| Training: testing various loss functions      | 1       |  :white_check_mark: |
| Additional: Neptune      | 1       |  :white_check_mark: |
| Additional: Docker      | 1       |  :x: |
|     |    |   |
| Sum      | 18       |  :x: |



## Progress documentation:


**9.01.2023 Task 1:** :white_check_mark:

- try basc inpainting model (true model will be developed later in time) 
- download ułomny dataset 128x128


**11.01.2023 Task 2:** 

- Construct basic architecture for experiment ( for example model type + optimizer)
- Do overfitting experiment (architecture)
- Discuss and research possible models + look into using a pretrained one
- discuss git for large data...


**11.01.2023 Task 3:** 
- Masking specific parts of face architecture (using pretrained object detector, objects being parts of face)
- See if masking specific face parts (mouth , eyes nose etc) makes sense




## Possible sources:
1. MAT: Mask Aware Transformer for Large Hole Image Inpainting | CVPR 2022

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; https://www.youtube.com/watch?v=gxD6lKz1cLQ






