# Skin Cancer Classification 

A Flask-based educational image-classification application using a trained
TensorFlow/Keras model.

## Required local structure

```text
skin-cancer-classification-app/
├── app.py
├── Dockerfile
├── requirements.txt
├── skin_cancer_model.keras
└── templates/
    └── index.html
```

> The model file is not included in this package. Copy your existing
> `skin_cancer_model.keras` into the project root.

## Important

The current class mapping assumes the model was trained with the default
alphabetical Keras directory order:

- `Cancer` = class 0
- `Non_Cancer` = class 1

Verify this mapping with one known image from each dataset folder before
publishing the application.

This application is for education and technical demonstration only and is not
a medical diagnostic tool.
