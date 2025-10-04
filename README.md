# FSL-Thesis
Run this by order
- python extract_crop_hands.py
- python preprocess_pipeline.py
- augmentation.py

## Folder Structure Information
- augmented *(automatically created due to augmentation.py)*
- ├──subfolder *(follows folder structure of preaugmentation)*
- crops *(automatically created due to extract_crop_hands.py)*
- ├──subfolder *(follows folder structure of videos)*
- preaugmentation *(manually created for alphanumeric classes without j and z)*
- ├──subfolder *(0)*
- ├──subfolder *(A)*
- processed *(automatically created due to preprocess_pipeline.py)*
- ├──subfolder *(follows folder structure of crops)*
- videos *(manually created for raw data)*
- ├──subfolder *(participant_alphanumericclass)*
- augmentation.py *(manually created)*
- augmented_annotations.csv *(automatically created due to augmentation.py)*
- cropped_annotations.csv *(automatically created due to extract_crop_hands.py)*
- extract_crop_hands.py *(manually created)*
- preprocess_pipeline.py *(manually created)*
- preprocessed_annotations.csv *(automatically created due to preprocess_pipeline.py)*