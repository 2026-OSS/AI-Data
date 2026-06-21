# MobileNetV2 Page Classifier Artifact v2

Tracked handoff files:

- `page_classifier_mobilenetv2.keras`: Keras MobileNetV2 page classifier model copied from `page_classifier_mobilenetv2 (2).keras`
- `page_classifier_class_names.json`: model output class order

The default class order used by the backend and webcam runner is `none,page1,page2,page3`.

Source notebook:

- `/Users/joyein/Downloads/2026_OSS_page_classifier_v3.ipynb`

Notes:

- The backend default page classifier now points to this v2 model.
- Export fresh v2 evaluation files from the notebook if `page_classifier_report.json`, `page_classifier_confusion_matrix.csv`, or `page_classifier_history.csv` are needed for handoff.
