# MobileNetV2 Page Classifier Artifact

Tracked handoff files:

- `page_classifier_mobilenetv2.keras`: Keras MobileNetV2 page classifier model
- `page_classifier_class_names.json`: model output class order
- `page_classifier_report.json`: test classification report
- `page_classifier_confusion_matrix.csv`: test confusion matrix
- `page_classifier_history.csv`: training history

The default class order used by the webcam runner is `none,page1,page2,page3`.

Current test report summary:

- accuracy: `1.0`
- class order: `none,page1,page2,page3`
- confusion matrix: diagonal only on the provided 40-image test split
