"""Pedestrian attribute head: label schema, datasets, embeddings, MLP head, report.

A frozen body ReID network embeds every training image once; a small MLP maps the stored
embedding to 67 attribute probabilities (see :mod:`reid.attributes.schema`). Only
:mod:`reid.attributes.head` (NumPy inference) is needed to use a trained head.
"""
