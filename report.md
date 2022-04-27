---
title: 'Identifying the author of French poems based on their prose'
author:
- Cameron Hochberg
- Jonathan Karkour
---

## Introduction

Machine learning has proved to be a great tool for classifying images. Recently there has also been a lot of research on text classification as well. However, most of the research has been mostly focused on English examples only. While this gave us great advances in the field, it also is very limiting as for the datasets available but there is also the risk of building models with heavy biases inherent to the english language.

## Problems

The idea of our project is to take poems from different authors of the same literary era and identify who the author is based on the writing. All the authors would be picked from the same century and since the literary movement would be the same, we hope to make it so that the only difference would be the actual writing itself rather than the style and emotion behind.

## Dataset

A dataset of french poem doesn't really exist so we went to create our own. Using the **Gutenberg project**, we used the text versions of the following collections of poems: **Les Fleurs du Mal** de *Charles Baudelaire*, **Les contemplations: Autrefois, 1830-1843** de *Victor Hugo*, **Œuvres complètes - Volume 1** de *Paul Verlaine*, and **La Comédie de la mort** de *Théophile Gautier*.
After downloading each collection, we used a custom script `text_splitter.py` to split each into separate poems blocks that we would then use to train our model. After doing some statistics, we saw that most poems were in between 70 and 140 words long and split each block so that they would be 100 words long and created 10000 examples for each class.
Each block was placed inside a folder representing its label.

## Methods



## References
