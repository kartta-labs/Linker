# Linker
This module connects text strings on a map to generate complete phrase (e.g., connecting “Los” and “Angeles” 
to generate “Los Angeles” or “Madison” and “Ave.” to generate “Madison Ave.”). The need for this module 
raises from our oservation of how text recognition module operates. The output of a text recognition tool 
(e.g., the Vision API) for a scanned map is usually a set of words which are not necessarily connected to 
each other. The challenge is how to reverse engineer the cartographic principles behind various label 
placement scenarios. The simple case would be a place phrase containing two text labels close to each other 
horizontally without any other nearby text labels. A challenging case would be a curved place phrase 
containing a group of large single characters with wide spacing (e.g., a place phrase labeling a long 
winding river).

## Definition
**Objective**: connect relevant text strings to make complete place phrases.

**Input**: text strings and their associated features such as minimum boundingboxes of individual text 
labels and font styles (e.g., capitalization, font sizes, etc.).

**Output**: complete place phrases (e.g., “Madison Ave.” from “Madison” and “Ave.”).

**Evaluation**: phrase-level accuracy using precision and recall.

## How to run
**Step1**:
`sh run_step1.sh` generates initial linkage prediction by utilizing the local information associated with each text patch, such as boundong box rotation angle, average font area, bounding box center locations and etc. The input folder `preprocess_dir` should provide information about `word_list` and `word_coords_list`. The `word_list` is a list of separate single words detected on the map and `word_coords_list` is a list of bounding box coordinates (x_min, y_min, x_max, y_max) that corresponds to the `word_list`. These two lists could be produced by either text detection models or APIs such as Google Vision API.

In this step, path and name of pretrained weight are required to specify through `dml_weight_dir` and `dml_weight_name`, a pretrained weight can be found [here](https://drive.google.com/drive/folders/1n4SO71w8iZHc0fAbhU8tCd16-28srH7o?usp=sharing)

**Step2**:
`sh run_step2.sh` refines the initial linkage prediction by adding global information, which means the model will not only considers the attributes of single word text regions, but also consider the image context it belongs to. Since the first step could filter out most of the negative linkages given a query text region, we only need to consider the neighborhood defined by the positive pairs. 

**Step3**:
`sh run_step3.sh` constructs a directed graph where nodes are separate text regions and edges are the positive linkages predicted from step2. We detect the connected component and sort the elements in the component according to the x-axis coordinate location, then the final location phrases could be produced.

The pretrained weight for deciding capitalization is [here](https://drive.google.com/drive/folders/1n4SO71w8iZHc0fAbhU8tCd16-28srH7o?usp=sharing).

