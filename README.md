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
