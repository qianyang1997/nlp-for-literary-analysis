# NLP for Literary Analysis

The goal of this project is to apply NLP techniques to literary analysis. The example use case in this repo is text classification on Barack and Michelle Obamas' autobiographies. For details on the project and results, see the project report pdf document.

A text classification API is built for the project. The API takes in custom string inputs and outputs a dictionary of prediction results (the probability of the input being authored by Barack or Michelle). Specific instructions on the API are outlined below.

How to run REST API:

Clone this repo and install relevant packages in requirements.txt.

Navigate to the root directory. If you don't have `git-lfs` installed, install it, then run:
```{sh}
git lfs install
git lfs pull
python3 api.py
```
Copy and paste the generated hyperlink in the browser. Type in sentences in the query box and you'll see the generated json output.
