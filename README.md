# ANNotation Assisted Direct Coupling Analisis (annaDCA)
This package contains the methods and scripts to train and sample an RBM model provided with data annotations.

## Installation
### Option 1: from PyPl
```bash
python -m pip install annaDCA
```
### Option 2: cloning the repository
```bash
git clone https://github.com/rossetl/annaDCA.git
cd annaDCA
python -m pip install .
```

## Usage
After installation, all the main routines can be launched through the command-line interface using the command `annadca`.
To see all the training options do
```bash
annadca train -h
```
To train the model with default arguments do
```bash
annadca train -d <path_data> -a <path_annotations> -o <output_directory> -l <model_tag>
```

#### Input data format
The input data should be:
- _binary variables_: plain text format. Each row is one data sample, variables are separated by white spaces
- _categorical variables_: [fasta](https://en.wikipedia.org/wiki/FASTA_format) format. Each data poin is a sequence of tokens with an header on top. The header row starts with `>`.

#### Annotation data format
Annotations bust be provided in a `csv` file with two columns. One column is called (mandatory!) `Name`, an the other column represents the labels and can have any chosen name.

For categorical varables, the `Name` field must match one of the headers in the fasta file, while for binay variables the order of the rows is used. 

Un-annotated data should not be present in the annotation file.