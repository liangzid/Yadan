# Convert Pre-trained Models.
Please use `convert_models.py` for convert *python huggingface transformers* pretrained model into a rust-bert style models.

(Noted that we build our task-oriented dialogue systems based on `rust-bert`.)


## Requirements

1. python with pytorch
2. rust related environments

## Running

```shell
# first activate your python environments
python convert_models.py --source_file /path/your/pre-defined-models/
```
