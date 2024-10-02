## GenIC

To reproduce experiments, run the following `sh` command:

```bash
bash run.sh mode lp_checkpoint pp_checkpoint with_type with_desc
```

Replace `mode` with either train or test, depending on the task you wish to perform.
For testing (test mode), specify the names of the models you want to use for lp_checkpoint (Link Prediction) and pp_checkpoint (Property Prediction).
Set `with_type` and `with_desc` to true or false based on whether you want to include entity types and descriptions, respectively.


Examples: 

To train with types and descriptions: 
```bash
bash run.sh train
```
To test on models trained with types but no descriptions:
```bash
bash run.sh test model_lp model_pp true false
```
