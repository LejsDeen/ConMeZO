## ConMeZO on RoBERTa

The code is for reproducing the results of ConMeZO on RoBERTa-large.

### Preparations

The datasets can be found [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). 
Please download it and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash prepare_datasets.sh
cd ..
```
### Examples

To reproduce the results (on dataset SST-2) of our paper, 
you can run them with the according configuration commands. 
Results on other datasets can be obtained by changing `TASK` to `sst-5`, `SNLI`, `MNLI`, `RTE`, and `trec`. For detailed hyperparameters, please refer to Appendix C.2 of our paper.
```bash
# ConMeZO
bash examples/cone.sh

# MeZO baseline
bash examples/mezo.sh
```
