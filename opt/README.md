## ConMeZO on OPT

The code is for reproducing the results of ConMeZO on OPT models (tested on OPT-1.3B and OPT-13B).

### Examples

To reproduce the results (on dataset SST-2) of our paper, 
you can run them with the according configuration commands.
Results on other datasets can be obtained by changing `TASK` (see the shell scripts for supported tasks).
For detailed hyperparameters, please refer to Appendix C.3 of our paper.
```bash
# ConMeZO
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-7 BS=8 CONE_THETA=1.35 CONE_BETA=0.95 bash examples/cone.sh

# MeZO baseline
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-7 BS=8 EPS=1e-3 bash examples/mezo.sh
```