# Codes for Knowledge Graph Embedding

pytorch>=1.10
geoopt=0.5.0

```
📦directory
 ┣ 📂data
 ┣ 📂manifolds
 ┣ 📂utils             # Utils for manifolds
 ┣ 📂optim
 ┣ 📜UltraModel.py    # Ultrahyperbolic model
 ┣ 📜README.md
 ┣ 📜load_data.py
 ┣ 📜main.py
 ┣ 📜run.fb15k327.32.sh
 ┣ 📜run.wn18rr.32.sh
 ```

## 1. Usage
 To run the experiments, simply run the corresponding training script, e.g.,
 ```bash
python main.py
 ```

 You can specify the arguments that are passed to the program:

`--dataset`           The dataset you are going to train the model on. Can be [FB15k-237, WN18RR]      

`--num_epochs`        Number of training steps.

`--batch_size`        Controls the batch size.

`--nneg`              Number of negative samples.

`--lr`                Controls the learning rate.

`--dim`               The dimension for entities and relations.

`--early_stop`        Controls the number of early stop step.

`--max_norm`          Controls the maximum norm of the last n dimension of the n+1 dimension entity and relation embeddings. Set to non-positive value to disable.

`--max_scale`         Controls the scaling factor in Lorentz linear layer.

`--margin`            Controls the margin when calculating the distance.

`--max_grad_norm`     Controls the maximum norm of the gradient.

`--real_neg`          If set, the negative samples will be guranteed to be real negative samples.

`--optimizer`         Optimizer. Can be [rsgd, radam].

`--valid_steps`       Controls the validation interval.
