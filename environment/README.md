## Environment


## Step-by-step environment setup

I will use anaconda virtual environment as an example, 

The guidance can be also adapted to python naive pip.

**Create environment**

```
conda create -n RSE-env python=3.7
```

**Activate the environment**

```
conda activate RSE-env
```

**Install Accelerate**

```
pip install importlib-metadata
pip install accelerate==0.12.0
```

**Install PyTorch**

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

**Install transformers**

```
pip install transformers==4.21.3
```

**Install datasets**

```
pip install datasets==2.3.2
```

**Install scipy**

```
pip install scipy==1.5.4
```

**Install prettytable (for eval)**

```
conda install -c conda-forge prettytable=3.3.0
```


**Install sk-learn (for eval)**

```
pip install scikit-learn==1.0.2
```

**Install pytrec-eval (for eval)**

```
pip install pytrec-eval==0.5
```

