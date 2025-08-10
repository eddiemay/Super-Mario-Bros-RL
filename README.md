# Super Mario Bros Reinforcement Learning

Let's create an AI that's able to play Super Mario Bros! We'll be using Double Deep Q Network Reinforcement Learning algorithm to do this.

Watch the accompanying YouTube video [here](https://youtu.be/_gmQZToTMac)! Hope you enjoy it!

## Installation

**First, clone this repository**

```bash
git clone https://github.com/Sourish07/Super-Mario-Bros-RL.git
```

**Next, create a virtual environment**

The command below is for a conda environment, but use whatever you're comfortable with. I'm using Python 3.10.12.

```bash
conda create --name smbrl python=3.10.12
```

Make sure you activate the environment.

```bash
conda activate smbrl
```

**Install requirements**

```bash
pip install -r requirements.txt
```

**Start training on a level**

```bash
python main.py SuperMarioBros-1-1-v0
```

Note: You can specify movement type: -r=Right Only, -s=Simple, -c=Complex