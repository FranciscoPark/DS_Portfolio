## 산업분류 자동화 인공지능 모델 개발
Using KoBart,KoBigBird, and KoElectra models, our team tuned the pretrained models to classifiy given industrial tags in Korean. Then, we averaged the proabibilities from each model and conducted a vote. 

## Folder Structure

```
pytorch-template/
│
├── train_kobart_v2.py - main script to start kobart training
├── train_kobigbird.py - main script to start kobigbird training
├── train_koelectra.py - main script to start koelectra training
├── test.py - evaluation of trained model
├── ensemble_inference.py - make predictions with three models
│
├── configs/<config>.yaml - holds configuration for training
│
├── data_loader/ - anything about data loading goes here
│   └── data_loaders.py
│
├── data/ - default directory for storing input data
│
├── model/ - models, losses, and metrics
│   ├── model.py
│   ├── metric.py
│   └── loss.py
│
├── saved/
│   ├── models/ - trained models are saved here
│   └── configs/ - trained models' config file in yaml
│
├── trainer/ - trainers
│   └── trainer.py
│
├── checkpoints/ - saving model parameter
│   └── 
├── csvs/ - saving model predictions
│   └── 
│
└── utils/ - small utility functions
    ├── util.py
    └── ...
```
