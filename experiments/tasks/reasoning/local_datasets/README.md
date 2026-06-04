# 本地推理数据集

将下载好的 GPQA Diamond 数据集放在这里：

```
local_datasets/
└── gpqa_diamond/
    ├── train.jsonl
    └── dataset_info.txt
```

## 放置方法

1. 在有代理的机器上下载：
```python
from datasets import load_dataset
import json

ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)

with open("train.jsonl", "w") as f:
    for item in ds:
        f.write(json.dumps(item) + "\n")
```

2. 打包上传到服务器：
```bash
tar -czvf gpqa_diamond.tar.gz train.jsonl
scp gpqa_diamond.tar.gz server:/home/hhw/zy/paroquant/experiments/tasks/reasoning/local_datasets/
```

3. 解压：
```bash
cd /home/hhw/zy/paroquant/experiments/tasks/reasoning/local_datasets/
tar -xzvf gpqa_diamond.tar.gz
mkdir -p gpqa_diamond
mv train.jsonl gpqa_diamond/
```
