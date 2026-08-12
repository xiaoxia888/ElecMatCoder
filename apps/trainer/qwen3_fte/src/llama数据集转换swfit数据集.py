from swift.llm.dataset.register import register_dataset
from swift.llm.dataset.utils import load_dataset, DatasetMeta

def load_my_dataset(dataset_path, **kwargs):
    # 自定义加载逻辑
    ds = load_dataset('json', data_files=dataset_path, **kwargs)
    def preprocess(example):
        return {
            'messages': [
                {'role': 'system', 'content': example['instruction']},
                {'role': 'user', 'content': example['input']},
                {'role': 'assistant', 'content': example['output']}
            ]
        }
    return ds.map(preprocess)

# 注册数据集
register_dataset(
    DatasetMeta(
        dataset_name='my_custom_dataset',
        dataset_loader=load_my_dataset,
        dataset_kwargs={}
    )
)