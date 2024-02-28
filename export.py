import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertForTokenClassification

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-chinese')
        self.crf = CRF(2, batch_first=True)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
        
model = Model()

dummy_input = (torch.randint(0, 1000, (1, 512)), torch.randint(0, 1, (1, 512)))

traced_bert = torch.jit.trace(model.bert, 
                              dummy_input, 
                              strict=False)

traced_crf = torch.jit.script(model.crf)

class CombinedModel(torch.nn.Module):
    def __init__(self, traced_bert, traced_crf):
        super(CombinedModel, self).__init__()
        self.traced_bert = traced_bert
        self.traced_crf = traced_crf

    def forward(self, input_ids, attention_masks):
        outputs = self.traced_bert(input_ids, attention_masks)
        decoded = self.traced_crf.decode(outputs['logits'][:, 1:, :]) 
        return decoded

cm = CombinedModel(traced_bert, traced_crf)
print(cm(*dummy_input))

model = torch.jit.script(cm)
model.eval() 

torch.onnx.export(model,               # TorchScript模型
                dummy_input,               # 模型输入的示例
                'model.onnx',      # 输出ONNX模型的文件名
                export_params=True,          # 导出模型参数
                opset_version=11,            # ONNX操作集的版本
                do_constant_folding=True,    # 是否执行常量折叠优化
                input_names=['input_ids', 'attention_masks'],       # 输入名
                output_names=['output'],   # 输出名
                dynamic_axes={'input_ids': {0: 'batch_size'},   # 动态轴的定义
                                'attention_masks': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})