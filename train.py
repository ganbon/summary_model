import torch
from tqdm import tqdm
from model import T5model
from news_dataset import News_Load
import copy
from pathlib import Path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=T5model()
model=model.to(device)
torch.cuda.empty_cache()
epoch=10
best_loss = float('Inf')
pad=model.tokenizer.pad_token_id
news=News_Load()
train,test=news.load_newsdata()
optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001)
history={
    'train_loss':[],
    'test_loss':[],
    'test_acc':[]
}
for e in range(epoch):
    model.train()
    train_losses=0
    pbar=tqdm(train)
    for data,target in pbar:
        optimizer.zero_grad()
        labels=target["input_ids"].to(device)
        labels[labels[:,:]==pad]=-100
        output=model(input_ids=data["input_ids"].to(device,dtype=torch.long)
                    ,attention_mask=data["attention_mask"].to(device,dtype=torch.long),
                    decoder_attention_mask=target["attention_mask"].to(device,dtype=torch.long),
                    labels=labels)
        loss=output['loss']
        optimizer.step()
        train_losses+=float(loss)
        pbar.set_postfix(loss=train_losses/epoch)
        loss_train=train_losses /len(data)
        history['train_loss'].append(loss)
    model.eval()
    test_losses = 0
    with torch.no_grad():
        for data, target in test:
            labels = target['input_ids'].to(device,dtype=torch.long)
            labels[labels[:, :] == pad] = -100

            outputs = model(
                input_ids=data['input_ids'].to(device,dtype=torch.long),
                attention_mask=data['attention_mask'].to(device,dtype=torch.long),
                decoder_attention_mask=target['attention_mask'].to(device,dtype=torch.long),
                labels=labels
            )
            test_loss = outputs['loss']
            test_losses += loss.item()
    if best_loss > test_loss:
        best_loss = test_loss
        best_model = copy.deepcopy(model)
        counter = 1
        
model_dir_path = Path('/content/drive/MyDrive/Colab Notebooks/model')
if not model_dir_path.exists():
    model_dir_path.mkdir(parents=True)

model.tokenizer.save_pretrained(model_dir_path)
best_model.model.save_pretrained(model_dir_path)    


