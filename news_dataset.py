import pathlib
import re
import pandas as pd
import unicodedata
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self,input,output):
        self.input=input
        self.output=output

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,index):
        return self.input[index],self.output[index]
    
class News_Load:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese",is_fast=True)
        self.enc_max_len=700
        self.dec_max_len=70
        self.batch_size=16
        
    def load_newsdata(self):
        input_txt,output_txt=self.news_data()
        input_list=[]
        output_list=[]
        for input,output in zip(input_txt,output_txt):
            input=self.normalize_neologd(input)
            input=self.remove_extra_spaces(input)
            output=self.normalize_neologd(output)
            output=self.remove_extra_spaces(output) 
            #input=unicodedata.normalize("NFKC",input)
            #output=unicodedata.normalize("NFKC",output)
            input_token=self.tokenizer(input,padding='max_length',max_length=self.enc_max_len)
            #output_token=self.tokenizer(output_list,padding='max_length',max_length=self.dec_max_len)
            if len(input_token['input_ids'])>self.enc_max_len:
                continue
            input_list.append(input)
            output_list.append(output)
        x_train,x_test,t_train,t_test=train_test_split(input_list,output_list,test_size=0.2, random_state=42, shuffle=True)
        train_data = [(src, tgt) for src, tgt in zip(x_train, t_train)]
        test_data = [(src, tgt) for src, tgt in zip(x_test, t_test)]
        train,test=self.convert_batch_data(train_data,test_data)
        return train,test


    def news_data(self):
        p_temp = pathlib.Path('text')
        article_list = []
        for p in p_temp.glob('**/*.txt'):
            media = str(p).split('/')[6]
            file_name = str(p).split('/')[7]
            if file_name != 'LICENSE.txt':
                with open(p, 'r',encoding="utf-8") as f:
                    article = f.readlines()
                    article = [re.sub(r'[\n \u3000]', '', i) for i in article]
            article_list.append([media, article[0], article[1], article[2], ''.join(article[3:])])
        article_df = pd.DataFrame(article_list,columns=['col_0', 'col_1', 'col_2', 'col_3','col_4'])
        df=article_df[article_df["col_0"]=='topic-news']
        input_list,output_list=article_df['col_4'].to_list(),article_df['col_3'].to_list()
        print(input_list[0],output_list[0])
        return input_list,output_list
        
    def remove_extra_spaces(self,s):
        s = re.sub('[ 　]+', ' ', s)
        blocks = ''.join(('\u4E00-\u9FFF',  
                      '\u3040-\u309F',  
                      '\u30A0-\u30FF',  
                      '\u3000-\u303F',  
                      '\uFF00-\uFFEF'   
                      ))
        basic_latin = '\u0000-\u007F'

        def remove_space_between(cls1, cls2, s):
            p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
            while p.search(s):
                s = p.sub(r'\1\2', s)
            return s

        s = remove_space_between(blocks, blocks, s)
        s = remove_space_between(blocks, basic_latin, s)
        s = remove_space_between(basic_latin, blocks, s)
        return s
    
    def unicode_normalize(self,cls, s):
        pt = re.compile('([{}]+)'.format(cls))
        def norm(c):
            return unicodedata.normalize('NFKC', c) if pt.match(c) else c

        s = ''.join(norm(x) for x in re.split(pt, s))
        s = re.sub('－', '-', s)
        return s

    def normalize_neologd(self,s):
        s = s.strip()
        s = self.unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)
        
        def maketrans(f, t):
            return {ord(x): ord(y) for x, y in zip(f, t)}

        s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  
        s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s) 
        s = re.sub('[~∼∾〜〰～]+', '〜', s)  
        s = s.translate(
            maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

        s = self.remove_extra_spaces(s)
        s = self.unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  
        s = re.sub('[’]', '\'', s)
        s = re.sub('[”]', '"', s)
        return s
    
    def convert_batch_data(self,train_data, valid_data):
        enc_max_len=self.enc_max_len
        dec_max_len=self.dec_max_len
        tokenizer=self.tokenizer
        def generate_batch(data):
            batch_src, batch_tgt = [], []
            for src, tgt in data:
                batch_src.append(src)
                batch_tgt.append(tgt)

            batch_src = tokenizer(batch_src, max_length=enc_max_len, truncation=True, padding="max_length", return_tensors="pt")
            batch_tgt = tokenizer(batch_tgt, max_length=dec_max_len, truncation=True, padding="max_length", return_tensors="pt")

            return batch_src, batch_tgt

        train_iter = DataLoader(train_data, batch_size=self.batch_size, collate_fn=generate_batch)
        valid_iter = DataLoader(valid_data, batch_size=self.batch_size, collate_fn=generate_batch)

        return train_iter, valid_iter