from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss as fs

from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key="openrouter api key")

sent2vec_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=100,length_function=len)
with open(r"path\to\the\txt\file\you\want",encoding="utf-8") as txt_file:
    txt = txt_file.read()
txt = text_splitter.split_text(txt)
txt_embedding = np.array([sent2vec_model.encode(chunk) for chunk in txt],dtype=np.float32)

vector_dbase = fs.IndexFlatL2(768)

vector_dbase.add(txt_embedding)

prompt_template = """Answer the question only according to the information provided below.
## Information : 
{}

## Question : 
# {}"""

q = input("question? : ")
q_embedding = np.array([sent2vec_model.encode(q)],dtype=np.float32)

_,index = vector_dbase.search(q_embedding,3)
infos = [txt[i] for i in index.tolist()[0]]

info_text = ""
for info in infos:
    info_text += info + "\n---\n"
info_text = info_text.strip()

prompt = prompt_template.format(info_text,q)

completion = client.chat.completions.create(
extra_headers={},
extra_body={},
model="deepseek/deepseek-r1-zero:free",
messages=[{"role":"user","content":prompt}])

print((completion.choices[0].message.content)[7:-1])
