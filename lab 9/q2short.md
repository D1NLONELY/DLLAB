import torch, random, os, unicodedata, string, glob, time, math, matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1
findFiles = glob.glob
def unicodeToAscii(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if c in all_letters)
def readLines(f): return [unicodeToAscii(l.strip()) for l in open(f, encoding='utf-8')]

category_lines = {os.path.splitext(os.path.basename(f))[0]: readLines(f) for f in findFiles('./data/data/names/*.txt')}
all_categories = list(category_lines.keys())
n_categories = len(all_categories)

class LSTMModel(torch.nn.Module):
    def __init__(s, i, h, o): super().__init__(); s.h=h; s.i2h=torch.nn.Linear(n_categories+i+h,h); s.i2o=torch.nn.Linear(n_categories+i+h,o); s.o2o=torch.nn.Linear(h+o,o); s.d=torch.nn.Dropout(0.1); s.sf=torch.nn.LogSoftmax(1)
    def forward(s, c, x, h): h = s.i2h(torch.cat((c, x, h),1)); o=s.sf(s.d(s.o2o(torch.cat((h, s.i2o(torch.cat((c,x,h),1))),1)))); return o, h
    def initHidden(s): return torch.zeros(1, s.h)

def tensorify(val, sz): t=torch.zeros(1, sz); t[0][val] = 1; return t
def inputTensor(l): t=torch.zeros(len(l),1,n_letters); [t.__setitem__((li,0,all_letters.find(l[li])),1) for li in range(len(l))]; return t
def targetTensor(l): return torch.LongTensor([all_letters.find(l[i]) for i in range(1,len(l))]+[n_letters-1])

def train(c, x, y): y.unsqueeze_(-1); h=rnn.initHidden(); rnn.zero_grad(); loss=torch.Tensor([0]); [loss.__iadd__(criterion(rnn(c,x[i],h)[0], y[i])) for i in range(x.size(0))]; loss.backward(); [p.data.add_(p.grad.data, alpha=-lr) for p in rnn.parameters()]; return loss.item()/x.size(0)

def timeSince(s): now=time.time()-s; return f'{int(now/60)}m {int(now%60)}s'

rnn,criterion,lr,n_iters,print_every,plot_every,total_loss= LSTMModel(n_letters,128,n_letters),torch.nn.NLLLoss(),0.0005,100000,5000,500,0
start,all_losses=time.time(),[]

for i in range(1,n_iters+1): c,l=random.choice(all_categories),random.choice(category_lines[c]); c,x,y=tensorify(all_categories.index(c),n_categories),inputTensor(l),targetTensor(l); loss=train(c,x,y); total_loss+=loss
    if i%print_every==0: print(f'{timeSince(start)} ({i} {i/n_iters*100:.0f}%) {loss:.4f}')
    if i%plot_every==0: all_losses.append(total_loss/plot_every); total_loss=0

plt.plot(all_losses); plt.show()
