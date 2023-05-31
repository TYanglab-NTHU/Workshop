import sys,os, math
from rdkit import Chem
import torch
import warnings
from PIL import Image  
from matplotlib import pyplot as plt, ticker
sys.path.append('../')
from rdkit.Chem import MACCSkeys, Draw
from fast_jtnn.vocab import *
from fast_jtnn.nnutils import create_var
from fast_jtnn.datautils import tensorize
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.jtprop_vae import JTPropVAE
from torch.nn import CosineSimilarity
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
hidden_size = 450
latent_size = 56
depthT = 20
depthG = 3
vocab = os.path.join('../','fast_molopt','data_vocab.txt')
vocab = [x.strip("\r\n ") for x in open(vocab)]
vocab = Vocab(vocab)
model_path=os.path.join('../','fast_molopt','vae_model','model.epoch-99')

def check_input(input):
    try:
        val = float(input)
        return val
    except :
        raise ValueError('LFS value must be a number and between 0~1 !')
    
def load_model(vocab=vocab,hidden_size=hidden_size,latent_size=latent_size,depthT=depthT,depthG=depthG):
    model = JTPropVAE(vocab, int(hidden_size), int(latent_size),int(depthT), int(depthG))
    dict_buffer = torch.load(model_path, map_location='cpu')
    model.load_state_dict(dict_buffer)
    model.eval()
    return model
def checksmile(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    smi = Chem.MolToSmiles(mol,kekuleSmiles=True,isomericSmiles=True)
    return smi
    
class LigandGenerator():
    def __init__(self,vocab=vocab,hidden_size=hidden_size,latent_size=latent_size,depthT=depthT,depthG=depthG):
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depthT = depthT
        self.depthG = depthG
        self.model = load_model(vocab,hidden_size,latent_size,depthT,depthG)
    
    def get_latent(self,ftrain=os.path.join('../fast_molopt/vae_model/latent_train_epoch_99-2'),dim0=0,dim1=1):
        df = pd.read_csv(ftrain,header=None)
        df1 = df.iloc[:,1:]
        scaler = StandardScaler().fit(df1)
        train_data = scaler.transform(df1)
        x_train = train_data[:,dim0]
        y_train = train_data[:,dim1]
        self.x_train = x_train
        self.y_train = y_train
        self.scaler = scaler
        
    def randomgen(self,num=1):
        gensmile = set()
        while len(gensmile) < num:
            z_tree = torch.randn(1, self.latent_size // 2)
            z_mol = torch.randn(1, self.latent_size  // 2)
            smi = self.model.decode(z_tree, z_mol, prob_decode=False)
            gensmile.add(smi)
        return gensmile
    
    def get_vector(self,smile=''):
        smi_target = [smile]
        tree_batch = [MolTree(smi) for smi in smi_target]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model.encode(jtenc_holder, mpn_holder)
        z_tree_mean = self.model.T_mean(tree_vecs)
        z_mol_mean = self.model.G_mean(mol_vecs)
        z_tree_log_var = -torch.abs(self.model.T_var(tree_vecs))
        z_mol_log_var = -torch.abs(self.model.G_var(mol_vecs))
        return z_tree_mean,z_mol_mean,z_tree_log_var,z_mol_log_var
    
    def get_lfs_from_smi(self,smile):
        z_tree_mean,z_mol_mean,z_tree_log_var,z_mol_log_var = self.get_vector(smile)
        lfs_pred = self.model.propNN(torch.cat((z_tree_mean,z_mol_mean),dim=1))
        lfs_pred = torch.clamp(lfs_pred,min=0,max=1)
        lfs = lfs_pred.item()
        return lfs
    
    def gen_from_target_withoutprop(self,target_smile,numsmiles=5,step_size=0.01):
        if not target_smile:
            raise ValueError('Target smile not defined! Change to random seed')
        decode_smiles_set = set()
        target_smile_cheeck = checksmile(target_smile)
        z_tree_mean,z_mol_mean,tree_var,mol_var = self.get_vector(target_smile)
        count = 0
        while numsmiles >= len(decode_smiles_set):
            epsilon_tree = create_var(torch.randn_like(z_tree_mean))
            epsilon_mol = create_var(torch.randn_like(z_mol_mean))
            z_tree_mean_new = z_tree_mean + torch.exp(tree_var / 2) * epsilon_tree * step_size
            z_mol_mean_new = z_mol_mean + torch.exp(mol_var / 2) * epsilon_mol * step_size
            smi = self.model.decode(z_tree_mean_new, z_mol_mean_new, prob_decode=False)
            count += 1
            if smi not in decode_smiles_set and smi != target_smile_cheeck:
                decode_smiles_set.add(smi)
            if count > numsmiles :
                step_size += 0.01
                count = 0
        return decode_smiles_set
    
    def scatter_plot(self,df,smi,p,dim1=0, dim2=1):
        mol = Chem.MolFromSmiles(smi)
        img = Draw.MolsToGridImage([mol],molsPerRow=1)
        png = img.data
        with open(os.path.join('../','data','%s.png'%smi),'wb+') as outf:
            outf.write(png)
        image = Image.open(os.path.join('../','data','%s.png'%smi))
        fig, axes = plt.subplots(1,2, figsize=(8, 6))  # (rows, columns, figsize)
        self.get_latent()
        xmin, xmax = -5, 5
        ymin, ymax = -5, 5
        xmajor, xminor = 2.5, 1.25
        ymajor, yminor = 2.5, 1.25
        df = pd.DataFrame(df)
        train_data = self.scaler.transform(df)
    
        xlabel = 'Z(%s)' % dim1
        ylabel = 'Z(%s)' % dim2
        axes[0].set_position([0.1, 0.1,0.8, 1.0])  # [left, bottom, width, height]
        axes[1].set_position([1, 0.5, 0.3, 0.4])  # [left, bottom, width, height]
        axes[1].imshow(image)  # 第二個子圖的資料
        axes[1].text(0.5, 1.0, '%.2f' %(p), horizontalalignment='center', verticalalignment='bottom', transform=axes[1].transAxes, fontsize=16)
        # for i in ['top', 'bottom', 'left', 'right']:
        #     axes[1].spines[i].set_visible(0)
        axes[1].axis('off')
        axes[0].scatter(self.x_train, self.y_train, c='black', s=0.5, edgecolors='black', alpha=0.2)
        axes[0].scatter(train_data[0][dim1],train_data[0][dim2],c='orange', s=30, edgecolors='black', alpha=1)

        axes[0].xaxis.set_major_locator(ticker.MultipleLocator(xmajor))
        axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
        axes[0].yaxis.set_major_locator(ticker.MultipleLocator(ymajor))
        axes[0].yaxis.set_minor_locator(ticker.MultipleLocator(yminor))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        axes[0].set_xlim(xmin, xmax)
        axes[0].set_ylim(ymin, ymax)
        axes[0].set_xlabel(xlabel, fontsize=16)
        axes[0].set_ylabel(ylabel, fontsize=16)
        plt.show()
        
    def LFS_optimization(self,LFS_target,inputsmile='',step_size=0.02,sign=-1,max_cycle=30,train_file='../data/latent_train_epoch_99-2.csv'):
        print('Running optimizaiotn...')
        cos = CosineSimilarity(dim=1)
        df = pd.read_csv(train_file,header=None)
        zs_ref = torch.Tensor(df.iloc[:,1:].values)
        LFS_target = check_input(LFS_target)
        inputsmicheck = checksmile(inputsmile)
        if 0 <= LFS_target <= 1:
            flag = True
            while flag:
                smis,zs,ps = [],[],[]
                count,ploss = 0,1.0
                ploss_threshold = 0.05
                # lfs = self.get_lfs_from_smi(inputsmile)
                t = 0
                # if abs(lfs - LFS_target) > 0.3:
                #     print('\nWarning! Input smile lfs is %.3f, but target lfs is %s.\nPredict lfs might not be trust-worthy!'%(lfs,LFS_target))
                #     checkpoint = input('Continue?[y/n]\n')
                #     if checkpoint.lower() == 'n':
                #         flag = False
                #         break
                while ploss > ploss_threshold and not math.isnan(ploss):
                    if count == 0:
                        z_tree_mean,z_mol_mean,tree_var,mol_var = self.get_vector(inputsmile)
                        epsilon_tree = create_var(torch.randn_like(tree_var))
                        epsilon_mol = create_var(torch.randn_like(mol_var))
                        lfs = self.model.propNN(torch.cat((z_tree_mean, z_mol_mean),dim=1))
                        lfs = torch.clamp(lfs,min=0,max=1).item()
                        delta_tree = torch.exp(tree_var / 2) * epsilon_tree * step_size
                        delta_mol = torch.exp(mol_var / 2) * epsilon_mol * step_size
                        z_tree_mean_new = z_tree_mean + delta_tree
                        z_mol_mean_new = z_mol_mean + delta_mol
                        count += 1
                    lfs_new = self.model.propNN(torch.cat((z_tree_mean_new, z_mol_mean_new),dim=1))
                    lfs_new = torch.clamp(lfs_new,min=0,max=1).item()
                    ploss = abs(lfs_new - LFS_target)
                    # print('LFS value is %s. Ploss is %s.' %(lfs_new, ploss))
                    # if ploss > 0.5:
                    #     count = 0
                    #     z_tree_mean,z_mol_mean,tree_var,mol_var = self.get_vector(inputsmile)
                    #     zs,ps = [],[]
                    #     continue
                    # if count == 10:
                    #     if ploss > abs(lfs - LFS_target):
                    #         sign = 1
                    #         print('Switching sign.')
                    delta_tree = sign * step_size * 2 * (lfs_new - LFS_target) * (lfs_new - lfs) / delta_tree / torch.sqrt(torch.Tensor([t+1])) 
                    delta_mol = sign * step_size * 2 * (lfs_new - LFS_target) * (lfs_new - lfs) / delta_mol / torch.sqrt(torch.Tensor([t+1])) 
                    # delta_tree = sign * step_size * ((lfs_new - lfs) / (z_tree_mean_new - z_tree_mean) * (1 + 2 * (lfs_new - LFS_target)) + 2 * (lfs_new - LFS_target) * (lfs - LFS_target) / (z_tree_mean_new - z_tree_mean)) # / torch.sqrt(torch.Tensor([t+1])) 
                    # delta_mol = sign * step_size * ((lfs_new - lfs) / (z_mol_mean_new - z_mol_mean) * (1 + 2 * (lfs_new - LFS_target)) + 2 * (lfs_new - LFS_target) * (lfs - LFS_target) / (z_mol_mean_new - z_mol_mean)) # / torch.sqrt(torch.Tensor([t+1])) 
                    # print(delta_tree[0])
                    lfs = lfs_new
                    z_tree_mean = (z_tree_mean_new)
                    z_mol_mean = (z_mol_mean_new)
                    z_tree_mean_new = z_tree_mean + delta_tree
                    z_mol_mean_new = z_mol_mean + delta_mol
                    t += 1
                    count += 1
                    similarity = cos(torch.cat((z_tree_mean,z_mol_mean),dim=1), zs_ref).max().item()
                    if 0.5 < similarity < 1:
                        # print('Similarity is %s.' %(similarity))
                        zs.append((z_tree_mean,z_mol_mean))
                        ps.append(lfs)
                    if len(ps) > max_cycle or math.isnan(ploss):
                        step_size *= 2
                        break
                if (ps[-1] - LFS_target) < ploss_threshold:
                    print('Start decoding...')
                    smis = [self.model.decode(*z, prob_decode=False) for z in zs]
                    smis_uniq = []
                    idxes = []
                    for i, smi in enumerate(smis[::-1]):
                        if smi not in smis_uniq:
                            smis_uniq.append(smi)
                            idxes.append(len(smis)-1-i)
                    if len(smis_uniq) > 1 and inputsmile not in smis_uniq:
                        # yield (smi,torch.cat((z_tree_mean,z_mol_mean),dim=1).tolist())
                        zs = [torch.cat(z,dim=1).tolist() for z in zs]
                        zs = [zs[idx] for idx in idxes[::-1]]
                        smis = smis_uniq[::-1]
                        ps = [ps[idx] for idx in idxes[::-1]]
                        return smis, zs, ps
                    elif inputsmile in smis_uniq:
                        print('Warning! Input smiles is the output smiles!')
                    elif None in smis_uniq:
                        print('Output smiles failure')
                    else:
                        print('There are fewer than 2 molecules found. Restarting...')
                else:
                    print('The error of the last LFS value is %.2f, which is larger than the threshold 0.05. Restarting...' %(ps[-1] - LFS_target))
        else:
            raise ValueError('target LFS must between 0~1 !')

if __name__ == '__main__':
    generator = LigandGenerator()
    lfs_optimizaiotn = generator.LFS_optimization(0.3,'c1ccccc1')
    for opt in lfs_optimizaiotn:
        print(opt)
