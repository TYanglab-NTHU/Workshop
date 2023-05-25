import sys,os
from rdkit import Chem
import torch
sys.path.append('../')
from fast_jtnn.vocab import *
from fast_jtnn.nnutils import create_var
from fast_jtnn.datautils import tensorize
from fast_jtnn.mol_tree import MolTree
from fast_jtnn.jtprop_vae import JTPropVAE
import pandas as pd
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
        
    def randomgen(self,num=1):
        gensmile = set()
        while len(gensmile) < num:
            z_tree = torch.randn(1, self.latent_size // 2)
            z_mol = torch.randn(1, self.latent_size  // 2)
            smi = self.model.decode(z_tree, z_mol, prob_decode=False)
            gensmile.add(smi)
        return gensmile
    
    def gen_from_target_withoutprop(self,target_smile,numsmiles=5,step_size=0.01):
        if not target_smile:
            raise ValueError('Target smile not defined! Change to random seed')
        decode_smiles_set = set()
        # decode_smiles_set.add(target_smile)
        tree_batch = [MolTree(target_smile)]
        target_smile_cheeck = checksmile(target_smile)
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model.encode(jtenc_holder, mpn_holder)
        z_tree_mean = self.model.T_mean(tree_vecs)
        z_mol_mean = self.model.G_mean(mol_vecs)
        tree_var = -torch.abs(self.model.T_var(tree_vecs))
        mol_var = -torch.abs(self.model.G_var(mol_vecs))
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
    
def get_lfs_from_smi(self,smile):
    self.get_vector(smile)
    # z_mol_log_var = -torch.abs(model.G_var(mol_vecs))
    lfs = self.model.propNN(torch.cat((self.z_tree_mean,self.z_mol_mean),dim=1))
    lfs = lfs.item()
    return lfs

# def gen_target_smi(self,LFS_target,smile='',step_size=0.1,sign=1):
#     LFS_target = check_input(LFS_target)
#     if 0 <= LFS_target <= 1:
#         flag = True
#         while flag:
#             smis,zs,ps = [],[],[]
#             smis_,pro = [],[]
#             count,ploss = 0,1
#             lfs = self.get_lfs_from_smi(smile)
#             if lfs - LFS_target > 0.3:
#                 print('\nWarning! Input smile lfs is %.3f, but target lfs is %s.\nPredict lfs might unconfident!'%(lfs.item(),LFS_target))
#                 checkpoint = input('Continue?[y/n]\n')
#                 if checkpoint.lower() == 'n':
#                     flag = False
#                     break
#             while ploss > 0.1:
#                 if count == 0:
#                     epsilon_tree = create_var(torch.randn_like(self.z_tree_mean))
#                     epsilon_mol = create_var(torch.randn_like(self.z_mol_mean))
#                     z_tree_mean_new = self.z_tree_mean + torch.exp(self.z_tree_log_var / 2) * epsilon_tree * step_size
#                     z_mol_mean_new = self.z_mol_mean + torch.exp(self.z_mol_log_var / 2) * epsilon_mol * step_size
#                     z_tree_mean = self.z_tree_mean
#                     z_mol_mean = self.z_mol_mean
#                     count += 1
#                 lfs_new = self.model.propNN(torch.cat((z_tree_mean_new, z_mol_mean_new),dim=1))
#                 ploss = abs(lfs_new.item() - LFS_target)
#                 print(ploss)
#                 if ploss > 1:
#                     count = 0
#                     self.get_lfs_from_smi(smile)
#                     zs,ps = [],[]
#                     continue
#                 delta_tree = sign * step_size * (lfs_new - lfs)/(z_tree_mean_new - z_tree_mean)
#                 delta_mol = sign * step_size * (lfs_new - lfs)/(z_mol_mean_new - z_mol_mean)
#                 lfs = lfs_new
#                 z_tree_mean = (z_tree_mean_new)
#                 z_mol_mean = (z_mol_mean_new)
#                 z_tree_mean_new = z_tree_mean + delta_tree
#                 z_mol_mean_new = z_mol_mean + delta_mol
#             zs.append([z_tree_mean,z_mol_mean]) 
#             ps.append(lfs_new)
#             decode_loss = ''
#             smis = [self.model.decode(*z, prob_decode=False) for z in zs]
#             for i,j in zip(smis,ps):
#                 if i != smile:
#                     smis_.append(i)
#                     pro.append(j.item())
#                     decode_loss = abs(self.get_lfs_from_smi(i) - j.item())
#             if smis_ != []:
#                 if decode_loss < 0.2:
#                     flag = False
                        
#         return smis_, pro, decode_loss
#     else:
#         raise ValueError('target LFS must between 0~1 !')
    

# if __name__ == '__main__':
#     model = LFSgenerator()
#     smis = ['C#N','CC#N']
#     # lfs = model.get_lfs_from_smi('C#N')
#     tree_batch = []
#     for smi in smis:
#         mol_tree = MolTree(smi)
#         mol_tree.recover()
#         mol_tree.assemble()
#         for node in mol_tree.nodes:
#             if node.label not in node.cands:
#                 node.cands.append(node.label)
#         del mol_tree.mol
#         for node in mol_tree.nodes:
#             del node.mol
#         tree_batch.append(mol_tree)
#     model.restore()
    
#     model_ = model.model
#     model_.cuda()

#     mol_batch = datautils.tensorize(tree_batch,model.vocab)
#     x_batch, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = mol_batch
#     x_tree_vecs, tree_message, x_mol_vecs = model_.encode(x_jtenc_holder,x_mpn_holder)
#     z_tree_vecs, tree_kl = model_.rsample(x_tree_vecs, model_.T_mean, model_.T_var)
#     z_mol_vecs, mol_kl = model_.rsample(x_mol_vecs, model_.G_mean, model_.G_var)
    
    
    
#     assm_loss, assm_acc = model_.assm(x_batch, x_jtmpn_holder, z_mol_vecs, tree_message)
#     jtmpn_holder,batch_idx = x_jtmpn_holder
#     fatoms,fbonds,agraph,bgraph,scope = jtmpn_holder
#     batch_idx = create_var(batch_idx)
#     cands_vecs = model_.jtmpn(fatoms,fbonds,agraph,bgraph,scope,tree_message)
#     x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)

#     x_mol_vecs = model_.A_assm(z_mol_vecs) #bilinear

    


#     cand_vecs = model_.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)


