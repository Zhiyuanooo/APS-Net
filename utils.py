import os
import torch
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import pdb
from torchvision import transforms
from PIL import Image
import time
from seg_losses import *
from matrics import *
from sklearn.metrics import roc_curve, auc
import csv


class Trainer(object):
    def __init__(self, model, optimizer, save_dir=None, save_freq=1):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.best_model_path = None
        self.best_accuracy = 0.0  # You can use this for best model selection based on accuracy
        
    def _loop(self, data_loader, ep, is_train=True):
        tensor2img = transforms.ToPILImage()
        loop_loss_class, correct, loop_loss_seg, loop_iou, loop_dice = [], [], [], [], []
        mode = 'train' if is_train else 'test'
        all_targets, all_predictions = [], []
        for data, tar, label in tqdm(data_loader):
            data, tar, label = data.to(self.device), tar.to(self.device), label.to(self.device)
            if is_train:
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
            out_class, out_seg = self.model(torch.cat([data, data, data], 1))
            n = out_seg.size(0)
            loss_class = F.cross_entropy(out_class, label)
 
            loss_seg = F.binary_cross_entropy(torch.sigmoid(out_seg.view(n, -1)), tar.view(n, -1)) + \
                        IOULoss(out_seg, tar)
            loss = 0.5 * loss_class + loss_seg

            loop_loss_class.append(loss_class.detach().cpu() / len(data_loader))
            loop_loss_seg.append(loss_seg.detach().cpu() / len(data_loader))
            out = (out_class.detach().cpu().data.max(1)[1] == label.detach().cpu().data).sum()
            correct.append(float(out) / len(data_loader.dataset))

            for j in range(n):
                loop_iou.append(iou_score(out_seg[j].detach().cpu(), tar[j].detach().cpu()))
                loop_dice.append(dice_coef(out_seg[j].detach().cpu(), tar[j].detach().cpu()))
                actual_class = label[j].detach().cpu().item()
                '''
                # Calculate TP, FP, TN, FN to compute SEN, SPC, PPV, NPV
                predicted_class = out_class[j].detach().cpu().argmax().item()
                

                tp = int(predicted_class == actual_class and actual_class == 1)
                fp = int(predicted_class != actual_class and actual_class == 1)  # True positives are when predicted_class = 1 and actual_class = 0
                tn = int(predicted_class == actual_class and actual_class == 0)  # True negatives are when predicted_class = 0 and actual_class = 0
                fn = int(predicted_class != actual_class and actual_class == 0)  # False negatives are when predicted_class = 0 and actual_class = 1


                sen = tp / (tp + fn + 1e-6)
                spc = tn / (tn + fp + 1e-6)
                ppv = tp / (tp + fp + 1e-6)
                npv = tn / (tn + fn + 1e-6)

                loop_sen.append(sen)
                loop_spc.append(spc)
                loop_ppv.append(ppv)
                loop_npv.append(npv)
                '''
                # Append predicted and actual class probabilities for ROC curve calculation
                predicted_prob = torch.softmax(out_class[j].detach().cpu(), dim=0).numpy()
                actual_prob = np.zeros(out_class.shape[1])
                actual_prob[actual_class] = 1.0
                all_targets.append(actual_prob)
                all_predictions.append(predicted_prob)
            
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Calculate the predicted segmentation mask as a probability map
            out_prob = torch.sigmoid(out_seg).detach().cpu().numpy()

        # Calculate ROC curve and AUC
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        fpr, tpr, _ = roc_curve(all_targets.ravel(), all_predictions.ravel())
        fnr = 1 - tpr
        tnr = 1 - fpr
        auc_score = auc(fpr, tpr)
        
        # Convert predicted probabilities to binary predictions based on a threshold (e.g., 0.5)
        threshold = 0.5
        binary_predictions = (all_predictions[:, 1] >= threshold).astype(int)

        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
        TP = np.sum((binary_predictions == 1) & (all_targets[:, 1] == 1))
        FP = np.sum((binary_predictions == 1) & (all_targets[:, 1] == 0))
        TN = np.sum((binary_predictions == 0) & (all_targets[:, 1] == 0))
        FN = np.sum((binary_predictions == 0) & (all_targets[:, 1] == 1))

        # Calculate Sensitivity (Sen), Specificity (Spe), Positive Predictive Value (Ppv), and Negative Predictive Value (Npv)
        SEN = TP / (TP + FN + 1e-6)
        SPE = TN / (TN + FP + 1e-6)
        PPV = TP / (TP + FP + 1e-6)
        NPV = TN / (TN + FN + 1e-6)
        F1 = 2 * (PPV * SEN) / (PPV + SEN + 1e-6)
        
        print(mode + '_clas: loss_class: {:.6f}, Acc: {:.6%}, Sen: {:.6f}, Spe: {:.6f}, Ppv: {:.6f}, Npv: {:.6f}, AUC: {:.6f} , F1: {:.6f} '.format(
        sum(loop_loss_class), sum(correct), SEN, SPE, PPV, NPV, auc_score, F1))
        
        print(mode + '_seg : loss_seg: {:.6f}, iou: {:.6%}, dice: {:.6f}'.format(sum(loop_loss_seg),sum(loop_iou)/len(loop_iou),sum(loop_dice)/len(loop_dice)))

        return sum(loop_loss_class), sum(correct), SEN, SPE, PPV, NPV, auc_score, F1

    def train(self, data_loader, ep):
        self.model.train()
        results = self._loop(data_loader, ep)
        return results

    def test(self, data_loader, ep):
        self.model.eval()
        results = self._loop(data_loader, ep, is_train=False)
        return results

    def loop(self, epochs, train_data, test_data, scheduler, save_freq):
        train_loader = torch.utils.data.DataLoader(train_data, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, pin_memory=True)
        f = open(self.save_dir + 'testlog.txt', 'w')
        f.close()
        for ep in range(1, epochs + 1):
            if scheduler is not None:
                scheduler.step()
            print('epoch {}'.format(ep))
            train_results = np.array(self.train(train_data, ep))
            test_results = np.array(self.test(test_data, ep))

            # Save the results for each epoch to a log file
            with open(self.save_dir + 'testlog.txt', 'a') as f:
                f.write(f"{ep}\t"
                        f"{test_results[0]:.6f}\t{test_results[1]:.6f}\t{test_results[2]:.6f}\t"
                        f"{test_results[3]:.6f}\t{test_results[4]:.6f}\t{test_results[5]:.6f}\t{test_results[6]:.6f}\t{test_results[7]:.6f}\n")
                
            if not ep % save_freq:
                self.save(ep)  # Save model weights
                
            if test_results[1] > self.best_accuracy:
                    self.best_accuracy = test_results[1]
                    self.best_model_path = self.save_dir + str(ep) + '_best'  +  '_models.pth'
                    torch.save(self.model.state_dict(), self.best_model_path)
                    # Save the predicted and true classification labels to CSV
                    self.save_predictions(test_data)
                    
        print('Training finished!')

    def save(self, epoch, **kwargs):
        if self.save_dir:
            name = self.save_dir + 'train' + str(epoch) + 'models.pth'
            torch.save(self.model.state_dict(), name)
            # torch.save(self.model, name)
            
    def get_best_model_path(self):
            return self.best_model_path
        
    def save_predictions(self, data_loader):
        self.model.eval()
        predictions = []
        true_labels = []
        file_names = []

        for data, tar, label in tqdm(data_loader):
            data, tar, label = data.to(self.device), tar.to(self.device), label.to(self.device)
            out_class, _ = self.model(torch.cat([data, data, data], 1))
            predicted_class = out_class.detach().cpu().argmax(dim=1).numpy()
            true_labels.extend(label.detach().cpu().numpy())
            predictions.extend(predicted_class.tolist())

            # 获取当前批次的文件名和标签，并将它们添加到列表中
            file_names.extend([fn for fn, _ in data_loader.dataset.imgs])
            # 注意：在此示例中忽略了 `_` 的元素（即标签），因为这里不需要它

        # 保存预测值、真实标签和文件名到CSV文件
        file_path = os.path.join(self.save_dir, 'predictions.csv')
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['File_Name', 'True_Label', 'Predicted_Label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for name, true, pred in zip(file_names, true_labels, predictions):
                writer.writerow({'File_Name': name, 'True_Label': true, 'Predicted_Label': pred})
                