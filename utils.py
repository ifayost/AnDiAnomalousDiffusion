import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Series:
    def __init__(self):
        self.n = 0
        self.series = {'1D':None, '2D':None, '3D':None}
        self.tmax = {'1D':0, '2D':0, '3D':0}
        self.labels = {'1D':None, '2D':None, '3D':None}
        self.task = '0'
    
    def __len__(self):
        return self.n
        
        
    def read(self, PATH):
        self.task = PATH[-5]
        series = pd.read_csv(PATH, header=None, sep="\n")
        series = series[0].str.split(';')
        dim_idx = series.map(lambda x: int(float(x[0])))
        labels = PATH[:-9]+'ref'+self.task+'.txt'
        labels = pd.read_csv(os.path.join(labels), header=None, sep=';').drop(0, axis=1)
        
        self.n = len(series)
        for i, dim in enumerate(['1D', '2D', '3D']):
            self.labels[dim] = labels[dim_idx == i+1]
            self.tmax[dim] = max(series[dim_idx == i+1].map(lambda x:len(x[1:])))
            if i == 0:
                self.series[dim] = series[dim_idx == i+1].map(lambda x:np.array(x[1:], dtype='float64'))
            else:
                self.series[dim] = series[dim_idx == i+1].map(lambda x:np.array(x[1:], dtype='float64').reshape(-1, i+1, order='F'))
    
    def differentiate(self, dim, d, thres):
        names = list(self.series.keys())
        
        def get_weight_ffd(d, thres, lim):
            w, k = [1.], 1
            ctr = 0
            while True:
                w_ = -w[-1] / k * (d - k + 1)
                if abs(w_) < thres:
                    break
                w.append(w_)
                k += 1
                ctr += 1
                if ctr == lim - 1:
                    break
            w = np.array(w[::-1]).reshape(-1, 1)
            return w
        
        w = get_weight_ffd(d, thres, self.tmax[names[dim-1]])

        def frac_diff_ffd(x, d, thres=1e-5):
            width = len(w) - 1
            output = []
            for i in range(width, len(x)):
                output.append(np.dot(w.T, x[i - width:i + 1])[0])
            return np.array(output)

        def function(serie):
            if dim == 1:
                return frac_diff_ffd(serie, d=d, thres=thres)
            elif dim == 2:
                x = frac_diff_ffd(serie[:,0], d=d, thres=thres).reshape(-1, 1)
                y = frac_diff_ffd(serie[:,1], d=d, thres=thres).reshape(-1, 1)
                return np.concatenate([x,y], axis=1)
            elif dim == 3:
                x = frac_diff_ffd(serie[:,0], d=d, thres=thres).reshape(-1, 1)
                y = frac_diff_ffd(serie[:,1], d=d, thres=thres).reshape(-1, 1)
                y = frac_diff_ffd(serie[:,2], d=d, thres=thres).reshape(-1, 1)
                return np.concatenate([x,y,z], axis=1)
                
        self.series[names[dim-1]] = self.series[names[dim-1]].map(function)
        self.tmax[names[dim-1]] = max(self.series[names[dim-1]].map(lambda x:len(x[1:])))*dim+1
        
    def get(self, idx, dim):
        if dim == 1:
            serie = np.zeros((len(idx), self.tmax['1D']), dtype='float64')
            for i, j in enumerate(self.series['1D'].iloc[idx]):
                serie[i, :len(j)] += j 
        elif dim == 2:
            serie = np.zeros((len(idx), self.tmax['2D']//2, 2), dtype='float64')
            for i, j in enumerate(self.series['2D'].iloc[idx]):
                serie[i, :j.shape[0], :] += j
        elif dim == 3:
            serie = np.zeros((len(idx), self.tmax['3D']//3, 3), dtype='float64')
            for i, j in enumerate(self.series['3D'].iloc[idx]):
                serie[i, :j.shape[0], :] += j
        else:
            print("Wrong dim number")
            
        dim = list(self.labels.keys())[dim-1]
        if self.task == '1':
            label = self.labels[dim].iloc[idx].values.reshape(-1)
        elif self.task == '2':
            label = self.labels[dim].iloc[idx].astype(int).values.reshape(-1)
        elif self.task == '3':
            label = self.labels[dim].iloc[idx].values
            
        return serie, label

# Plot for a given serie the ADF and the corrleation of the differentiated serie
# as a function of the differentation degree d.

def plot_ADF(serie, thres):
    stats = []
    serie = serie[serie!=0]
    for d in np.linspace(0,1,11):
        snorm = frac_diff_ffd(serie, d, thres=thres)
        serie = serie[snorm!=0]
        snorm = snorm[snorm!=0]
        corr = np.corrcoef(serie, snorm)[0,1]
        snorm = adfuller(snorm,maxlag=1,regression='c',autolag=None)
        stats.append(list(snorm[:4])+[snorm[4]['5%']]+[corr])
    stats = np.array(stats)
    fig, ax1 = plt.subplots(dpi=100, figsize=(5,3))
    colors = ["#6c71c4", "#2aa198", "#d33682"]
    ax1.plot(np.linspace(0,1,11), stats[:,0], color=colors[0])
    ax1.plot(np.linspace(0,1,11), stats[:,4], linestyle='--', color=colors[2])
    ax1.set_ylabel('ADF', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(0,1,11), stats[:,5], color=colors[1])
    ax2.set_ylabel('Correlation', color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
# Function to eval predicted alpha coefficients

def test_analysis_alpha(pred,ytest,algo = "Raw RF",thres=0.3):
    """
    pred = predicted alpha values
    ytest = real alpha values
    algo = algorithm description (only for plot titles)
    thres = threshold to separate Good-Bad residuals
    """
    residuals = np.abs(pred-ytest)
    print("MAE = {:.4f} \n".format(residuals.mean()))
    
    bad = residuals > thres
    good = residuals <= thres
    
    plt.figure(figsize=(10,7))
    
    ax1 = plt.subplot(2,2,1)
    ax1.hist(ytest,bins=20,label="real")
    ax1.hist(pred,bins=20,alpha=0.6,label="predicted")
    ax1.legend()
    ax1.set_xlabel("$\\alpha$",size=12)
    ax1.set_title("Predictions distro - {}".format(algo))
    
    ax2 = plt.subplot(2,2,2)
    ax2.hist(np.abs(pred-ytest),bins=40)
    ax2.set_xlabel("$|predicted-real|$",size=12)
    ax2.set_title("Residuals distro - {}".format(algo))
    
    ax3 = plt.subplot(2,2,3)
    ax3.scatter(pred,ytest,s=1,c="blue")
    ax3.plot([0,1,2],c="red",linestyle="dashed")
    ax3.set_xlabel("predicted",size=12)
    ax3.set_ylabel("real",size=12)
    ax3.set_title("real vs predicted $\\alpha$")
    
    ax4 = plt.subplot(2,2,4)
    ax4.hist(ytest[bad],color="red",bins=20,label="$residual > {}$".format(thres),alpha=0.4)
    ax4.hist(ytest[good],color="green",bins=20,label="$residual \\leq {}$".format(thres),alpha=0.4)
    ax4.set_xlabel("real $\\alpha$",size=12)
    ax4.set_title("Good-Bad residuals distro")
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
