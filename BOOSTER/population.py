# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:55:17 2017

@author: giguerf
"""
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
if 'C:/Users/giguerf/Documents' not in sys.path:
    sys.path.insert(0, 'C:/Users/giguerf/Documents')
from GitHub.COM import openbook as ob
#from GitHub.COM import plotstyle as style
from GitHub.COM import get_peak as gp

readdir = os.fspath('P:/BOOSTER/SAI/Y7')
savedir = os.fspath('P:/BOOSTER/Plots/Y7')

time, fulldata, *_ = ob.openbook(readdir)
#time, fulldata = time, fulldata
#%%

table = pd.read_excel('P:/BOOSTER/boostertable.xlsx', index_col = 0)
import xlrd
faro = pd.DataFrame(columns = ['LBS Top','LBS Bottom'])
for file in os.listdir('P:/BOOSTER/Faro/'):
    tcn = file[:-4]
    for sheetname in ['BELT MEASURE POSITION 14','BELT MEASURE POSITION 16']:
        try:
            raw = pd.read_excel('P:/BOOSTER/Faro/'+tcn+'.xls', sheetname=sheetname, index_col=4)
        except xlrd.XLRDError:
            try:
                raw = pd.read_excel('P:/BOOSTER/Faro/'+tcn+'.xls', sheetname=sheetname.title(), index_col=4)
            except xlrd.XLRDError:
                pass
            else:
                raw = raw['X']
                top = (raw.iloc[0]-raw).loc[['Top Lap Belt Left','Top Lap Belt Right']]
                bot = (raw.iloc[0]-raw).loc[['Bottom  Lap Belt left','Bottom  Lap Belt Right']]
                
                faro.loc[tcn+'_'+sheetname[-2:],'LBS Top'] = top.mean(axis=0)
                faro.loc[tcn+'_'+sheetname[-2:],'LBS Bottom'] = bot.mean(axis=0)
        else:
            raw = raw['X']
            top = (raw.iloc[0]-raw).loc[['Top Lap Belt Left','Top Lap Belt Right']]
            bot = (raw.iloc[0]-raw).loc[['Bottom  Lap Belt left','Bottom  Lap Belt Right']]
            
            faro.loc[tcn+'_'+sheetname[-2:],'LBS Top'] = top.mean(axis=0)
            faro.loc[tcn+'_'+sheetname[-2:],'LBS Bottom'] = bot.mean(axis=0)

faro = faro.apply(pd.to_numeric)                
temp = pd.merge(table, faro, how = 'outer', left_index = True, right_index = True)
tcnlist = list(table.index)

chlist = ['14CHST0000Y7ACXC', '14ILACLELOY7FOXB', '14ILACLEUPY7FOXB', '14ILACRILOY7FOXB', '14ILACRIUPY7FOXB', '14LUSP0000Y7FOXA', '14LUSP0000Y7FOZA', '14PELV0000Y7ACXA']
names = ['Chest', 'Iliac Lower Left', 'Iliac Upper Left', 'Iliac Lower Right', 'Iliac Upper Right', 'Lumbar X', 'Lumbar Z', 'Pelvis']
chname = dict(zip(chlist,names))

peakdata = {}
peaktable = pd.DataFrame()
for channel in fulldata:
    peakdata[channel] = pd.DataFrame(columns = [chname[channel]])
    for tcn in fulldata[channel].columns.intersection(tcnlist):
        if channel in ['14ILACRIUPY7FOXB','14ILACRILOY7FOXB','14ILACLEUPY7FOXB','14ILACLELOY7FOXB']:
            lim = 'max'
        else:
            lim = 'min'
        peakdata[channel].loc[tcn] = gp.get_peak(time, fulldata[channel][tcn], lim=lim).peak
    peaktable = pd.concat([peaktable, peakdata[channel]], axis=1)

t = pd.merge(temp, peaktable, how='outer', left_index=True, right_index=True)
t = t.rename(columns = {'LBS Bottom':'LBS'})
t = t[t['Type_Mannequin'] == 'HIII Enfant'].drop(['Location','Date_essai','Type_Mannequin','Seat','UAS'], axis=1)
#t = t.reset_index()

t.loc[(t.Position == 14) & (t.ModeleAuto == 'PRIUS'), 'Class'] = 'Lap'
t.loc[(t.Position == 16) & (t.ModeleAuto == 'PRIUS'), 'Class'] = 'Pelvis'
t.loc[(t.Position == 14) & (t.ModeleAuto != 'PRIUS'), 'Class'] = 'Pelvis'
t.loc[(t.Position == 16) & (t.ModeleAuto != 'PRIUS'), 'Class'] = 'Lap'

t = t.sort_values(['LBS'])
#t.to_excel('P:/BOOSTER/Plots/FullSampleData.xlsx')
t['Toward'] = -(t.loc[:,'Pelvis'] - t.loc[:,'Chest'])
t2 = pd.merge(t[t.Class == 'Lap'],t[t.Class == 'Pelvis'],how='outer',on='TCN',suffixes=[' Lap',' Pelvis'])
t2['Delta LBS'] = t2['LBS Lap'] - t2['LBS Pelvis']
t2 = t2.sort_values(['Delta LBS'])
t2 = t2.dropna(axis = 0, subset = ['Delta LBS'])
t2 = t2.set_index('TCN')
#%%

#tcns = ['TC12-004', 'TC16-129', 'TC16-132', 'TC18-105', 'TC14-503_2', 'TC14-503_4', 'TC14-503_5']
tcns = ['TC16-129', 'TC14-503_2', 'TC14-503_4', 'TC14-503_5', 'TC16-132', 'TC18-105', 'TC12-004']
tcnpos = dict(zip(list(range(7)),tcns))

tc14 = [s + '_14' for s in tcns]
tc16 = [s + '_16' for s in tcns]
tclap = [s if t.loc[s].Class == 'Lap' else r for s,r in zip(tc14,tc16)]
tcpelvis = [s if t.loc[s].Class == 'Pelvis' else r for s,r in zip(tc14,tc16)]
#%% Lumbar
plt.close('all')

plotdf = -peaktable

channels = ['Lumbar X', 'Lumbar Z']
plotlap = plotdf.loc[tclap, channels].transpose()
plotpelvis = plotdf.loc[tcpelvis, channels].transpose()
#    print(plot16)

fig, axs = plt.subplots(1,2,sharey=True)

plotlap.plot.barh(ax=axs[0])
plotpelvis.plot.barh(ax=axs[1])
##
#bars = axs[0].patches
#hatches = ''.join(h*len(plotlap) for h in ['','/','/','/','','',''])
#
#for bar, hatch in zip(bars, hatches):
#    bar.set_hatch(hatch)
##
lims = axs[1].get_xlim()
axs[0].set_xlim((lims[1],lims[0]))

axs[0].legend_.remove()
axs[1].legend_.remove()
axs[1].yaxis.set_ticks_position('none')


h, l = axs[1].get_legend_handles_labels()
#labels = l
#labels = tcns
labels = ['{} [{:.1f}]'.format(i, t2.loc[s[:-3],'Delta LBS']) for i,s in enumerate(l)]
lgd = axs[1].legend(h, labels, loc = 6, bbox_to_anchor=(1, 0.5))

plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.subplots_adjust(wspace=.0)

axs[0].text(0.05, 0.02, 'Belt Towards Lap', transform=axs[0].transAxes, horizontalalignment='left')
axs[1].text(0.95, 0.02, 'Belt Towards Pelvis', transform=axs[1].transAxes, horizontalalignment='right')

#plt.savefig('P:/BOOSTER/Plots/barchart_Lumbar.png', dpi = 300, bbox_inches="tight", bbox_extra_artist = lgd)


#%% Chest / Pelvis
plt.close('all')

fig = plt.figure()
ax = plt.gca()
df = t2.loc[tcns,['Toward Pelvis','Toward Lap']]
df.plot(kind='barh', width=0.75, ax=ax)

'''Change Labels?'''
#h = []
#labels = []
#for i,s in enumerate(tcns):
#    h.append(ax.plot(float('nan'),float('nan'))[0])
#    labels.append('{} [{:.1f}]'.format(i, t2.loc[s,'Delta LBS']))
##labels = ['{} [{:.1f}]'.format(i, t2.loc[s[:-3],'Delta LBS']) for i,s in enumerate(tcns)]
#fig.legend(h, labels, loc = 6, bbox_to_anchor=(1, 0.5))

plt.xticks(np.arange(-5,45,5))
plt.setp(ax.get_yticklabels(), visible=False)

ax.set_xlim((-5,45))
ax.set_title('In-Vehicle Paired Comparison')
ax.set_xlabel('Disparity between Chest and Pelvis in Peak Acceleration [g]\n(Chest $g_{x}$ - Pelvis $g_{x}$)')#\nWhen value is positive, the Pelvis Peak is greater in magnitude\n than the Chest Peak')
ax.set_ylabel('Paired tests')
h, l = ax.get_legend_handles_labels()
ax.legend([h[1][0],h[0][0]], ['Belt Away From Pelvis','Belt On Pelvis'], loc = 5, bbox_to_anchor=(1, 0.65))
plt.tight_layout()
ax.set_axisbelow(True)
ax.grid(color = '0.80')

#plt.savefig('P:/BOOSTER/Plots/barchart.png', dpi = 300, bbox_inches="tight")

tableout = t.loc[tclap+tcpelvis,['Class','LBS','Chest','Pelvis','Toward']]
tableout = tableout.rename(columns = {'Toward':'Chest -Pelvis'})
tableout = tableout.rename(columns = {'LBS':'Belt Score'})
#tableout.to_excel('P:/BOOSTER/Plots/barchart data2.xlsx')

#%% Scatter plots

plt.close('all')

scatterdf = t.loc[tclap+tcpelvis,['LBS','Chest','Pelvis','Toward']]
scatterdf = scatterdf.rename(columns = {'Toward':'Delta'})

#fig, axs = plt.subplots(2,2,sharex = 'row', sharey = 'row', figsize=(6.4*4,4.8*4))
#axs = axs.flatten()
plt.figure(figsize=(6.4*2,4.8*2))
axs = [plt.subplot(221),plt.subplot(222),plt.subplot(223)]

subs = dict(zip(['Chest','Pelvis','Delta'],axs))
fit = []
rr = []

for column in subs:
    ax = subs[column]
    
    scatterdf.plot.scatter('LBS', column, ax=ax, s=10)#, c=tableout['color'])
    m, b, r, p, sig = scipy.stats.linregress(scatterdf['LBS'], scatterdf[column])
    fit_fn = np.poly1d([m,b])
    ax.plot(scatterdf['LBS'], fit_fn(scatterdf['LBS']), color = (.60,.60,.60))
    
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.setp(ax.get_yticklabels(), visible=True)
    
    fit.append(fit_fn)
    rr.append(r)

axs[0].text(.650, .75, '%s\n$R^2$ = %s' % (fit[0], np.round(rr[0]**2,2)), rotation = 20,transform=axs[0].transAxes)
axs[0].set_xlabel('Distance Away from Pubis in X-Direction')
axs[0].set_ylabel('Chest Peak Acceleration [g]')

axs[1].text(.650, .5, '%s\n$R^2$ = %s' % (fit[1], np.round(rr[1]**2,2)), rotation = -10,transform=axs[1].transAxes)
axs[1].set_xlabel('Distance Away from Pubis in X-Direction')
axs[1].set_ylabel('Pelvis Peak Acceleration [g]')

axs[2].text(.650, .78, '%s\n$R^2$ = %s' % (fit[2], np.round(rr[2]**2,2)), rotation = 35,transform=axs[2].transAxes)
axs[2].set_xlabel('Distance Away from Pubis in X-Direction')
axs[2].set_ylabel('Disparity between Chest and Pelvis Peak Accelerations [g]')

##h, l = ax.get_legend_handles_labels()
##ax.legend(h, l, loc = 5, bbox_to_anchor=(1, 0.45))
##style.legend(ax, loc = 'lower right')
#
#plt.savefig('P:/BOOSTER/Plots/scatter.png', dpi = 300, bbox_inches="tight")

#%% Belt Score Distribution
plt.close('all')
from GitHub.COM import plotstyle as style

t = t.rename(columns = {'Toward':'Difference'})
groupA = t.loc[t.Group == 'A',['Group','LBS','Chest','Pelvis','Difference']].dropna(axis=0)
#groupB = t.loc[t.Group == 'B',['Group','LBS','Chest','Pelvis','Difference']].dropna(axis=0)
groupC = t.loc[t.Group == 'C',['Group','LBS','Chest','Pelvis','Difference']].dropna(axis=0)
groups = pd.concat([groupA,groupC],axis=0)
position = pd.DataFrame(list(zip([0,1],['A','C'])), columns=['X','Group'])

#plotdata = pd.merge(position, groups, on='Group', how='outer')
plotdata = groups

axs = style.subplots(1, 2, sharey='all', visible=False)

#ax1 = plotdata.plot.scatter('X','LBS')
#for column in ['LBS','Chest','Pelvis','Difference']:
#plotdata.boxplot(ax=ax, column=['Chest','Pelvis','Difference'], by='Group', grid=False)
plotdata[plotdata.Group == 'A'].boxplot(ax=axs[0], column=['Chest','Pelvis','Difference'], grid=False)
plotdata[plotdata.Group == 'C'].boxplot(ax=axs[1], column=['Chest','Pelvis','Difference'], grid=False)

axs[0].set_xlabel('Group \'A\'')
axs[1].set_xlabel('Group \'C\'')
axs[0].set_ylabel('Acceleration X [g]')
plt.subplots_adjust(wspace=.0)
#%%
    
groupA = t.loc[t.Group == 'A',['Group','LBS','Chest','Pelvis','Difference']]
#groupB = t.loc[t.Group == 'B',['Group','LBS','Chest','Pelvis','Difference']]
groupC = t.loc[t.Group == 'C',['Group','LBS','Chest','Pelvis','Difference']]

statsA = pd.DataFrame([groupA.mean(axis=0),groupA.std(axis=0)], index=['Average','Standard Deviation'])
#statsB = pd.DataFrame([groupB.mean(axis=0),groupB.std(axis=0)], index=['Average','Standard Deviation'])
statsC = pd.DataFrame([groupC.mean(axis=0),groupC.std(axis=0)], index=['Average','Standard Deviation'])
#statstable = pd.concat([statsA,statsB,statsC], axis=0)
statstable = pd.concat([statsA,statsC], axis=0)
#statstable['Group'] = ['A','A','B','B','C','C']
statstable['Group'] = ['A','A','C','C']
#statstable = statstable.set_index('Group')
statstable = statstable.rename(columns = {'LBS':'Belt Score'})
#statstable.to_excel('P:/BOOSTER/Plots/GroupsAC_Table.xlsx')
