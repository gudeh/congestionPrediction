import os
import sys
import shutil
import csv
import re # handle type to category names AND_X1
import time
import math #ceil()
import random
import numpy as np
import pandas as pd
import statistics
from pandas.api.types import is_numeric_dtype
import seaborn as sns
from pathlib import Path #Reading CSV files
from random import shuffle #shuffle train/valid/test
from itertools import combinations # ablation to test multiple features
import scipy
from scipy import stats
from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.colorbar as mcb
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import   SummaryWriter #Graphical visualization
from torch.utils.data import DataLoader, RandomSampler

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.dataloading import DataLoader
from dgl.data.ppi import PPIDataset #TODO remove
import dgl.nn as dglnn
import networkx as nx #drawing graphs

from sklearn.metrics import r2_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
from torchmetrics.regression import KendallRankCorrCoef #Same score as congestionNet

validFeatures = [ 'closeness',  'betweenness' , 'eigen', 'pageRank', 'inDegree', 'outDegree', 'type', 'area', 'input_pins', 'output_pins' ] # , 'harmonic',, 'load',  'percolation'
#validFeatures = [ 'betweenness', 'closeness', 'input_pins', 'output_pins' ]
externalCentralities = [ 'closeness', 'harmonic', 'betweenness', 'load',  'percolation' ]
globalNormMode = 'oneZero' #'meanStd' #'oneZero'                         

mainMaxIter      = 1
runSetup         = 1
FULLTRAIN        = False
DOKFOLD          = True
FIXEDSPLIT       = True
num_folds        = 2
MANUALABLATION   = True

stdCellFeats = [ 'type', 'area', 'input_pins', 'output_pins' ]
fullAblationCombs = [ 'area', 'input_pins', 'output_pins', 'inDegree', 'outDegree', 'type', 'eigen', 'pageRank' , 'closeness', 'betweenness' ]
feat2d = 'feat' 

labelName =  'routingHeat'
secondLabel = 'placementHeat'

LOADSECONDDS    = False
MIXEDTEST       = False
FUSIONDS        = False

maxEpochs = 800
minEpochs = 200
# maxEpochs = 10
# minEpochs = 10

useEarlyStop = True
step      = 0.005
improvement_threshold = 0.000001 
patience = 45  # Number of epochs without improvement to stop training
accumulation_steps = 4

DOLEARN         = True
REMOVEFAKERAM   = True

DRAWOUTPUTS     = True # draw pred versus label after learn?
DRAWGRAPHDATA   = False # draws histograms and correlation matrix
DRAWHEATCENTR   = False

DEBUG           = 0 #1 for evaluation inside train
CUDA            = True
SKIPFINALEVAL   = False #TODO True
SELF_LOOP       = True
COLAB           = False


if CUDA:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
    
if CUDA:
    print( "torch.cuda.is_available():", torch.cuda.is_available() )
    print( "torch.cuda.device_count():", torch.cuda.device_count() )
    print( "torch.cuda.device(0):", torch.cuda.device(0) )
    print( "torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0) )
print( "dgl.__version_:", dgl.__version__ )


def getF1( tp, tn, fp, fn ):
    precision = np.divide( tp, np.add( tp, fp ) )
    recall = np.divide( tp, np.add( tp, fn ) )
    f1_score = np.multiply( 2, np.divide( np.multiply( precision, recall ), np.add( precision, recall ) ) )
    return f1_score

def getPN( label, predict, threshold ):
    tensor_min = predict.min()
    tensor_max = predict.max()
    predict = ( predict - tensor_min) / (tensor_max - tensor_min)
    
    mask1 = label   >= threshold
    mask2 = predict >= threshold
    tp = torch.logical_and(  mask1,  mask2 ).sum().item()
    tn = torch.logical_and( ~mask1, ~mask2 ).sum().item()
    fp = torch.logical_and( ~mask1,  mask2 ).sum().item()
    fn = torch.logical_and(  mask1, ~mask2 ).sum().item()
    return tp, tn, fp, fn

def dynamicConcatenate( featTensor, tensor2 ):
    if feat2d in featTensor:
        if featTensor[ feat2d ].dim() == 1:
            ret = torch.cat( ( featTensor[ feat2d ].unsqueeze(1), tensor2.unsqueeze(1) ), dim = 1 )
        else:
            ret = torch.cat( ( featTensor[ feat2d ], tensor2.unsqueeze(1) ), dim = 1 )
    else:
        ret = tensor2
    #print( "ret:", ret )
    return ret


def process_and_write_csvs( paths ):
    categorical_columns = [ "type" ]
    pandasNormCols = [ labelName, 'area', 'input_pins', 'output_pins'  ]
    master_df = pd.DataFrame()
    for path in paths:
        csv_path = os.path.join( path, "gatesToHeatSTDfeatures.csv")
        if os.path.isfile( csv_path ):
            df = pd.read_csv( csv_path, index_col = 'id', dtype = { 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )
            master_df = pd.concat( [ master_df, df ], ignore_index = True )
    master_df = master_df[ master_df[ labelName ] > 0 ]
    if REMOVEFAKERAM:
        master_df.loc[ master_df[ 'type' ].str.contains( 'fakeram', case = False ), 'type' ] = "-1"
        master_df.loc[ master_df[ 'type' ] == '-1', 'area'] = 0
        master_df.loc[ master_df[ 'type' ] == '-1', 'input_pins'] = 0
        master_df.loc[ master_df[ 'type' ] == '-1', 'output_pins'] = 0
        # master_df.loc[ master_df[ 'type' ] == '-1', labelName] = 0
    # master_df.to_csv("master_df")
    
    min_values = master_df[ pandasNormCols ].replace( 0, np.nan ).min()
    max_values = master_df[ pandasNormCols ].max()
    print("min and max for normalization:\nmin:", min_values, "\nmax", max_values )
    
    categorical_mapping = {}
    for column in categorical_columns:
        categorical_mapping[ column ] = master_df[ column ].astype( 'category' ).cat.categories.tolist()
    # print( "categorical_mapping:\n" )#, categorical_mapping )
    # for key, value in categorical_mapping.items():
    #     print(f'{key}: {value}')

    for path in paths:
        csv_path = os.path.join( path, "gatesToHeatSTDfeatures.csv" )
        if os.path.isfile( csv_path ):
            df = pd.read_csv( csv_path, index_col = 'id', dtype = { 'name':'string', 'type':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )
            for column, categories in categorical_mapping.items():                
                df[ column ] = pd.Categorical( df[ column ], categories = categories ).codes

            # TODO check if this is ok
            df[ pandasNormCols ] = df[ pandasNormCols ].replace( 0, min_values )
            if globalNormMode == 'oneZero':
                df[ pandasNormCols ] = ( df[ pandasNormCols ] - min_values ) / ( max_values - min_values )
            if globalNormMode == 'meanStd':
                mean_values = df[ pandasNormCols ].mean()
                std_values =  df[ pandasNormCols ].std()
                df[ pandasNormCols ] = ( df[ pandasNormCols ] - mean_values) / std_values
                
            
            df.loc[ df[ 'type' ] == -1, labelName] = -1

            output_path = os.path.join( path, "preProcessedGatesToHeat.csv" )
            df.to_csv(output_path, index=True)


#TODO REMOVE THIS AFTER NEW PREPROCESS DONE
def preProcessData( listDir ):
    nameToCategory =  {}
    typeToNameCat =   {}
    typeToSize =      {}
    labelToStandard = {}
    graphs  =         {}
    labelsAux = pd.Series( name = labelName )
    #	regexp = re.compile( r'(^.*)\_X(\d?\d)$' ) #_X1 works ok for nangate
    regexp = re.compile( r'(^.*)\_?[x|xp](\d?\d)', re.IGNORECASE ) 
    for path in listDir:
        #print( "Circuit:",path )
        # gateToHeat = pd.read_csv( path / 'gatesToHeatSTDfeatures.csv', index_col = 'id', dtype = { 'type':'category' } )
        gateToHeat = pd.read_csv( path / 'gatesToHeatSTDfeatures.csv', index_col = 'id', dtype = { 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )                    
        labelsAux = pd.concat( [ labelsAux, gateToHeat[ labelName ] ], names = [ labelName ] )
        graphs[ path ] = gateToHeat#.append( gateToHeat )
        # for cellType in gateToHeat[ rawFeatName ]:
        for cellType in gateToHeat[ 'type' ]:
            #print( "cellType", type( cellType ), cellType )
            # pattern = r"(FILLCELL|TAPCELL|.*ff.*|.*clk.*|.*dlh.*|.*dll.*|.*tlat.*)"
            pattern = r"(FILLCELL|TAPCELL)"
            if not re.search(pattern, cellType, re.IGNORECASE):
                match = re.match( regexp, cellType )
                #print( "wanted cell type:", cellType )
                if match: 
                    if len( match.groups( ) ) == 2:
                        if cellType not in typeToNameCat:
                            typeToNameCat[ cellType ] = len( typeToNameCat )
                else:
                    print( "WARNING: Unexpected cell type:", cellType )    
                    typeToNameCat[ cellType ] = -1
                    typeToSize[ cellType ]    = -1
            else:
                #print( "Removing unwanted cell type:", cellType )
                typeToNameCat[ cellType ] = -1
                typeToSize[ cellType ]    = -1


    ##################### FOR GLOBAL NORMALIZATION ON HEAT VALUES ###########################
    df = labelsAux
    print( "df before remove -1:\n", type(df), "\n", df,"\n")
    #    df = ( df[ df < 0 ] )
#    df = df.loc[ df >= 0 ]
    df = df.loc[ df > 0 ] # trying to remove also 0 heat
    print( "df after remove -1:\n", df,"\n")
    
    dfMin = float( df.min() )
    dfMax = float( df.max() )
    mean  = df.mean()
    std   = df.std()
    median = df.median()
    p75 = df.quantile( 0.75 )  
    p25 = df.quantile( 0.25 )
    
    print( "dfMin:", dfMin, "dfMax:", dfMax )
    print( "mean:", mean, "std:", std )
    for key in df:
        #print( "label:",label,"key:",key )
        if key not in labelToStandard: # and key >= 0:
            #labelToStandard[ key ] = ( key - median ) / ( p75 - p25 ) # quantile
            labelToStandard[ key ] = ( key - dfMin ) / ( dfMax - dfMin ) # 0 to 1
            #labelToStandard[ key ] = ( key - mean ) / std  # z value (mean)
    print( "\n\n\labelToStandard:\n", sorted( labelToStandard.items() ), "size:", len( labelToStandard  ) )
    #######################################################################
    # print( "\n\n\nnameToCategory:\n", nameToCategory, "size:", len( nameToCategory  ) )
    # print( "\n\n\ntypeToNameCat:\n", typeToNameCat, "size:", len( typeToNameCat  ) )
    # print( "\n\n\ntypeToSize:\n", typeToSize, "size:", len( typeToSize  ) )

    for key, g in graphs.items():
        # g[ rawFeatName ]  = g[ rawFeatName  ].replace( typeToNameCat )
        g[ 'type' ]  = g[ 'type'  ].replace( typeToNameCat )
        g[ labelName ] = g[ labelName ].replace( labelToStandard )
        g.to_csv( key / 'preProcessedGatesToHeat.csv' )

def aggregateData( listDir, csvName ):
    aggregatedDF = pd.DataFrame()
    for path in listDir:
        inputData = pd.read_csv( path / csvName )
        # inputData = pd.read_csv( path / csvName, index_col = 'id', dtype = { 'type':'string', 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )                    
        #TODO: new features are not considered here
        # inputData = inputData[ [ rawFeatName, labelName, secondLabel ] ]
        inputData = inputData[ [ 'type', labelName, secondLabel ] ]
#        print( "inputData before concat:\n", inputData )
        aggregatedDF = pd.concat( [ aggregatedDF, inputData ] )
#    aggregatedDF.set_index( 'type' )
    return aggregatedDF

def writeDFrameData( listDir, csvName, outName ):
    with open( outName, 'w' ) as f:
        f.write( "IC name,#gates,#edges,"+labelName+"Min,"+labelName+"Max,"+secondLabel+"Min,"+secondLabel+"Max\n" )
        for path in listDir:
            inputData = pd.read_csv( path / csvName )
            # inputData = pd.read_csv( path / csvName, index_col = 'id', dtype = { 'type':'string', 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )                    
            edgesData = pd.read_csv( path / 'DGLedges.csv' )
            print( "ic name split:", str( path ).rsplit( '/' ) )
            icName = str( path ).rsplit( '/' )[-1]
            f.write( icName + "," + str( inputData.shape[0] ) + "," + str( edgesData.shape[0] ) + "," )
            f.write( str( ( inputData [ labelName ] ).min() ) + "," + str(inputData[ labelName ].max() ) + "," )
            f.write( str( inputData[ secondLabel ].min()) + "," + str(inputData[ secondLabel ].max()) + "\n" )	


class DataSetFromYosys( DGLDataset ):
    def __init__( self, listDir, ablationList ):
        self.graphPaths = []
        self.graphs = []
        self.names = []
        self.allNames = []
        self.ablationFeatures = ablationList
        self.namesOfFeatures = []

        for idx in range( len( listDir ) ):
            self.allNames.append( str( listDir[idx] ).rsplit( '/' )[-1] )
            print( self.allNames[idx],",", end="" )
    #        train, validate, test = np.split(files, [int(len(files)*0.8), int(len(files)*0.9)])

        self.graphPaths = listDir
        self.names      = self.allNames

        super().__init__( name='mydata_from_yosys' )

    def drawCorrelationPerGraph( self, fileName ):
        num_graphs = len( self.graphs )
        num_rows = int( np.sqrt( num_graphs ) )
        num_cols = int( np.ceil( num_graphs / num_rows ) )
        fig, axes = plt.subplots( num_rows, num_cols, figsize = ( 12, 8 ) )
        for i, graph in enumerate( self.graphs ):
            tensor = graph.ndata[ feat2d ]
            print( "feat :", graph.ndata[ feat2d ].shape, graph.ndata[ feat2d ] )
            print( "label:", graph.ndata[ labelName ].shape, graph.ndata[ labelName ] )
            reshape = graph.ndata[ labelName ].view( -1, 1 )
            print( "reshape:", reshape.shape, reshape )
            tensor = torch.cat( ( tensor, reshape ), 1 )
            print( "\n\ntensor to make matrix correlation:", tensor.shape, tensor )
            correlation_matrix = np.corrcoef( tensor, rowvar = False )
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]
            im = ax.imshow( correlation_matrix, cmap = 'viridis', interpolation = 'nearest', vmin=-1, vmax=1, )
            ax.set_title(  self.names[i] )
            ax.set_xticks( np.arange( len( self.namesOfFeatures ) +1 ) ) 
            ax.set_yticks( np.arange( len( self.namesOfFeatures ) +1 ) )
            ax.set_xticklabels( self.namesOfFeatures + [ labelName ], rotation = 30 )
            ax.set_yticklabels( self.namesOfFeatures + [ labelName ] )
            for j in range( len( self.namesOfFeatures ) +1 ):
                for k in range( len( self.namesOfFeatures ) +1 ):
                    ax.text(k, j, format( correlation_matrix[j, k], ".1f" ),
                            ha = "center", va = "center", color = "white" )
        fig.tight_layout()
        cbar = fig.colorbar( im, ax=axes.ravel().tolist() )
        plt.savefig( "corrMatrix-"+fileName+".png" )
        plt.close( 'all' )
        plt.clf()
        
    # def drawSingleCorrelationMatrix( self, fileName ):
    #     print( "Start: drawSingleCorrelationMatrix!" )
    #     num_graphs = len(self.graphs)
    #     all_data = []
    #     for graph in self.graphs:
    #         tensor = graph.ndata[feat2d]
    #         reshape = graph.ndata[labelName].view(-1, 1)
    #         tensor = torch.cat((tensor, reshape), 1)
    #         all_data.append(tensor)
    #     combined_data = torch.cat(all_data, 0)
    #     correlation_matrix = np.corrcoef(combined_data, rowvar=False)
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     im = ax.imshow(correlation_matrix, cmap='turbo', interpolation='nearest', vmin=-1, vmax=1)
    #     ax.set_title("Combined Correlation Matrix")
    #     ax.set_xticks(np.arange(len(self.namesOfFeatures) + 1))
    #     ax.set_yticks(np.arange(len(self.namesOfFeatures) + 1))
    #     ax.set_xticklabels(self.namesOfFeatures + [labelName], rotation=30)
    #     ax.set_yticklabels(self.namesOfFeatures + [labelName])        
    #     for j in range(len(self.namesOfFeatures) + 1):
    #         for k in range(len(self.namesOfFeatures) + 1):
    #             ax.text(k, j, format(correlation_matrix[j, k], ".1f"),
    #                     ha="center", va="center", color="white")
    #     fig.tight_layout()
    #     cbar = fig.colorbar(im)
    #     plt.savefig("corrMatrix-" + fileName + ".png")
    #     plt.close('all')
    #     plt.clf()
    #     print( "End: drawSingleCorrelationMatrix!" )

    def drawSingleCorrelationMatrix( self, fileName, dsName ):
        print("**********Start: drawSingleCorrelationMatrix!**********")
        num_graphs = len(self.graphs)
        all_data = []

        for graph in self.graphs:
            #print("graph.ndata:", graph.ndata)
            tensor = graph.ndata[feat2d]
            reshape = graph.ndata[labelName].view(-1, 1)
            tensor = torch.cat((tensor, reshape), 1)
            all_data.append(tensor)

        combined_data = torch.cat(all_data, 0)
        pearson_corr_matrix = np.corrcoef(combined_data, rowvar=False)
        spearman_corr_matrix, _ = spearmanr(combined_data, axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        im1 = axes[0].imshow(pearson_corr_matrix, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
        axes[0].set_title("Pearson Correlation Matrix")
        axes[0].set_xticks(np.arange(len(self.namesOfFeatures) + 1))
        axes[0].set_yticks(np.arange(len(self.namesOfFeatures) + 1))
        axes[0].set_xticklabels(self.namesOfFeatures + [labelName], rotation=30)
        axes[0].set_yticklabels(self.namesOfFeatures + [labelName])

        for j in range(len(self.namesOfFeatures) + 1):
            for k in range(len(self.namesOfFeatures) + 1):
                axes[0].text(k, j, format(pearson_corr_matrix[j, k], ".2f"),
                             ha="center", va="center", color="black")

        im2 = axes[1].imshow(spearman_corr_matrix, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
        axes[1].set_title("Spearman Correlation Matrix")
        axes[1].set_xticks(np.arange(len(self.namesOfFeatures) + 1))
        axes[1].set_yticks(np.arange(len(self.namesOfFeatures) + 1))
        axes[1].set_xticklabels(self.namesOfFeatures + [labelName], rotation=30)
        axes[1].set_yticklabels(self.namesOfFeatures + [labelName])

        for j in range(len(self.namesOfFeatures) + 1):
            for k in range(len(self.namesOfFeatures) + 1):
                axes[1].text(k, j, format(spearman_corr_matrix[j, k], ".2f"),
                             ha="center", va="center", color="black")

        fig.tight_layout()
        cbar1 = fig.colorbar(im1, ax=axes[0])
        cbar2 = fig.colorbar(im2, ax=axes[1])

        plt.savefig("corrMatrices-" + dsName + "-" + fileName + ".png")
        plt.close('all')
        plt.clf()

        print("************End: drawSingleCorrelationMatrix!*********")
        
    def drawHeatCentrality( self, fileName, dsName, clip_min=None, clip_max=None ):
        print("self.namesOfFeatures:", self.namesOfFeatures)
        for g in self.graphs:
            labelDone = False
            designName = g.name
            print("************* INSIDE DRAWHEAT CENTRALITY *****************")
            print("Circuit:", designName)
            print("feat2d:", feat2d)
            if g.ndata.get(feat2d) is None:
                print( "g.ndata.get(feat2d) is None!" )
                break
            print("g.ndata[feat2d]:", type(g.ndata[feat2d]), g.ndata[feat2d].shape, flush=True)
            print( "features:", self.namesOfFeatures )
            print( g.ndata[feat2d], flush=True )
            positions = g.ndata["position"].to(torch.float32).to("cpu")
            feat_values = g.ndata[ feat2d ]
            
            label_values = g.ndata[ labelName ]
            num_columns = 1 if len( feat_values.shape ) == 1  else feat_values.shape[1]  # Get the number of columns in the 2D tensor
            if num_columns != len(self.namesOfFeatures):
                print("ERROR: namesOfFeatures and num_columns with different sizes in drawHeatCentrality!!")
            # fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
            if  num_columns == 1:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                axes = [ax,ax]  # Create a list with a single Axes object
            else:
                #fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
                fig, axes = plt.subplots(1, num_columns + 1, figsize=(8 * (num_columns + 1), 6))
            for i, ax in enumerate(axes[:-1]):
                dummy_image = ax.imshow([[0, 1]], cmap='coolwarm')
                feat_values_i = feat_values if len(feat_values.shape) == 1 else feat_values[:, i]
                clip_min = feat_values_i.min()
                clip_max = feat_values_i.max()
                feat_values_i = np.clip(feat_values_i, clip_min, clip_max)
                for pos, feat_value in zip(positions, feat_values_i):
                    xmin, ymin, xmax, ymax = pos.tolist()
                    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=cm.RdYlGn((feat_value - clip_min) / (clip_max - clip_min)))
                    ax.add_patch(rect)
                ax.set_xlim(positions[:, 0].min() - 1, positions[:, 2].max() + 4)
                ax.set_ylim(positions[:, 1].min() - 1, positions[:, 3].max() + 1)
                ax.set_title(self.namesOfFeatures[i])
                ax.set_aspect('equal')  # Set aspect ratio to equal for proper rectangle visualization
            # Plot labels
            if( num_columns > 1 ):
                ax = axes[-1]
                for pos, label_value in zip(positions, label_values):
                    xmin, ymin, xmax, ymax = pos.tolist()
                    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=cm.RdYlGn((label_value - label_values.min()) / (label_values.max() - label_values.min())))
                    ax.add_patch(rect)
                ax.set_xlim(positions[:, 0].min() - 1, positions[:, 2].max() + 4)
                ax.set_ylim(positions[:, 1].min() - 1, positions[:, 3].max() + 1)
                ax.set_title(f"{labelName}")
                ax.set_aspect('equal')
            
            # cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # Define colorbar position (adjust as needed)
            cmap = cm.RdYlGn  # Use the same colormap as your rectangles
            # cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)  # You can adjust the size and pad as needed

            # Add colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='vertical')
            cbar.set_label('Reference Colorbar')
            # plt.tight_layout()
            plt.savefig(f"heatMap-{dsName}-{designName}-{featNames}")
            plt.close('all')
            for ax in axes:
                ax.clear()
            plt.clf()

            # Label figure
            if not labelDone and len( self.namesOfFeatures ) == 1:
                print( "drawing label heatmap!" )
                label_values = g.ndata[labelName]
                fig, ax = plt.subplots(figsize=(6, 6))
                for pos, label_value in zip(positions, label_values):
                    xmin, ymin, xmax, ymax = pos.tolist()
                    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor=cm.RdYlGn((label_value - label_values.min()) / (label_values.max() - label_values.min())))
                    ax.add_patch(rect)
                ax.set_xlim(positions[:, 0].min() - 1, positions[:, 2].max() + 4)
                ax.set_ylim(positions[:, 1].min() - 1, positions[:, 3].max() + 1)
                ax.set_title(f"{labelName} Visualization")
                ax.set_aspect('equal')
                cax = fig.add_axes([0.15, -0.1, 0.7, 0.03])
                cmap = cm.RdYlGn
                cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
                cbar.set_label( 'Reference Colorbar' )
                plt.savefig( f"heatMap-{dsName}-{designName}-Label" ) # firstDS )
                plt.close('all')
                labelDone = True
                print( "DONE drawing label heatmap!" )
          
    def drawDataAnalysis( self, fileName, dsName ):
        print("************* INSIDE DRAW DATA ANALYSIS *****************", flush=True)
        if self.graphs[0].ndata.get(feat2d) is None:
            print( "g.ndata.get(feat2d) is None!" )
            return
        agg_features = [graph.ndata[feat2d] for graph in self.graphs]
        agg_labels = [graph.ndata[labelName] for graph in self.graphs]

        print( "self.namesOfFeatures:", self.namesOfFeatures )
        print( "agg_features[0].shape:", agg_features[0].shape )
        print( "agg_features:", type( agg_features ), len( agg_features ) )#, "\n", agg_features )
        min_max_data = []
        for i, feature_name in enumerate(self.namesOfFeatures):
            agg_features_2d = [feat if len(feat.shape) == 1 else feat[:, i] for feat in agg_features]
            feature_values = torch.cat(agg_features_2d, dim=0).cpu().numpy()
            #print( "agg_features_2d:", len( agg_features_2d ) )
            #print( "feature_values:", feature_values.shape )
            min_value = min(feature_values)
            max_value = max(feature_values)
            min_max_data.append((f'Min_{feature_name}', min_value))
            min_max_data.append((f'Max_{feature_name}', max_value))

        label_values = torch.cat(agg_labels).cpu().numpy()
        min_label_value = min(label_values)
        max_label_value = max(label_values)
        min_max_data.append(('Min_Label', min_label_value))
        min_max_data.append(('Max_Label', max_label_value))

        total_num_nodes = sum(graph.number_of_nodes() for graph in self.graphs)
        total_num_edges = sum(graph.number_of_edges() for graph in self.graphs)

        with open(f"dataSet-MinAndMax-{fileName}.csv", mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(min_max_data)
            csv_writer.writerow(('Total_Num_Nodes', total_num_nodes))
            csv_writer.writerow(('Total_Num_Edges', total_num_edges))

        
        num_features = 1 if len( agg_features[0].shape ) == 1 else agg_features[0].shape[1]
        num_labels = 1  # Assuming there's only one label column
        num_plots = num_features + num_labels

        num_rows = math.ceil(math.sqrt(num_plots))
        num_cols = math.ceil(num_plots / num_rows)

        plt.figure(figsize=(12, 6))
        for i in range(num_features):
            plt.subplot(num_rows, num_cols, i + 1)          
            agg_features_2d = [feat if len(feat.shape) == 1 else feat[:, i] for feat in agg_features]
            feature_values = torch.cat(agg_features_2d, dim=0).cpu().numpy()
            # sns.boxplot( data = feature_values )
            # stats.probplot( feature_values, plot=plt)
            # sns.violinplot( data = feature_values, inner="quartile")  
            sns.histplot(feature_values, kde=False, bins=20)
            plt.xlabel(self.namesOfFeatures[i])
            plt.ylabel('Count')
            plt.title(self.namesOfFeatures[i])

        plt.subplot(num_rows, num_cols, num_plots)  # Label in a separate subplot
        sns.histplot(torch.cat(agg_labels).cpu().numpy(), kde=False)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Label')

        plt.suptitle('Aggregated Data Distribution for All Graphs')
        plt.tight_layout()
        plt.savefig("histograms-" + dsName + "-" + fileName + ".png")
        plt.close('all')
        plt.clf()

#QQ PLOT WORKING
    # def drawDataAnalysis(self, fileName):
    #     print("************* INSIDE DRAW DATA ANALYSIS *****************", flush=True)
    #     agg_features = [graph.ndata[feat2d] for graph in self.graphs]
    #     agg_labels = [graph.ndata[labelName] for graph in self.graphs]
    #     num_features = agg_features[0].shape[1]
    #     num_labels = 1  # Assuming there's only one label column
    #     num_rows = math.ceil((num_features + num_labels) / 2)
    #     plt.figure(figsize=(12, 6))
    #     for i in range(num_features):
    #         plt.subplot(num_rows, 2, i + 1)
    #         # Create a Q-Q plot for the current feature
    #         data = torch.cat([feat[:, i] for feat in agg_features]).cpu().numpy()
    #         stats.probplot(data, dist='norm', plot=plt)
    #         plt.xlabel('Theoretical Quantiles')
    #         plt.ylabel('Ordered Values')
    #         plt.title(f'Q-Q Plot for ' + self.namesOfFeatures[i])
    #     plt.subplot(num_rows, 2, num_rows * 2)  # Label in a separate subplot
    #     # Create a Q-Q plot for the labels
    #     label_data = torch.cat(agg_labels).cpu().numpy()
    #     stats.probplot(label_data, dist='norm', plot=plt)
    #     plt.xlabel('Theoretical Quantiles')
    #     plt.ylabel('Ordered Values')
    #     plt.title('Q-Q Plot for Labels')
    #     plt.suptitle('Q-Q Plots for Aggregated Data')
    #     plt.tight_layout()
    #     plt.savefig("qqPlot-" + fileName + ".png")
    #     plt.close('all')
    #     plt.clf()
        

    def drawDataAnalysisPerGraph(self, filePrefix):
        print("************* INSIDE DRAW DATA ANALYSIS FOR EACH GRAPH *****************", flush=True)
        for i, graph in enumerate(self.graphs):
            print("graph.name:", graph.name)
            if graph.ndata.get(feat2d) is None:
                print( "g.ndata.get(feat2d) is None!" )
                break
            agg_features = graph.ndata[feat2d]
            agg_labels = graph.ndata[labelName]
            # num_features = agg_features.shape[1]
            num_features = 1 if len( agg_features.shape ) == 1 else agg_features.shape[1]
            num_labels = 1  # Assuming there's only one label column
            total_plots = num_features + num_labels
            num_cols = 2  # Number of columns for the subplot grid
            num_rows = (total_plots + num_cols - 1) // num_cols  # Calculate the number of rows dynamically

            plt.figure(figsize=(12, num_rows * 3))  # Adjust the figure size based on the number of rows

            for j in range(total_plots):
                plt.subplot(num_rows, num_cols, j + 1)
                if j < num_features:
                    # sns.histplot(agg_features[:, j].cpu().numpy(), kde=True, bins=20)
                    sns.histplot( agg_features.cpu().numpy() if len(agg_features.shape) == 1 else agg_features[:, j].cpu().numpy(), kde=True, bins=20)
                    # [feat if len(feat.shape) == 1 else feat[:, i] for feat in agg_features]
                    plt.xlabel(self.namesOfFeatures[j])
                    plt.ylabel('Count')
                    plt.title(f'{self.namesOfFeatures[j]} ({graph.name})')
                else:
                    sns.histplot(agg_labels.cpu().numpy(), kde=True)
                    plt.xlabel('Labels')
                    plt.ylabel('Count')
                    plt.title(f'Label ({graph.name})')

            plt.suptitle(f'Data Distribution for {graph.name}')
            plt.tight_layout()
            plt.savefig(f"graphLevel({graph.name})-{filePrefix}.png")
            plt.close('all')
            plt.clf()


    def process( self ):
        for path in self.graphPaths:
	        graph = self._process_single( path )
	        self.graphs.append( graph )

    def normalize_scores( self, scores_list, mode = 'oneZero' ):
        if mode == 'oneZero':
            min_score = min( scores_list )
            max_score = max( scores_list )
            normalized_data = [(score - min_score) / (max_score - min_score) for score in scores_list ]
        if mode == 'meanStd':
            mean_value = np.mean( scores_list ) 
            std_dev = np.std( scores_list )
            normalized_data = [ ( value - mean_value ) / std_dev for value in scores_list ]
        if mode == 'log':
            normalized_data = [ np.log(value + 1e-10 ) for value in scores_list ]
        return normalized_data
                
    def _process_single( self, designPath ):
        print( "\n\n########## PROCESS SINGLE #################" )
        print( "      Circuit:", str( designPath ).rsplit( '/' )[-1] )
        print( "###########################################\n" )
        print( "self.ablationFeatures (_process_single):", type( self.ablationFeatures ), len( self.ablationFeatures ), self.ablationFeatures )
        graphName = str( designPath ).rsplit( '/' )[-1]
        positions = pd.read_csv( str( designPath ) + "/gatesPosition_" + str( designPath ).rsplit( '/' )[-1] + ".csv"  )
        positions = positions.rename( columns = { "Name" : "name" } )
        # nodes_data = pd.read_csv( designPath / 'preProcessedGatesToHeat.csv', index_col = 'id' )
        nodes_data = pd.read_csv( designPath / 'preProcessedGatesToHeat.csv', index_col = 'id', dtype = { 'type':'int64', 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'float64', 'output_pins':'float64', 'logic_function':'string' } )                    
        ################### PROCESS EXTERNAL CENTRALITIES   ###########################
        print( "BEFORE MERGE by id (external centralities) nodes_data:", nodes_data.shape )#, "\n", nodes_data )                
        for centr in externalCentralities:
            # print( "loading centrality:", centr )
            if centr in self.ablationFeatures:
                external = pd.read_csv( str( designPath ) +'/'+ graphName+'-'+centr+'.csv' ) 
                # print( "external centralities pandas:", external.shape )
                for col in external.columns:
                    #print("col", col)
                    if col.startswith( 'id' ):
                        external = external.rename( columns = { col: 'id' } )
                        filtered_columns = [ col for col in external.columns if 'id' in col or col.startswith( 'feat' ) ]
                        external = external[ filtered_columns ]
                        external = external.rename( columns = lambda x: centr if x.startswith( 'feat' ) else x )
                        # if centr not in self.namesOfFeatures:
                        #     self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, torch.tensor( nodes_data[ centr ].values ) )
                        #     self.namesOfFeatures.append( centr )
                        break
                external.set_index( 'id', inplace = True )
                nodes_data = pd.merge( nodes_data, external, left_index=True, right_index=True, how='left')
                # nodes_data = pd.merge( nodes_data, external, on = 'id', how = 'left' ) 
        print( "AFTER MERGE by id (external centralities) nodes_data:", nodes_data.shape )#, "\n", nodes_data )
        print( "nodes_data columns:", list( nodes_data.columns ) )
        # nodes_data.to_csv( "nodes_data_"+ graphName+".csv", index = True )
        ###############################################################################
        nodes_data = nodes_data.sort_index()
        edges_data = pd.read_csv( designPath / 'DGLedges.csv')
        edges_src  = torch.from_numpy( edges_data['Src'].to_numpy() )
        edges_dst  = torch.from_numpy( edges_data['Dst'].to_numpy() )

        print( "BEFORE MERGE by name nodes_data:", nodes_data.shape )#, "\n", nodes_data )
        nodes_data = pd.merge( nodes_data, positions, on = "name" )
        print( "AFTER MERGE by name nodes_data:", nodes_data.shape )#, "\n", nodes_data )  
        print( "self.ablationFeatures:", type( self.ablationFeatures ) )#, "\n", self.ablationFeatures )

        for column in self.ablationFeatures:
            if column not in nodes_data:
                nodes_data[ column ] = 0
        #nodes_data.update(pd.DataFrame({column: 0 for column in self.ablationFeatures if column not in nodes_data}, index=[0]))
        df = nodes_data[ stdCellFeats + [ labelName ] ]
        
        print("df.columns:",df.columns)
        # df_wanted = (df[rawFeatName] >= 0) & (df[labelName] > 0) if rawFeatName in self.res else (df[labelName] > 0)
        df_wanted = ( df[ "type" ] >= 0 ) & ( df[ labelName] > 0 ) if 'type' in self.ablationFeatures else ( df[ labelName ] > 0 )
        df_wanted = ~df_wanted
        
    #		print( "df_wanted:", df_wanted.shape, "\n", df_wanted )
        removedNodesMask = torch.tensor( df_wanted )
    #		print( "removedNodesMask:", removedNodesMask.shape )#, "\n", removedNodesMask )
        print( "nodes_data:", type( nodes_data ), nodes_data.shape ) #, "\n", nodes_data )
        idsToRemove = torch.tensor( nodes_data.index )[ removedNodesMask ]
        print( "idsToRemove:", idsToRemove.shape ) #,"\n", torch.sort( idsToRemove ) )

    ###################### BUILD GRAPH #####################################        
        self.graph = dgl.graph( ( edges_src, edges_dst ), num_nodes = nodes_data.shape[0] )
        self.graph.name = graphName

        for featStr in stdCellFeats:
            if featStr in self.ablationFeatures:
                self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, torch.tensor( nodes_data[ featStr ].values ) )
                if featStr not in self.namesOfFeatures:
                    self.namesOfFeatures.append( featStr )

        for centr in externalCentralities:
            if centr in self.ablationFeatures:
                self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, torch.tensor( nodes_data[ centr ].values ) )
                if centr not in self.namesOfFeatures:                    
                    self.namesOfFeatures.append( centr )        
    
        self.graph.ndata[ labelName  ]  = ( torch.from_numpy ( nodes_data[ labelName   ].to_numpy() ) )
        self.graph.ndata[ "position" ] = torch.tensor( nodes_data[ [ "xMin","yMin","xMax","yMax" ] ].values )
    ################### REMOVE NODES #############################################
        print( "---> BEFORE REMOVED NODES:")
        print( "\tself.graph.nodes()", self.graph.nodes().shape )#, "\n", self.graph.nodes() )
        #print( "\tself.graph.ndata\n", self.graph.ndata )
        isolated_nodes_before = ( ( self.graph.in_degrees() == 0 ) & ( self.graph.out_degrees() == 0 ) ).nonzero().squeeze(1)
        print( "isolated nodes before any node removal:", isolated_nodes_before.shape )
        self.graph.remove_nodes( idsToRemove )
        isolated_nodes = ( ( self.graph.in_degrees() == 0 ) & ( self.graph.out_degrees() == 0 ) ).nonzero().squeeze(1)
        print( "isolated_nodes:", isolated_nodes.shape ) #, "\n", isolated_nodes )
        self.graph.remove_nodes( isolated_nodes )
        print( "\n---> AFTER REMOVED NODES:" )
        print( "\tself.graph.nodes()", self.graph.nodes().shape ) #, "\n", self.graph.nodes() )    
    ################### EIGENVECTOR  ################################################
        if 'eigen' in self.ablationFeatures:
            print( "calculating eigenvector!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            eigen_scores = nx.eigenvector_centrality_numpy( nx_graph, max_iter = 5000 )
            eigen_scores_list = list( eigen_scores.values() )            
            eigen_tensor = torch.tensor( self.normalize_scores( eigen_scores_list, globalNormMode ) )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, eigen_tensor )
            if 'Eigen' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'Eigen' )
    ################### PAGE RANK ################################################    
        if 'pageRank' in self.ablationFeatures:
            nx_graph = self.graph.to_networkx()
            pagerank_scores = nx.pagerank(nx_graph)
            pagerank_scores_list = list( pagerank_scores.values() )
            pagerank_tensor = torch.tensor( self.normalize_scores( pagerank_scores_list , globalNormMode ) )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, pagerank_tensor )
            if 'pageRank' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'pageRank' )
    ################### IN DEGREE ################################################    
        if 'inDegree' in self.ablationFeatures:
            nx_graph = self.graph.to_networkx()
            inDegree_scores = nx.in_degree_centrality(nx_graph)
            inDegree_scores_list = list( inDegree_scores.values() )
            inDegree_tensor = torch.tensor( self.normalize_scores( inDegree_scores_list, globalNormMode ) )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, inDegree_tensor )
            if 'inDegree' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'inDegree' )
    ################### OUT DEGREE ################################################    
        if 'outDegree' in self.ablationFeatures:
            nx_graph = self.graph.to_networkx()
            outDegree_scores = nx.out_degree_centrality(nx_graph)
            outDegree_scores_list = list( outDegree_scores.values() )
            outDegree_tensor = torch.tensor( self.normalize_scores( outDegree_scores_list, globalNormMode ) )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, outDegree_tensor )
            if 'outDegree' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'outDegree' )
##################################################################################
        # meanStdNorm = [ 'closeness', 'harmonic', 'betweenness', 'load',  'percolation' , 'eigen', 'pageRank', 'inDegree', 'outDegree' ]
        # mean_values = self.graph.ndata[ meanStdNorm ].mean()
        # std_values = self.graph.ndata[ meanStdNorm ].std()
        # self.graph.ndata[ meanStdNorm ] = ( self.graph.ndata[ meanStdNorm ] - mean_values) / std_values
        
        if SELF_LOOP:
            self.graph = dgl.add_self_loop( self.graph )           
        print( "self.ablationFeatures (_process_single):", type( self.ablationFeatures ), len( self.ablationFeatures ), self.ablationFeatures )
        print( "---> _process_single DONE.\n\tself.graph.ndata", type( self.graph ) ) #, "\n\t", self.graph.ndata, flush = True )
        return self.graph
    
    # ###################### LOGIC DEPTH #####################################
    #     #drawGraph( self.graph, "afterRemove_" + self.graph.name )
    #     def is_acyclic(graph):
    #         try:
    #             cycle = nx.find_cycle( graph.to_networkx(), orientation='original' )
    #             print("cycle:", cycle, flush = True )
    #             return False  # Found a cycle
    #         except nx.NetworkXNoCycle:
    #             return True  # No cycles found
    #     src = [0, 1, 2, 3]
    #     dst = [1, 2, 3, 0]
    #     G = dgl.graph((src, dst))
    #     src = [0, 1, 2]
    #     dst = [1, 2, 3]
    #     G2 = dgl.graph((src, dst))
        
    #     src = [ 180,179,181,181,177,176,138,137,137,136,136,135,134,175,178 ]
    #     dst = [ 178,178,177,180,175,175,176,177,179,176,181,181,179,133,132 ]
    #     G3 = dgl.graph((src, dst))

    #     print( "check cycle in graph G" )
    #     print( "G is_acyclic:", type( is_acyclic( G ) ),  is_acyclic( G ) )
    #     print( "check cycle in graph G2" )
    #     print( "G2 is_acyclic:", type( is_acyclic( G2 ) ),  is_acyclic( G2 ) )
    #     print( "check cycle in graph G3" )
    #     print( "G3 is_acyclic:", type( is_acyclic( G3 ) ),  is_acyclic( G3 ) )
    #     drawGraph( G, "G") 
    #     drawGraph( G2, "G2" )
    #     drawGraph( G3, "G3" )

    #     print( "check cycle in graph self.graph", flush = True )
    #     print( "self graph is_acyclic:", type( is_acyclic(  self.graph ) ),  is_acyclic( self.graph ), flush = True )
    #     if 'logicDepth' in self.ablationFeatures:
    #         print( "calculating logic depth!" )
    #         print("calculating logic depth!", self.graph.name, flush=True)
    #         depths = np.zeros(self.graph.number_of_nodes(), dtype=int)
    #         inputs = [node for node in self.graph.nodes() if self.graph.in_degrees(node) == 0 and self.graph.out_degrees(node) > 0]
    #         outputs = [node for node in self.graph.nodes() if self.graph.out_degrees(node) == 0 and self.graph.in_degrees(node) > 0]

    #         print("depths:", len(depths))
    #         print("inputs:", len(inputs), flush=True)
    #         print("outputs:", len(outputs), flush=True)

    #         for output_node in outputs:
    #             stack = [(output_node, 0)]
    #             while stack:
    #                 node, depth = stack.pop()
    #                 depths[node] = depth
    #                 for pred_node in self.graph.predecessors(node):
    #                     if depths[pred_node] >= depth + 1:
    #                         continue
    #                     stack.append((pred_node, depth + 1))
    #         self.graph.ndata[feat2d] = dynamicConcatenate(self.graph.ndata, torch.tensor(depths))

           
    def __getitem__( self, i ):
        return self.graphs[i]

    def __len__( self ):
        #return 1
        return len( self.graphs )
        
    def printDataset( self ):
        totalNodes = 0
        totalEdges = 0
        print( "\n\n###", "size:", len( self.graphs ) )
        for idx in range( len( self.graphs ) ):
            print( "\t>>>", idx," - ", self.names[idx], "\t\tV:", self.graphs[idx].num_nodes(), "E:", self.graphs[idx].num_edges() ) #, "\n", self.graphs[idx] )
            totalNodes += self.graphs[idx].num_nodes()
            totalEdges += self.graphs[idx].num_edges()    
        print( "Total Vertices:", totalNodes )
        print( "Total Edges:   ", totalEdges )
        #drawGraph( self.graphs[idx], self.names[idx] )

    def getNames( self ):
	    return self.names

    def appendGraph( self, inDataSet ):
        self.graphPaths += inDataSet.graphPaths
        self.graphs     += inDataSet.graphs
        self.names      += inDataSet.names
        self.allNames   += inDataSet.allNames
        # self.graphPaths.append( inDataSet.graphPaths )
        # self.graphs.append( inDataSet.graphs )
        # self.names.append( inDataSet.names )
        # self.allNames.append( inDataSet.allNames )
        #self.mode = "complete_dataset"
        if self.ablationFeatures != inDataSet.ablationFeatures:
            print( "ERROR! unexpected ablation list!!" )
        




def drawGraph( graph, graphName ):
    print( "Drawing graph:", graphName )
#    print("graph:",type(graph))
#    print('We have %d nodes.' % graph.number_of_nodes())
#    print('We have %d edges.' % graph.number_of_edges())
    nx_G = graph.to_networkx()
    pos = nx.kamada_kawai_layout( nx_G )
    plt.figure( figsize=[15,7] )
    nx.draw( nx_G, pos, with_labels = True, node_color = [ [ .7, .7, .7 ] ] )
    #	plt.show()
    plt.savefig( graphName )
    plt.close( 'all' )
    plt.clf()

#    print("len graph.ndata:",len(graph.ndata))
#    print("type graph.ndata:",type(graph.ndata))

def drawHeat( tensorLabel, tensorPredict, drawHeatName, graph ):
    designName = graph.name
    print( "************* INSIDE DRAWHEAT *****************" )
    print( "Circuit:", designName )
    print( "label:", type( tensorLabel ), tensorLabel.shape )
    print( "predict:", type( tensorPredict ), tensorPredict.shape )

    predict_normalized = ( tensorPredict - tensorPredict.min() ) / ( tensorPredict.max() - tensorPredict.min() )
    label_normalized   = tensorLabel #( tensorLabel - tensorLabel.min()) / ( tensorLabel.max() - tensorLabel.min() )    
    plt.close( 'all' )
    plt.clf()

############################# HEATMAPS ###############################################################
    positions = graph.ndata[ "position" ].to( torch.float32 ).to( "cpu" )
    if not torch.equal( graph.ndata[ labelName ].to(device).to( torch.float32 ), tensorLabel ):
        print( "\n\n\n\n SOMETHING WRONG!!! UNEXPECTED LABELS IN DRAW HEAT \n\n\n" )
        print( "positions.to(device):\n", graph.ndata[ labelName ].to(device).to( torch.float32 ), "\n\ntensorLabel:\n", tensorLabel )

    fig, ( ax1, ax2 ) = plt.subplots( 1, 2, figsize = ( 12, 6 ) )
    dummy_image1 = ax1.imshow( [ [ 0, 1 ] ], cmap = 'coolwarm' )
    dummy_image2 = ax2.imshow( [ [ 0, 1 ] ], cmap = 'coolwarm' ) 
    for pos, pred, lab in zip( positions, predict_normalized, label_normalized ):
        xmin, ymin, xmax, ymax = pos.tolist()
        rect1 = Rectangle( ( xmin, ymin), xmax - xmin, ymax - ymin, facecolor = cm.coolwarm( pred.item() ) )
        rect2 = Rectangle( ( xmin, ymin), xmax - xmin, ymax - ymin, facecolor = cm.coolwarm( lab.item() ) )
        # Add the rectangle patches to the respective plots
        ax1.add_patch( rect1 )
        ax2.add_patch( rect2 )

    ax1.set_xlim( positions[ :, 0 ].min() - 1, positions[ :, 2 ].max() + 4 )
    ax1.set_ylim( positions[ :, 1 ].min() - 1, positions[ :, 3 ].max() + 1 )
    ax1.set_title( 'Predict' )
    ax1.set_aspect( 'equal' )  # Set aspect ratio to equal for proper rectangle visualization

    # Create colorbar for Predict
    cax1 = fig.add_axes( [ 1, 0, 0.03, 0.25 ] )  # Define colorbar position
    mcb.ColorbarBase( cax1, cmap = 'coolwarm', norm = mcolors.Normalize( vmin = predict_normalized.min(), vmax = predict_normalized.max() ), orientation = 'vertical' )
    cax1.set_ylabel( 'Prediction' )

    ax2.set_xlim( positions[ :, 0 ].min() - 1, positions[ :, 2 ].max() + 4 )
    ax2.set_ylim( positions[ :, 1 ].min() - 1, positions[ :, 3 ].max() + 1 ) 
    ax2.set_title( 'Label' )
    ax2.set_aspect( 'equal' )  # Set aspect ratio to equal for proper rectangle visualization

    # Create colorbar for Tensor Label
    cax2 = fig.add_axes( [ 1, 0, 0.03, 0.25 ] )  # Define colorbar position
    mcb.ColorbarBase( cax2, cmap = 'coolwarm', norm = mcolors.Normalize( vmin = label_normalized.min(), vmax = label_normalized.max() ), orientation = 'vertical' )
    cax2.set_ylabel( 'Label' )
    
    auxName = drawHeatName.replace( "/", "/heatmaps/", 1 )
    print( "auxName:", auxName )
    plt.savefig( auxName )
    plt.close( 'all' )
    ax1.clear()
    ax2.clear()
    plt.clf()

######################################################################################################

################################# VALUES PLOT ########################################################
    # WORKING!
    # types = graph.ndata[ feat2d ].to( torch.float32 ).to( "cpu" )
    # predict_aux = predict_normalized[:10000].to( "cpu" )
    # label_aux   = label_normalized  [:10000].to( "cpu" )
    # plt.scatter(range(len(predict_aux ) ), predict_aux, color='blue', label='Predict')
    # plt.scatter(range(len(label_aux ) ), label_aux, color='red', label='Label')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # plt.title('Predicted vs. Labeled Values')
    # plt.legend()
    
    # auxName = drawHeatName.replace( "/", "/valuesPlot/", 1 )
    # print( "auxName:", auxName )
    # plt.savefig( auxName )
    # plt.close( 'all' )
    # plt.clf()
######################################################################################################
    
################################# RESIDUAL ########################################################    

    # WORKING!
    # residual = label_normalized - predict_normalized
    # residual_np = residual.to("cpu").numpy()
    # residual_np = residual_np[~np.isnan(residual_np)]
    # if residual_np.size > 0:  # Check if the filtered array is not empty
    #     plt.figure(figsize=(8, 6))
    #     plt.hist(residual_np, bins=50, edgecolor='black')
    #     plt.xlabel('Index')
    #     plt.ylabel('Value')
    #     plt.title('Residual Plot')
    #     plt.xlim( -1, 1 )
    #     plt.legend()
    
    # auxName = drawHeatName.replace( "/", "/errorTable/", 1 )
    # print( "auxName:", auxName )
    # plt.savefig( auxName )
    # plt.close( 'all' )
    # plt.clf()
######################################################################################################

############################ HISTOGRAMS ##############################################################
    # WORKING!
    # tensor1 = predict_normalized
    # tensor2 = label_normalized
    # min_value = min(tensor1.min().item(), tensor2.min().item())
    # max_value = max(tensor1.max().item(), tensor2.max().item())
    # bucket_ranges = [min_value,
    #                  min_value + (max_value - min_value) / 4,
    #                  min_value + (max_value - min_value) / 2,
    #                  min_value + (max_value - min_value) * 3 / 4,
    #                  max_value]
    # match_counts = [0] * len(bucket_ranges)
    # total_counts = len( tensor1 )
    # for val1, val2 in zip(tensor1, tensor2):
    #     for i in range(len(bucket_ranges) - 1):  # Exclude the last bucket range
    #         if bucket_ranges[i] <= val1 < bucket_ranges[i+1] and bucket_ranges[i] <= val2 < bucket_ranges[i+1]:
    #             match_counts[i] += 1
    # fig, (ax2, ax1, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 2, 2]})
    # ax1.hist( [ tensor1.tolist(), tensor2.tolist() ], bins = bucket_ranges, alpha = 0.7, label = [ 'predict', 'label' ] )
    # ax1.set_ylabel( 'Frequency' )
    # ax3.set_ylabel( 'Frequency' )
    # for i in range(len(bucket_ranges) - 1):  # Exclude the last bucket range
    #     bucket_center = ( bucket_ranges[i] + bucket_ranges[ i+1 ] ) / 2
    #     width = ( bucket_ranges[i+1] - bucket_ranges[i] ) * 0.8
    #     ax1.text( bucket_center, match_counts[i], str( int( match_counts[i] ) ), ha='center' )
    #     ax3.text( bucket_center, match_counts[i], str( round( ( match_counts[i] / total_counts )*100,1 ) ), ha='center' )
    #     ax3.bar( bucket_center, match_counts[i], width = width, color = 'green' )
    # ax2.hist( tensor1.tolist(), bins = bucket_ranges, alpha=0.7, label='predict')
    # ax2.hist( tensor2.tolist(), bins = bucket_ranges, alpha=0.7, label='label')
    # ax2.set_xlabel('Value')
    # ax2.set_ylabel('Frequency')
    # ax2.legend()
    # fig.suptitle('Histogram Comparison')
    # plt.tight_layout()
    # auxName = drawHeatName.replace("/", "/histogram/", 1)
    # print("auxName:", auxName)
    # plt.savefig(auxName)
    # plt.close('all')
######################################################################################################

    
    
class SAGE( nn.Module ):
    def __init__( self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv( in_feats = in_feats, out_feats = hid_feats, aggregator_type = 'lstm' )
        self.conv2 = dglnn.SAGEConv( in_feats = hid_feats, out_feats = out_feats, aggregator_type = 'lstm' )
        self.activation = nn.Sigmoid()

    def forward( self, graph, inputs ):
        # inputs are features of nodes
        h = self.conv1( graph, inputs )
        h = F.relu(h)
        h = self.conv2( graph, h )
        h = torch.sigmoid( h )
        return h
	

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
#        heads = [4,4,6,6,6]
        self.gatLayers = nn.ModuleList()
        self.gatLayers.append(dglnn.GATConv(in_size, hid_size, heads[0], activation=F.relu, allow_zero_in_degree=not SELF_LOOP))
        self.gatLayers.append(dglnn.GATConv(hid_size * heads[0], hid_size, heads[1], residual=True, activation=F.relu, allow_zero_in_degree=not SELF_LOOP))
        self.gatLayers.append(dglnn.GATConv(hid_size * heads[1], out_size, heads[2], residual=True, activation=None, allow_zero_in_degree=not SELF_LOOP))
#        self.gatLayers.append(dglnn.GATConv(hid_size * heads[2], out_size, heads[3], residual=True, activation=F.relu, allow_zero_in_degree=not SELF_LOOP))
#        self.gatLayers.append(dglnn.GATConv(hid_size * heads[3], out_size, heads[4], residual=True, activation=None, allow_zero_in_degree=not SELF_LOOP))
        
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gatLayers):
            h = layer(g, h)
            if i == len(self.gatLayers) - 1:  # Apply mean pooling to the last layer
                h = h.mean(1)
            else:
                h = h.flatten(1)
        return h


THRESHOLD = 0.75

def evaluate(g, features, labels, model, path, device):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        if features.dim() == 1:
            features = features.unsqueeze(1)
        predicted = model(g, features)
        predicted = predicted.squeeze(1)
        
        print("\t>>>> labels in evaluate:", type(labels), labels.shape, labels[:10])
        print("\t>>>> predicted after squeeze:", type(predicted), predicted.shape, predicted[:10])

        binary_preds =  (predicted > THRESHOLD).long()        
        binary_labels = (labels >    THRESHOLD).long()

        print("\t>>>> binary_preds:", binary_preds[:10])
        print("\t>>>> binary_labels:", binary_labels[:10])

        tp = (binary_preds * binary_labels).sum().item()
        tn = ((1 - binary_preds) * (1 - binary_labels)).sum().item()
        fp = (binary_preds * (1 - binary_labels)).sum().item()
        fn = ((1 - binary_preds) * binary_labels).sum().item()

        print(f"\t>>>> tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")

        precision, recall, f1, _ = precision_recall_fscore_support(binary_labels.cpu(), binary_preds.cpu(), average='binary')
        accuracy = accuracy_score(binary_labels.cpu(), binary_preds.cpu())

        # Debugging prints to check the metrics
        print(f"\t>>>> Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")

        mse = F.mse_loss(predicted, labels).item()
        print("Mean Squared Error:", mse)
        rmse = math.sqrt(mse)
        print("Root Mean Squared Error:", rmse)
        kendall = KendallRankCorrCoef(variant='a').to(device)

        print("calculating kendal...", flush=True)
        score_kendall = kendall(predicted, labels)
        print("Kendall calculated", flush=True)

        predicted_cpu = predicted.cpu().detach().numpy()
        labels_cpu = labels.cpu().detach().numpy()
        corrPearson, pPearson = pearsonr(predicted_cpu, labels_cpu)
        print("Pearson correlation:", corrPearson, flush=True)
        corrSpearman, pSpearman = spearmanr(predicted_cpu, labels_cpu)
        print("Spearman correlation:", corrSpearman, flush=True)

        if len(path) > 0:
            print("\tdrawing output")
            path = path + "k{:.4f}".format(score_kendall) + ".png"
            if DRAWOUTPUTS:
                drawHeat(labels.to(torch.float32), predicted.to(torch.float32), path, g)
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        else:
            memory_usage = 0
        print("memory usage in evaluate:", memory_usage)

        return score_kendall, rmse, corrPearson, pPearson, corrSpearman, pSpearman, precision, recall, f1, accuracy

def evaluate_in_batches(dataloader, device, model, image_path=""):
    total_kendall = 0.0
    total_corrPearson = 0.0
    total_corrSpearman = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_accuracy = 0.0
    total_rmse = 0.0

    for batch_id, batched_graph in enumerate(dataloader):
        print("batch_id (eval_in_batches):", batch_id)
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata[feat2d].float().to(device)
        labels = batched_graph.ndata[labelName].to(device)

        score_kendall, rmse, corrPearson, _, corrSpearman, _, precision, recall, f1, accuracy = evaluate(
            batched_graph, features, labels, model, image_path, device)
        
        total_kendall += score_kendall
        total_rmse += rmse
        total_corrPearson += corrPearson
        total_corrSpearman += corrSpearman
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_accuracy += accuracy        

    num_batches = batch_id + 1
    avg_kendall = round(total_kendall.item() / num_batches, 3)
    avg_rmse = round(total_rmse / num_batches, 3)
    avg_corrPearson = round(total_corrPearson / num_batches, 3)
    avg_corrSpearman = round(total_corrSpearman / num_batches, 3)
    avg_precision = round(total_precision / num_batches, 3)
    avg_recall = round(total_recall / num_batches, 3)
    avg_f1 = round(total_f1 / num_batches, 3)
    avg_accuracy = round(total_accuracy / num_batches, 3)

    return avg_kendall, avg_rmse, avg_corrPearson, avg_corrSpearman, avg_precision, avg_recall, avg_f1, avg_accuracy


def evaluate_single( graph, device, model, path ):
    graph = graph.to( device )
    features = graph.ndata[ feat2d ].float().to( device )
    labels   = graph.ndata[ labelName ].to( device )
    print( "evaluate single--->", path )                               
    score_kendall, rmse, corrPearson, _, corrSpearman, _ , precision, recall, f1, accuracy = evaluate( graph, features, labels, model, path, device )
    print( "Single graph score - Kendall:", score_kendall, ", corrPearson:", corrSpearman )
    return round( score_kendall.item(), 3 ), round( corrSpearman.item(), 3 ), round( corrPearson.item(), 3 )


def train( train_dataloader, device, model, writerName ):
    print( "device in train:", device )
    writer = SummaryWriter( comment = writerName )

    if CUDA:
        # torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    accumulated_memory_usage = 0
    max_memory_usage = 0
    
    #loss_fcn = nn.BCEWithLogitsLoss()
#    loss_fcn = nn.CrossEntropyLoss()
#    loss_fcn = nn.L1Loss() 
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters(), lr = step, weight_decay = 0 )
    
###################  Stop loop ###########################
    best_loss = float('inf')  # Initialize the best training loss with a large value
    #best_val_loss = float('inf')  # Initialize the best validation loss with a large value
    epochs_without_improvement = 0  # Counter for epochs without improvement
    #val_epochs_without_improvement = 0  # Counter for validation epochs without improvement
    
    for epoch in range( maxEpochs + 1 ):
        model.train()
        total_loss = 0.0
        accumulated_loss = 0.0

        for batch_id, batched_graph in enumerate(train_dataloader):
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata[feat2d].float()
            if features.dim() == 1:
                features = features.float().unsqueeze(1)
            logits = model(batched_graph, features)
            labels = batched_graph.ndata[labelName].float()
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            loss = loss_fcn(logits, labels)

################# Without accumulate #################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
################# Accumulate gradients ################################
            # accumulated_loss += loss
            # if (batch_id + 1) % accumulation_steps == 0:
            #     accumulated_loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     total_loss += accumulated_loss.item()
            #     accumulated_loss = 0.0
#####################################################################
            if CUDA:
                memory_usage = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                max_memory_usage = max( max_memory_usage, memory_usage )
                accumulated_memory_usage += memory_usage
            
        average_loss = total_loss / len(train_dataloader)
        if CUDA:
            average_memory_usage = accumulated_memory_usage / len(train_dataloader)
        else:
            average_memory_usage = 0
        if average_loss < best_loss - improvement_threshold:
            best_loss = average_loss
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1

        ###### Validation loop #######
        # model.eval()
        # total_val_loss = 0
        # for batch_id, batched_graph in enumerate(valid_dataloader):
        #     batched_graph = batched_graph.to(device)
        #     features = batched_graph.ndata[feat2d].float()
        #     if features.dim() == 1:
        #         features = features.float().unsqueeze(1)
        #     logits = model(batched_graph, features)
        #     labels = batched_graph.ndata[labelName].float()
        #     if labels.dim() == 1:
        #         labels = labels.unsqueeze(-1)
        #     loss = loss_fcn(logits, labels)
        #     total_val_loss += loss.item()

        # average_val_loss = total_val_loss / len(valid_dataloader)
        # if average_val_loss < best_val_loss - improvement_threshold:
        #     best_val_loss = average_val_loss
        #     val_epochs_without_improvement = 0  # Reset the counter
        # else:
        #     val_epochs_without_improvement += 1
        # if useEarlyStop and (epochs_without_improvement >= patience or val_epochs_without_improvement >= patience):
        if useEarlyStop and ( ( epoch >= minEpochs ) and (epochs_without_improvement >= patience ) or ( average_loss <= 0.0000000001 ) ): # or val_epochs_without_improvement >= patience):
            print("=======> Early stopping!")
            break

        # print( "Epoch {:05d} | Train Loss {:.4f} | Valid Loss {:.4f} | ". format( epoch, average_loss, average_val_loss ), flush = True, end="" )
        # print( "best_loss:", round( best_loss, 5 ), " | best_val_loss:", round( best_val_loss, 5 ), end=" | " )
        # print( "epochs_without_improvement:", epochs_without_improvement, "val_without_improvement:", val_epochs_without_improvement )
        print( "Epoch {:05d} | Train Loss {:.4f} ". format( epoch, average_loss ), flush = True, end="" )
        print( "best_loss:", round( best_loss, 5 ), end=" | " )
        print( "epochs_without_improvement:", epochs_without_improvement )
        print(f"Epoch: {epoch+1}, Max Memory(MB): {max_memory_usage}, Average Memory(MB): {average_memory_usage}", flush=True)
        
        #writer.add_scalar( "Loss Valid", average_val_loss, epoch )
        writer.add_scalar( "Loss Train", average_loss, epoch )
    writer.flush()
    writer.close()
    torch.cuda.empty_cache()
    return epoch, max_memory_usage, accumulated_memory_usage


def runExperiment( setup ):
    # setup = 2
    if not FUSIONDS:
        if setup == 1:
            firstDS        = 'nangateV2'
            secondDS       = 'asap7V2'
            dsAbbreviated  = 'NG45'
            dsAbbreviated2 = 'A7' 
        elif setup == 2:
            firstDS        = 'asap7V2'
            secondDS       = 'nangateV2'
            dsAbbreviated  = 'A7'
            dsAbbreviated2 = 'NG45'
    else:
        firstDS       = 'nangateV2'
        secondDS      = 'asap7V2'
        dsAbbreviated = 'NG45+A7'

# if __name__ == '__main__':
    startTimeAll = time.time()
    imageOutput = "image_outputs"
    print(f'Training Yosys Dataset with DGL built-in GATConv module.')

    listDir = []	
    if COLAB:
        dsPath = '/content/drive/MyDrive/tese - datasets/dataSet'
    else:
        dsPath = Path.cwd() / firstDS
        
    for designPath in Path( dsPath ).iterdir():
	    if designPath.is_dir() and "runs" not in str( designPath ):
		    print("designPath:", designPath )
		    listDir.append( designPath )

    ##################################################################################
    ############################# Pre Processing #####################################
    ##################################################################################
    # writeDFrameData( listDir, 'gatesToHeatSTDfeatures.csv', "DSinfoBeforePreProcess.csv" )
    # df = aggregateData( listDir, 'gatesToHeatSTDfeatures.csv' )
    # print( "\n\n#######################\n## BEFORE PRE ##\n####################### \n\nallDFs:\n", df )
    # for col in df:
    #     print( "describe:\n", df[ col ].describe() )
    # df.to_csv( "aggregatedDFBefore-pandasLevel.csv" )
    # df = df.drop( df.index[ df[ labelName ] < 0 ] )
    # df.hist( bins = 50, figsize = (15,12) )
    # plt.savefig( "BeforePreProcess-pandasLevel" )
    # plt.close( 'all' )
    # plt.clf()

    process_and_write_csvs( listDir )
    #preProcessData( listDir )
    if LOADSECONDDS or FUSIONDS:
        secondListDir = []
        secondDSPath = Path.cwd() / secondDS
        for designPath in Path( secondDSPath ).iterdir():
            if designPath.is_dir() and "runs" not in str( designPath ):
                print("designPath:", designPath )
                secondListDir.append( designPath )
        process_and_write_csvs( secondListDir )
        if FUSIONDS:
            listDir = listDir + secondListDir
	
    # writeDFrameData( listDir, 'preProcessedGatesToHeat.csv', "DSinfoAfterPreProcess.csv" )
    # df = aggregateData( listDir, 'preProcessedGatesToHeat.csv' )
    # print( "\n\n#######################\n## AFTER PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
    # for col in df:
    #         print( "\n####describe:\n", df[ col ].describe() )
    # df.to_csv( "aggregatedDFAfter-pandasLevel.csv" )
    # df = df.drop( df.index[ df[ labelName ] < 0 ])
    # #df = df.drop( df.index[ df[ rawFeatName ] < 0 ] )
    # df.hist( bins = 50, figsize = (15,12) )
    # plt.savefig( "AfterPreProcess-pandasLevel" )
    # plt.close( 'all' )
    # plt.clf()
    ##################################################################################
    ##################################################################################
    ##################################################################################
    
    summary = "runSummary.csv"
    ablationResult = "ablationResult.csv"
    with open( summary, 'w' ) as f:
        f.write("")
    with open( ablationResult, 'w' ) as f:
        f.write( "Features,Train-Mean,Train-SD,Test-Mean,Test-SD,TrainPearson-Mean,TrainPearson-SD,TrainSpearman-Mean,TrainSpearman-SD" )
        if not MIXEDTEST:
            f.write( "\n" )
        else:
            f.write( ",mixTestKendall,mixTestPearson,mixTestSpearman\n" )
    if os.path.exists( "runs" ):
        shutil.rmtree( "runs" )

    prefix = "[\'"
    for folder_name in os.listdir("."):
        if folder_name.startswith(prefix) and os.path.isdir(os.path.join(".", folder_name)):
            folder_path = os.path.join(".", folder_name)
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            except OSError as e:
                print(f"Error deleting folder {folder_path}: {e}")
                
    print( ">>>>>> listDir:" )
    for index in range(len(listDir)):
        print("\tIndex:", index, "- Path:", listDir[index])

    if not MANUALABLATION:
        ablationList = []
        minCombSize = 4
        maxCombSize = minCombSize + 1 # len( fullAblationCombs ) + 1
        for combSize in range( minCombSize, maxCombSize  ):
            print( "combinationSize:", combSize, ", combinations:", len( list( combinations( fullAblationCombs, combSize) ) ) )
            ablationList += list( combinations( fullAblationCombs, combSize ) )
            print( "ablationList:", len( ablationList ), ablationList )
    else:
        #all features
        # ablationList = [(string,) for string in validFeatures] + [tuple(validFeatures)] # um por um, depois todos juntos
        
        # ablationList = [ tuple(  [item for item in validFeatures if item != 'type' ] ) ]
        # # ablationList += [ ( 'closeness', 'eigen' , 'outDegree')  ] # A7 only, best corr with label
        # # ablationList += [ ( 'closeness', 'betweenness', 'pageRank', 'eigen' , 'inDegree', 'outDegree')  ] # NG45 only, best corr with label
        # # ablationList += [ ( 'closeness', 'betweenness', 'pageRank', 'eigen' , 'input_pins', 'output_pins')  ] # NG45 only, best corr with label

        #TCAS1 features
        ablationList =   [('closeness', 'betweenness', 'pageRank', 'eigen' , 'inDegree', 'outDegree', 'area', 'input_pins', 'output_pins')]
        ablationList += [ ( 'area', 'input_pins', 'output_pins' ) ]
        ablationList += [ ( 'closeness', 'betweenness', 'pageRank', 'eigen', 'inDegree', 'outDegree' ) ]
        ablationList += [ ( 'closeness', 'betweenness', 'pageRank', 'eigen') ]
        ablationList += [ ( 'closeness', 'inDegree', 'outDegree' ) ]
        ablationList += [ ( 'eigen', 'pageRank' , 'inDegree', 'outDegree')]

        

    print( "MANUALABLATION:", MANUALABLATION )
    print( "ablationList:", len( ablationList ), ablationList )
    for item in ablationList:
        for sub_item in item:
            if sub_item not in validFeatures:
                print(f"Error: '{sub_item}' is not in validFeatures.")
                sys.exit()
    for ablationIter in ablationList:
        for mainIteration in range( 0, mainMaxIter ):
            print( "##################################################################################" )
            print( "########################## NEW MAIN RUN  ########################################" )
            print( "##################################################################################" )
            print( "mainIteration:", mainIteration )
            print( "--> combination_list:", len( ablationList ), ablationList )
            with open( ablationResult, 'a' ) as f:
                abbreviations = {
                    'inDegree': 'ID',
                    'outDegree': 'OD',
                    'input_pins': 'IP',
                    'output_pins': 'OP',
                    'percolation': 'PR',
                    'pageRank':    'PG'
                }
                abbreviatedFeatures = [ abbreviations.get( s, s[:1].capitalize() ) for s in ablationIter ]
                print("abbreviatedFeatures for ablationResult:", abbreviatedFeatures)
                f.write( '|'.join( abbreviatedFeatures ) )
                
            with open( summary, 'a' ) as f:
                f.write( "trainDS: " + dsAbbreviated + ",MIXEDTEST:" )
                if MIXEDTEST and not FUSIONDS:
                    f.write( secondDS )
                else:
                    f.write( "False" )
                f.write( ",#Circuits:" + str( len( listDir ) ) )
                f.write( ",minEpochs:" + str( minEpochs ) )
                f.write( ",maxEpochs:" + str( maxEpochs ) )
                f.write( ",step:" + str( round( step, 5 ) ) )
                f.write( ",labelName:" + labelName )
                f.write( ",features: " ) 
                f.write( "| ".join( abbreviatedFeatures ) )
                f.write( ",FULLTRAIN: " + str( FULLTRAIN ) )
                f.write( ",MANUALABLATION:" + str( MANUALABLATION ) )
                f.write( ",improvement_threshold:" + str( improvement_threshold ) )
                f.write( ",patience:" + str( patience ) )
                f.write( "\ntrainIndices,testIndices,finalEpoch,runtime(min),MaxMemory,AverageMemory,Circuit Test,TrainKendall,TestKendall,trainRMSE,testRMSE,TrainPearson,TestPearson,TrainSpearman,TestSpearman,trainPrecision,testPrecision,trainRecall,testRecall,trainF1,testF1,trainAccuracy,testAccuracy\n" )

            print( "\n%%%%%%%%%%%%%%%%%%%%%%%%%%\nablationIter:", type( ablationIter ), len( ablationIter ), ablationIter, "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%\n", flush = True )
            ablationIter = list( ablationIter )
            if os.path.exists( imageOutput ):
                shutil.rmtree( imageOutput )
            if DRAWOUTPUTS and mainIteration == 0:
                os.makedirs( imageOutput )
                aux = imageOutput + "/histogram"
                os.makedirs( aux )
                aux = imageOutput + "/heatmaps"
                os.makedirs( aux )
                aux = imageOutput + "/errorTable"
                os.makedirs( aux )
                aux = imageOutput + "/valuesPlot"
                os.makedirs( aux )
            kendallTest =  []
            kendallTrain = []
            pearsonTrain = []
            spearmanTrain = []
            
            if not FUSIONDS:
                theDataset    = DataSetFromYosys( listDir, ablationIter )
                if LOADSECONDDS:
                    secondDataset = DataSetFromYosys( secondListDir, ablationIter )
            else:
                theDataset = DataSetFromYosys( listDir, ablationIter )
                
            if DRAWGRAPHDATA:
                theDataset.drawSingleCorrelationMatrix( '|'.join( abbreviatedFeatures ), dsAbbreviated )
                theDataset.drawDataAnalysis(   '|'.join( abbreviatedFeatures ), dsAbbreviated )
                if LOADSECONDDS and not FUSIONDS:
                    secondDataset.drawSingleCorrelationMatrix( '|'.join( abbreviatedFeatures ), dsAbbreviated2 )
                    secondDataset.drawDataAnalysis(   '|'.join( abbreviatedFeatures ), dsAbbreviated2 )

            if DRAWHEATCENTR:
                theDataset.drawHeatCentrality( '|'.join( abbreviatedFeatures ), dsAbbreviated )    
                if LOADSECONDDS and not FUSIONDS:
                    secondDataset.drawHeatCentrality( '|'.join( abbreviatedFeatures ), dsAbbreviated2 )
                
            if not DOLEARN:
                # sys.exit()
                continue

            kf = KFold( n_splits = num_folds )
            for fold, ( train_indices, test_indices ) in enumerate( kf.split( theDataset ) ):
                # LASCAS comparison, same as HUAWEI
                # train_indices = [i for i in range(len(theDataset)) if i !=2 and i !=4] # remove swerv and bp_be_top

                # HUAWEI's ablation, only Black_parrot
                # train_indices = [7] 
                # test_indices = [ i for i in range( len( theDataset ) ) if i != 7 ]  # black_parrot, 4:remove bp_be_top

                # Only ng45 nonRepeating as test
                # train_indices = [ 6, 3, 8, 13, 12, 2, 1, 11 ]  # 6-aes, 3-gcd, ibex-8, 13-jpeg, 12-swerv_wr, 2-swerv, 1-dynamicNode, 11-eth
                # test_indices =  [ 0, 4, 5, 7, 9, 10 ] # 0-bp_fe, 4-bp_be, 5-rocket, 7-bp, 9-ariane, bp_multi

                # # Only A7 nonRepeating as test
                train_indices = [ 4, 2, 5, 10, 8, 1, 0, 7  ]  # 4-aes, 2-gcd, 5-ibex, 10-jpeg, 8-swerv_wr, 1-swerv, 0-dynamic, 7-eth
                test_indices =  [ 3, 9 ] # uart, mockarray, riscv

                # Fusion - 
                # train_indices = [ 6, 3, 8, 13, 12, 2, 1, 11, 18, 16, 19, 24, 22, 15, 14, 21 ]
                # test_indices =  [ 0, 4, 5, 7, 9, 10, 17, 20, 23 ]

                if FULLTRAIN:
                    train_indices = [ i for i in range( len( theDataset ) ) ]
                    test_indices = []
                print(f"Fold {fold+1}/{num_folds}")
                #train_indices, valid_indices = train_indices[:-len(test_indices)], train_indices[-len(test_indices):]
                        
                startIterationTime = time.time()
                print( "##################################################################################" )
                print( "#################### New CrossValid iteration  ###################################" )
                print( "##################################################################################" )
                features = theDataset[0].ndata[ feat2d ]
                if( features.dim() == 1 ):
                    features = features.unsqueeze(1)
                in_size = features.shape[1]
                out_size = 1 
                print( "in_size", in_size,",  out_size", out_size, flush = True )
                model = GAT( in_size, 256, out_size, heads=[4,4,6] ).to( device )
                # model = GAT( in_size, 128, out_-size, heads=[4,4,6]).to( device )
                # model = SAGE( in_feats = in_size, hid_feats = 125, out_feats  = out_size ).to( device )

                print( "\n###################" )
                print( "## MODEL DEFINED ##"   )
                print( "###################\n", flush = True )
                print( "train_indices:", type( train_indices ), train_indices )
                #print( "valid_indices:", type( valid_indices ), valid_indices )
                print( "test_indices:", type( test_indices ), test_indices )

                if FULLTRAIN:
                    train_dataloader = GraphDataLoader( theDataset, batch_size = 1 )#5 , sampler = train_sampler )
                    #valid_dataloader = GraphDataLoader( theDataset, batch_size = 1 )
                    test_dataloader  = None
                else:
                    train_dataloader = GraphDataLoader( torch.utils.data.dataset.Subset( theDataset, train_indices ), batch_size = 1 )#5 , sampler = train_sampler )
                    #valid_dataloader = GraphDataLoader( torch.utils.data.dataset.Subset( theDataset, valid_indices ), batch_size = 1 )
                    test_dataloader  = GraphDataLoader( torch.utils.data.dataset.Subset( theDataset, test_indices  ), batch_size = 1 )

                print( "len( train_dataloader ) number of batches:", len( train_dataloader ) )
                print( "\n###################"   )
                print( "## THE DATASET ####"   )
                print( "###################"   )
                theDataset.printDataset()
                
                writerName =  "-" + labelName +"-"+ str( len( train_indices ) ) +"-"+ str( len( test_indices ) )
                # writerName += "-" + '|'.join( ablationIter ) + "-" + str( mainIteration )
                writerName += "-#Feat"+ str( len( ablationIter ) ) + "-" + str( mainIteration )
                writerName += " T-"+ ';'.join( theDataset.getNames()[i][:6] for i in test_indices )
                finalEpoch, maxMem, avergMem = train( train_dataloader, device, model, writerName )
                finalEpoch += 1

                print('######################\n## Final Evaluation ##\n######################\n', flush=True)
                startTimeEval = time.time()
                if not SKIPFINALEVAL:
                    if not FULLTRAIN:
                        test_kendall, test_rmse, test_corrPearson, test_corrSpearman, test_precision, test_recall, test_f1, test_accuracy = evaluate_in_batches(test_dataloader, device, model)
                    else:
                        test_kendall = test_rmse = test_corrPearson = test_corrSpearman = test_precision = test_recall = test_f1 = test_accuracy = 0
                    train_kendall, train_rmse, train_corrPearson, train_corrSpearman, train_precision, train_recall, train_f1, train_accuracy = evaluate_in_batches(train_dataloader, device, model)

                    # TODO: improve this, problem when accessing each graph name with batched graphs
                    if DRAWOUTPUTS and mainIteration == 0: 
                        for n in test_indices:
                            g = theDataset[n].to(device)
                            path = theDataset.names[n]
                            path = imageOutput + "/test-" + path + "-testIndex" + '|'.join(map(str, test_indices)) + "-e" + str(finalEpoch) + "-feat" + '|'.join(abbreviatedFeatures)
                            evaluate_single(g, device, model, path)  # using only for drawing for now
                        for n in train_indices:
                            g = theDataset[n].to(device)
                            path = theDataset.names[n]
                            path = imageOutput + "/train-" + path + "-trainIndex" + '|'.join(map(str, train_indices)) + "-e" + str(finalEpoch) + "-feat" + '|'.join(abbreviatedFeatures)
                            evaluate_single(g, device, model, path)  # using only for drawing for now
                else:
                    test_kendall = test_rmse = test_corrPearson = test_corrSpearman = train_kendall = train_corrPearson = train_corrSpearman = test_precision = test_recall = test_f1 = test_accuracy = train_precision = train_recall = train_f1 = train_accuracy = 0

                print("Total Train Kendall {:.4f}".format(train_kendall))
                print("Total Train RMSE {:.4f}".format(train_rmse))
                print("Total Train CORRPEARSON {:.4f}".format(train_corrPearson))
                print("Total Train corrSpearman {:.4f}".format(train_corrSpearman))
                print("Total Train Precision {:.4f}".format(train_precision))
                print("Total Train Recall {:.4f}".format(train_recall))
                print("Total Train F1 {:.4f}".format(train_f1))
                print("Total Train Accuracy {:.4f}".format(train_accuracy))

                print("Total Test Kendall {:.4f}".format(test_kendall))
                print("Total Test RMSE {:.4f}".format(test_rmse))
                print("Total Test CORRPEARSON {:.4f}".format(test_corrPearson))
                print("Total Test corrSpearman {:.4f}".format(test_corrSpearman))
                print("Total Test Precision {:.4f}".format(test_precision))
                print("Total Test Recall {:.4f}".format(test_recall))
                print("Total Test F1 {:.4f}".format(test_f1))
                print("Total Test Accuracy {:.4f}".format(test_accuracy))

                print("\n###############################\n## FinalEvalRuntime:", round((time.time() - startTimeEval) / 60, 1), "min ##\n###############################\n")
                iterationTime = round((time.time() - startIterationTime) / 60, 1)
                print("\n###########################\n## IterRuntime:", iterationTime, "min ##\n###########################\n", flush=True)

                kendallTest.append(test_kendall)
                kendallTrain.append(train_kendall)
                pearsonTrain.append(train_corrPearson)
                spearmanTrain.append(train_corrSpearman)

                with open(summary, 'a') as f:
                    f.write('|'.join(map(str, train_indices)) + ',' + '|'.join(map(str, test_indices)) + "," + str(finalEpoch) + "," + str(iterationTime) + "," + str(maxMem) + "," + str(avergMem / finalEpoch))
                    f.write("," + "| ".join(theDataset.getNames()[i] for i in test_indices) + "," + str(train_kendall) + "," + str(test_kendall))
                    f.write("," + str(train_rmse) + "," + str(test_rmse))
                    f.write("," + str(train_corrPearson) + "," + str(test_corrPearson))
                    f.write("," + str(train_corrSpearman) + "," + str(test_corrSpearman))
                    f.write("," + str(train_precision) + "," + str(test_precision))
                    f.write("," + str(train_recall) + "," + str(test_recall))
                    f.write("," + str(train_f1) + "," + str(test_f1))
                    f.write("," + str(train_accuracy) + "," + str(test_accuracy) + "\n")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if FULLTRAIN:
                    break
                if DOKFOLD:
                    del model
                    del train_dataloader
                    del test_dataloader
                if FIXEDSPLIT:
                    break

                # K fold loop end here
                with open(summary, 'a') as f:
                    #TODO add average and std dev for new metrics (MSE, precision, recall, etc...) for Kfold
                    f.write(",,,,,,Average," + str(sum(kendallTrain) / len(kendallTrain)) + "," + str(sum(kendallTest) / len(kendallTest)) + "\n")
                    f.write(",,,,,,Median," + str(statistics.median(kendallTrain)) + "," + str(statistics.median(kendallTest)) + "\n")
                    f.write(",,,,,,Std Dev," + (str(statistics.stdev(kendallTrain)) if len(kendallTrain) > 1 else "N/A") + "," + (str(statistics.stdev(kendallTest)) if len(kendallTest) > 1 else "N/A") + "\n")

            if MIXEDTEST and LOADSECONDDS and not FUSIONDS:
                ##################################################################################
                ######################### MIXED TECHNOLOGY TESTING ###############################
                ##################################################################################

                #TODO: this is not tested after new metrics for TCAS1
                print( '######################\n###### MIXED TEST ######\n######################\n', flush = True )
                startTimeMixedTest = time.time()
                test_indices2 = [ i for i in range( len( secondDataset ) ) ]# if i !=2 and i !=4] # remove swerv and bp_be_top
                test_dataloader2  = GraphDataLoader( secondDataset, batch_size = 1 )

                #test_kendall2, test_corrPearson2, test_corrSpearman2    = evaluate_in_batches( test_dataloader2,  device, model )
                test_kendall2, test_rmse2, test_corrPearson2, test_corrSpearman2, test_precision2, test_recall2, test_f12, test_accuracy2 = evaluate_in_batches( test_dataloader2,  device, model )

                endTimeMixedTest = round( ( time.time() - startTimeMixedTest ) / 3600, 2 )
                with open( summary, 'a' ) as f:
                    f.write( "mixed test:"+ dsAbbreviated2 +"," )
                    f.write( '|'.join( map( str, test_indices2 ) ) + ',' )
                    f.write( ","+  str( endTimeMixedTest )+","+ "| ".join( secondDataset.getNames()[i] for i in test_indices2 ) +",,,mixedTest-"+ dsAbbreviated2 +","+ str( test_kendall2 ))
                    f.write( ",,"+ str( test_corrPearson2 ) )
                    f.write( ",,"+ str( test_corrSpearman2 )  +"\n" )
                print( "mixed test kendall ", dsAbbreviated2, test_kendall2 )
                print( "time mixed test:", endTimeMixedTest )
                if DRAWOUTPUTS and mainIteration == 0: 
                    for n in test_indices2:
                        g = secondDataset[ n ].to( device )
                        path = secondDataset.names[ n ]
                        path = imageOutput + "/mixedTest-" + path +"-idx"+ '|'.join( map( str, test_indices2 ) )+"-e"+str( finalEpoch )+"-feat"+'|'.join( abbreviatedFeatures )
                        evaluate_single( g, device, model, path ) #using only for drawing for now
            with open( ablationResult, 'a' ) as f:
                f.write( ","+ str( sum( kendallTrain ) / len( kendallTrain ) ) +","+ ( str( statistics.stdev( kendallTrain ) ) if len( kendallTrain ) > 1 else "N/A" ) )
                f.write( ","+ str( sum( kendallTest ) / len( kendallTest ) )   +","+ ( str( statistics.stdev( kendallTest ) )  if len( kendallTest )  > 1 else "N/A" ) )
                f.write( ","+ str( sum( pearsonTrain ) / len( pearsonTrain ) ) +","+ ( str( statistics.stdev( pearsonTrain ) ) if len( pearsonTrain ) > 1 else "N/A" ) )
                f.write( ","+ str( sum( spearmanTrain ) / len( spearmanTrain ) ) +","+ ( str( statistics.stdev( spearmanTrain ) ) if len( spearmanTrain ) > 1 else "N/A" ) )
                if not MIXEDTEST:
                    f.write( "\n" )
                else:
                    f.write( ","+ str( test_kendall2 ) + ","+ str( test_rmse2 ) + ","+ str( test_corrPearson2 ) +","+ str( test_corrSpearman2 ) + "\n" )
                
            folder_name = f"{str(abbreviatedFeatures)}-{mainIteration}"
            if DRAWOUTPUTS and mainIteration == 0:
                shutil.move( imageOutput, folder_name )
            del theDataset
            # ablation loop end here
            
        with open( ablationResult, 'a' ) as f:
            f.write("\n")
    
    endTimeAll = round( ( time.time() - startTimeAll ) / 3600, 2 )
    with open( summary, 'a' ) as f:
        f.write( ",,featCombinations:"+ str( len( ablationList ) )+"," + str( endTimeAll ) + " hours" ) 
    print("\n\n All finished, runtime:", endTimeAll, "hours" )

    folders = [folder for folder in os.listdir() if os.path.isdir(folder)]
    pattern = r'^\d+'
    counters = [int(re.match(pattern, folder).group()) for folder in folders if re.match(pattern, folder)]
    if counters:
        max_counter = max( counters )
        new_counter = max_counter + 1
    else:
        new_counter = 0
    folder_name = str(new_counter)
    os.mkdir(folder_name)
    excluded_folders = [ "nangateV2", "asap7V2", "nangateV1+closeness+between", "asap7V1+closeness+between", "nangate", "backup", "c17", "gcd", "regression.py", ".git", "toyDataset"]
    for item in os.listdir():
        print( "item:", item )
        if not re.match( pattern, item ) and item not in excluded_folders:
            shutil.move( item, os.path.join( folder_name, item ) )
    shutil.copy("regression.py", os.path.join(folder_name, "regression.py"))
    with open( 'log.log', 'w' ) as f:
        f.write('')


if __name__ == '__main__':
    print( "\n\n-------------------------------\n---------Run setup 1 (NG first DS)----------\n-------------------------\n\n" )
    runExperiment( runSetup )
    # if not FUSIONDS:
    #     print( "\n\n-------------------------------\n---------Run setup 2 (A7 first DS)----------\n-------------------------\n\n" )
    #     runExperiment( 2 )
