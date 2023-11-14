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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter #Graphical visualization
from torch.utils.data import DataLoader, RandomSampler

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.dataloading import DataLoader
from dgl.data.ppi import PPIDataset #TODO remove
import dgl.nn as dglnn
import networkx as nx #drawing graphs

from sklearn.metrics import r2_score, f1_score #Score metric
from sklearn.model_selection import KFold
from torchmetrics.regression import KendallRankCorrCoef #Same score as congestionNet

validFeatures = [ 'percolation', 'harmonic', 'information', 'subgraph', 'load', 'between', 'closeness', 'CFbetween', 'CFcloseness', 'eigen', 'pageRank', 'inDegree', 'outDegree', 'type', 'area', 'input_pins', 'output_pins' ]
mainMaxIter      = 5
FULLTRAIN        = True
DOKFOLD          = False
num_folds        = 2
MANUALABLATION   = True
feat2d = 'feat' 
stdCellFeats = [ 'type' ] #, 'area', 'input_pins', 'output_pins' ]
#fullAblationCombs = [ 'area', 'input_pins', 'output_pins', 'type', 'eigen', 'pageRank', 'inDegree', 'outDegree' ]  #, 'closeness', 'between' ] # logicDepth
fullAblationCombs = [ 'between' ]

            

labelName =  'routingHeat'
secondLabel = 'placementHeat'
dsFolderName = 'ng45-onlyETH'
MIXEDTEST     = False
dsFolderName2 = 'asap7'

maxEpochs = 2
minEpochs = 1
useEarlyStop = True
step      = 0.005
improvement_threshold = 0.000001 
patience = 35  # Number of epochs without improvement to stop training
accumulation_steps = 4

DOLEARN         = False
DRAWOUTPUTS     = False
DRAWCORRMATRIX  = False
DRAWGRAPHDATA   = True


DEBUG           = 0 #1 for evaluation inside train
CUDA            = True
SKIPFINALEVAL   = False #TODO True
SELF_LOOP = True
COLAB     = False

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


def process_and_write_csvs(paths):
    categorical_columns=["type"]
    columns_to_normalize = [ labelName ] #, 'area', 'input_pins', 'output_pins'  ] #+ stdCellFeats

    # Step 1: Initialize an empty dataframe to store all data
    master_df = pd.DataFrame()

    # Step 3: Read and concatenate all CSVs into the master dataframe
    for path in paths:
        csv_path = os.path.join(path, "gatesToHeat.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv( csv_path, index_col = 'id', dtype = { 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )
            master_df = pd.concat([master_df, df], ignore_index=True)

    master_df = master_df[master_df[labelName] > 0]
    # Step 2: Determine the min and max values for columns to normalize
    # min_values = 0 #master_df[columns_to_normalize].min()
    min_values = master_df[columns_to_normalize].replace(0, np.nan).min()
    max_values = master_df[columns_to_normalize].max()

    print("min and max for normalization:\nmin:", min_values, "\nmax", max_values )
    # Step 4: Determine the categorical mapping for the specified categorical columns
    categorical_mapping = {}
    for column in categorical_columns:
        categorical_mapping[column] = master_df[column].astype('category').cat.categories.tolist()

    # Step 5: Normalize the values in specified columns across all CSVs
    for path in paths:
        csv_path = os.path.join( path, "gatesToHeat.csv" )
        if os.path.isfile( csv_path ):
            df = pd.read_csv( csv_path, index_col = 'id', dtype = { 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )
            for column, categories in categorical_mapping.items():
                df[ column ] = pd.Categorical( df[ column ], categories = categories ).codes

            df[columns_to_normalize] = df[columns_to_normalize].replace(0, min_values)
            # Normalize specified columns using min-max scaling
            df[columns_to_normalize] = (df[columns_to_normalize] - min_values) / (max_values - min_values)

            # Step 6: Write the modified dataframe to a new CSV
            output_path = os.path.join(path, "preProcessedGatesToHeat.csv")
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
        # gateToHeat = pd.read_csv( path / 'gatesToHeat.csv', index_col = 'id', dtype = { 'type':'category' } )
        gateToHeat = pd.read_csv( path / 'gatesToHeat.csv', index_col = 'id', dtype = { 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'int64', 'output_pins':'int64', 'logic_function':'string' } )                    
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
        
    def drawSingleCorrelationMatrix( self, fileName ):
        print( "Start: drawSingleCorrelationMatrix!" )
        num_graphs = len(self.graphs)
        all_data = []
        for graph in self.graphs:
            tensor = graph.ndata[feat2d]
            reshape = graph.ndata[labelName].view(-1, 1)
            tensor = torch.cat((tensor, reshape), 1)
            all_data.append(tensor)
        combined_data = torch.cat(all_data, 0)
        correlation_matrix = np.corrcoef(combined_data, rowvar=False)
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(correlation_matrix, cmap='turbo', interpolation='nearest', vmin=-1, vmax=1)
        ax.set_title("Combined Correlation Matrix")
        ax.set_xticks(np.arange(len(self.namesOfFeatures) + 1))
        ax.set_yticks(np.arange(len(self.namesOfFeatures) + 1))
        ax.set_xticklabels(self.namesOfFeatures + [labelName], rotation=30)
        ax.set_yticklabels(self.namesOfFeatures + [labelName])        
        for j in range(len(self.namesOfFeatures) + 1):
            for k in range(len(self.namesOfFeatures) + 1):
                ax.text(k, j, format(correlation_matrix[j, k], ".1f"),
                        ha="center", va="center", color="white")
        fig.tight_layout()
        cbar = fig.colorbar(im)
        plt.savefig("corrMatrix-" + fileName + ".png")
        plt.close('all')
        plt.clf()
        print( "End: drawSingleCorrelationMatrix!" )
        
    def drawHeatCentrality(self, fileName, clip_min=None, clip_max=None):
        print("self.namesOfFeatures:", self.namesOfFeatures)
        for g in self.graphs:
            designName = g.name
            print("************* INDSIDE DRAWHEAT CENTRALITY *****************")
            print("Circuit:", designName)
            print("feat2d:", feat2d)
            print("g.ndata[feat2d]:", type(g.ndata[feat2d]), g.ndata[feat2d].shape, flush=True)
            positions = g.ndata["position"].to(torch.float32).to("cpu")
            feat_values = g.ndata[feat2d]
            num_columns = feat_values.shape[1]  # Get the number of columns in the 2D tensor
            if num_columns != len(self.namesOfFeatures):
                print("ERROR: namesOfFeatures and num_columns with different sizes in drawHeatCentrality!!")
            fig, axes = plt.subplots(1, num_columns, figsize=(6 * num_columns, 6))
            for i, ax in enumerate(axes):
                dummy_image = ax.imshow([[0, 1]], cmap='coolwarm')
                feat_values_i = feat_values[:, i]
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
            cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # Define colorbar position (adjust as needed)
            cmap = cm.RdYlGn  # Use the same colormap as your rectangles
            cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal')
            cbar.set_label('Reference Colorbar')
            plt.savefig(designName + "-" + fileName)
            plt.close('all')
            for ax in axes:
                ax.clear()
            plt.clf()

# sns.boxplot(data=torch.cat([feat[:, i] for feat in agg_features]).cpu().numpy())
# stats.probplot(torch.cat([feat[:, i] for feat in agg_features]).cpu().numpy(), plot=plt)
#sns.violinplot(data=torch.cat([feat[:, i] for feat in agg_features]).cpu().numpy(), inner="quartile")            
    def drawDataAnalysis(self, fileName):
        print("************* INDSIDE DRAW DATA ANALYSIS *****************", flush=True)
        agg_features = [graph.ndata[feat2d] for graph in self.graphs]
        agg_labels = [graph.ndata[labelName] for graph in self.graphs]

        print( "self.namesOfFeatures:", self.namesOfFeatures )
        print( "agg_features[0].shape:", agg_features[0].shape )
        print( "agg_features:", type( agg_features ), "\n", agg_features )
        min_max_data = []
        for i, feature_name in enumerate(self.namesOfFeatures):
            feature_values = torch.cat([feat[:, i] for feat in agg_features]).cpu().numpy()
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

        with open(f"graphLevelAnalysis_{fileName}.csv", mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(min_max_data)
            csv_writer.writerow(('Total_Num_Nodes', total_num_nodes))
            csv_writer.writerow(('Total_Num_Edges', total_num_edges))

        num_features = agg_features[0].shape[1]
        num_labels = 1  # Assuming there's only one label column
        num_plots = num_features + num_labels

        num_rows = math.ceil(math.sqrt(num_plots))
        num_cols = math.ceil(num_plots / num_rows)

        plt.figure(figsize=(12, 6))
        for i in range(num_features):
            plt.subplot(num_rows, num_cols, i + 1)
            sns.histplot(torch.cat([feat[:, i] for feat in agg_features]).cpu().numpy(), kde=True, bins=20)
            plt.xlabel(self.namesOfFeatures[i])
            plt.ylabel('Count')
            plt.title(self.namesOfFeatures[i])

        plt.subplot(num_rows, num_cols, num_plots)  # Label in a separate subplot
        sns.histplot(torch.cat(agg_labels).cpu().numpy(), kde=True)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Aggregated Labels')

        plt.suptitle('Aggregated Data Distribution for All Graphs')
        plt.tight_layout()
        plt.savefig("graphLevelAggregatedAnalysis-" + fileName + ".png")
        plt.close('all')
        plt.clf()

#QQ PLOT WORKING
    # def drawDataAnalysis(self, fileName):
    #     print("************* INDSIDE DRAW DATA ANALYSIS *****************", flush=True)
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
        

    def drawDataAnalysisForEachGraph(self, filePrefix):
        print("************* INSIDE DRAW DATA ANALYSIS FOR EACH GRAPH *****************", flush=True)
        for i, graph in enumerate(self.graphs):
            print("graph.name:", graph.name)
            agg_features = graph.ndata[feat2d]
            agg_labels = graph.ndata[labelName]
            num_features = agg_features.shape[1]
            num_labels = 1  # Assuming there's only one label column
            total_plots = num_features + num_labels
            num_cols = 2  # Number of columns for the subplot grid
            num_rows = (total_plots + num_cols - 1) // num_cols  # Calculate the number of rows dynamically

            plt.figure(figsize=(12, num_rows * 3))  # Adjust the figure size based on the number of rows

            for j in range(total_plots):
                plt.subplot(num_rows, num_cols, j + 1)
                if j < num_features:
                    sns.histplot(agg_features[:, j].cpu().numpy(), kde=True, bins=20)
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
                
    def centralityToCsv( self, designPath, centralityName ):
        node_data_attributes = self.graph.ndata.keys()
        combined_df = pd.DataFrame()
        for attribute_name in node_data_attributes:
            node_data = self.graph.ndata[ attribute_name ]
            df = pd.DataFrame( node_data.numpy() ) 
            df.columns = [ f"{attribute_name}_{i}-{centralityName}" for i in range( df.shape[1] ) ]
            combined_df = pd.concat([combined_df, df], axis=1)
        combined_df.to_csv(f'{designPath}/{self.graph.name}-{centralityName}.csv', index=True)
                
    def _process_single( self, designPath ):
        print( "\n\n########## PROCESS SINGLE #################" )
        print( "      Circuit:", str( designPath ).rsplit( '/' )[-1] )
        print( "###########################################\n" )
        print( "self.ablationFeatures (_process_single):", type( self.ablationFeatures ), len( self.ablationFeatures ), self.ablationFeatures )
        positions = pd.read_csv( str( designPath ) + "/gatesPosition_" + str( designPath ).rsplit( '/' )[-1] + ".csv"  )
        positions = positions.rename( columns = { "Name" : "name" } )
        # nodes_data = pd.read_csv( designPath / 'preProcessedGatesToHeat.csv', index_col = 'id' )
        nodes_data = pd.read_csv( designPath / 'preProcessedGatesToHeat.csv', index_col = 'id', dtype = { 'type':'int64', 'name':'string', 'conCount':'int64', 'routingHeat':'float64', 'area':'float64', 'input_pins':'float64', 'output_pins':'float64', 'logic_function':'string' } )                    
        nodes_data = nodes_data.sort_index()
        edges_data = pd.read_csv( designPath / 'DGLedges.csv')
        edges_src  = torch.from_numpy( edges_data['Src'].to_numpy() )
        edges_dst  = torch.from_numpy( edges_data['Dst'].to_numpy() )

        print( "BEFORE MERGE nodes_data:", nodes_data.shape )#, "\n", nodes_data )
        nodes_data = pd.merge( nodes_data, positions, on = "name" )
        print( "AFTER MERGE nodes_data:", nodes_data.shape )#, "\n", nodes_data )
        
        print( "self.ablationFeatures:", type( self.ablationFeatures ) )#, "\n", self.ablationFeatures )

        for column in self.ablationFeatures:
            if column not in nodes_data:
                nodes_data[ column ] = 0
        df = nodes_data[stdCellFeats + [labelName]].copy()  # Create a copy of the DataFrame
        df['id'] = nodes_data.index.values
        
        print("df.columns:",df.columns)
        df_wanted = ( df[ "type" ] >= 0 ) & ( df[ labelName] > 0 ) if 'type' in self.ablationFeatures else ( df[ labelName ] > 0 )
        df_wanted = ~df_wanted
        
        removedNodesMask = torch.tensor( df_wanted )
        print( "nodes_data:", type( nodes_data ), nodes_data.shape ) #, "\n", nodes_data )
        idsToRemove = torch.tensor( nodes_data.index )[ removedNodesMask ]
        print( "idsToRemove:", idsToRemove.shape ) #,"\n", torch.sort( idsToRemove ) )

    ###################### BUILD GRAPH #####################################        
        self.graph = dgl.graph( ( edges_src, edges_dst ), num_nodes = nodes_data.shape[0] )
        self.graph.name = str( designPath ).rsplit( '/' )[-1]

        for featStr in stdCellFeats:
            if featStr in self.ablationFeatures:
                self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, torch.tensor( nodes_data[ featStr ].values ) )
                if featStr not in self.namesOfFeatures:
                    self.namesOfFeatures.append( featStr )
        self.graph.ndata[ labelName  ] = ( torch.from_numpy ( nodes_data[ labelName   ].to_numpy() ) )
        self.graph.ndata[ "position" ] = torch.tensor( nodes_data[ [ "xMin","yMin","xMax","yMax" ] ].values )
        self.graph.ndata[ "id" ]       = torch.tensor( nodes_data.index.values )
################### REMOVE NODES #############################################
        print( "---> BEFORE REMOVED NODES:")
        print( "\tself.graph.nodes()", self.graph.nodes().shape )#, "\n", self.graph.nodes() )
        isolated_nodes_before = ( ( self.graph.in_degrees() == 0 ) & ( self.graph.out_degrees() == 0 ) ).nonzero().squeeze(1)
        print( "isolated nodes before any node removal:", isolated_nodes_before.shape )
        self.graph.remove_nodes( idsToRemove )
        isolated_nodes = ( ( self.graph.in_degrees() == 0 ) & ( self.graph.out_degrees() == 0 ) ).nonzero().squeeze(1)
        print( "isolated_nodes:", isolated_nodes.shape ) #, "\n", isolated_nodes )
        self.graph.remove_nodes( isolated_nodes )
        print( "\n---> AFTER REMOVED NODES:" )
        print( "\tself.graph.nodes()", self.graph.nodes().shape ) #, "\n", self.graph.nodes() )
        # check_graph = self.graph.to_networkx().to_undirected()
        # check_graph.remove_nodes_from( list(nx.isolates(check_graph)) )
        # print("nx.is_connected(check_graph)", nx.is_connected(check_graph))
        # del check_graph
################### PERCOLATION ######################################
        if any( "percolation" == s for s in self.ablationFeatures ):
            print( "calculating percolation!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            nx.set_node_attributes( nx_graph, 0.1, 'percolation')
            print( "nx_graph:\n", nx_graph, flush = True )
            percolation_scores = nx.percolation_centrality( nx_graph )
            percolation_scores_list = list( percolation_scores.values() )
            min_score = min( percolation_scores_list )
            max_score = max( percolation_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in percolation_scores_list ] 
            percolation_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, percolation_tensor )
            if 'percolation' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'percolation' )
            self.centralityToCsv( designPath, 'percolation' )
################### HARMONIC ######################################
        if any( "harmonic" == s for s in self.ablationFeatures ):
            print( "calculating harmonic!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            harmonic_scores = nx.harmonic_centrality( nx_graph )
            harmonic_scores_list = list( harmonic_scores.values() )
            min_score = min( harmonic_scores_list )
            max_score = max( harmonic_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in harmonic_scores_list ] 
            harmonic_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, harmonic_tensor )
            if 'harmonic' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'harmonic' )
            self.centralityToCsv( designPath, 'harmonic' )
################### SUBGRAPH ######################################
        if any( "subgraph" == s for s in self.ablationFeatures ):
            print( "calculating subgraph!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            subgraph_scores = nx.subgraph_centrality( nx_graph )
            subgraph_scores_list = list( subgraph_scores.values() )
            min_score = min( subgraph_scores_list )
            max_score = max( subgraph_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in subgraph_scores_list ] 
            subgraph_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, subgraph_tensor )
            if 'subgraph' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'subgraph' )
            self.centralityToCsv( designPath, 'subgraph' )        
################### INFORMATION  ######################################
        if any( "information" == s for s in self.ablationFeatures ):
            print( "calculating information!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            # information_scores = nx.information_centrality( nx_graph )
            information_scores = nx.approximate_current_flow_betweenness_centrality( nx_graph )
            information_scores_list = list( information_scores.values() )
            min_score = min( information_scores_list )
            max_score = max( information_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in information_scores_list ] 
            information_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, information_tensor )
            if 'information' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'information' )
            self.centralityToCsv( designPath, 'information' )            
################### CURRENT FLOW CLOSENESS ######################################
        if any( "CFcloseness" == s for s in self.ablationFeatures ):
            print( "calculating CFcloseness!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            CFcloseness_scores = nx.current_flow_closeness_centrality( nx_graph )
            CFcloseness_scores_list = list( CFcloseness_scores.values() )
            min_score = min( CFcloseness_scores_list )
            max_score = max( CFcloseness_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in CFcloseness_scores_list ] 
            CFcloseness_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, CFcloseness_tensor )
            if 'CFcloseness' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'CFcloseness' )
            self.centralityToCsv( designPath, 'CFcloseness' )
################### CLOSENESS  ################################################
        if any( "closeness" == s for s in self.ablationFeatures ):
            print( "calculating closeness!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            close_scores = nx.closeness_centrality( nx_graph )
            close_scores_list = list( close_scores.values() )
            min_score = min( close_scores_list )
            max_score = max( close_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in close_scores_list ] 
            close_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, close_tensor )
            if 'closeness' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'closeness' )
            self.centralityToCsv( designPath, 'closeness' )
################### CURRENT FLOW BETWEENNESS  ################################################
        if any( "CFbetween" == s for s in self.ablationFeatures ):
            print( "calculating CFbetween!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            CFbetween_scores = {} 
            components = list(nx.connected_components( nx_graph ))
            for component in components:
                subgraph = nx_graph.subgraph(component)  # Create a subgraph for the connected component
                current_flow_betweenness_subgraph = nx.current_flow_betweenness_centrality(subgraph)
                for node, value in current_flow_betweenness_subgraph.items():
                    CFbetween_scores[node] = value

            # CFbetween_scores = nx.current_flow_betweenness_centrality( nx_graph )
            CFbetween_scores_list = list( CFbetween_scores.values() )
            min_score = min( CFbetween_scores_list )
            max_score = max( CFbetween_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in CFbetween_scores_list ] 
            CFbetween_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, CFbetween_tensor )
            if 'CFbetween' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'CFbetween' )
            self.centralityToCsv( designPath, 'CFbetween' )
################### BETWEENNESS  ################################################
        # if 'between' in self.ablationFeatures:
        if any( "between" == s for s in self.ablationFeatures ):
            print( "calculating betweenness!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            between_scores = nx.betweenness_centrality( nx_graph )
            betweenness_list = list( between_scores.values() )
            min_score = min( betweenness_list )
            max_score = max( betweenness_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in betweenness_list ] 
            between_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, between_tensor )
            if 'betweenness' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'betweenness' )
            self.centralityToCsv( designPath, 'betweenness' )
#################### LOAD  ################################################
        if 'load' in self.ablationFeatures:
            print( "calculating load!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            load_scores = nx.load_centrality( nx_graph )
            load_scores_list = list( load_scores.values() )
            min_score = min( load_scores_list )
            max_score = max( load_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in load_scores_list ] 
            load_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, load_tensor )
            if 'load' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'load' )
            self.centralityToCsv( designPath, 'load' )                        
################### EIGENVECTOR  ################################################
        if 'eigen' in self.ablationFeatures:
            print( "calculating eigenvector!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            eigen_scores = nx.eigenvector_centrality_numpy( nx_graph, max_iter = 5000 )
            eigen_scores_list = list( eigen_scores.values() )
            min_score = min( eigen_scores_list )
            max_score = max( eigen_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in eigen_scores_list ] 
            eigen_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, eigen_tensor )
            if 'Eigen' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'Eigen' )
################### PAGE RANK ################################################    
        if 'pageRank' in self.ablationFeatures:
            nx_graph = self.graph.to_networkx()
            pagerank_scores = nx.pagerank(nx_graph)
            pagerank_scores_list = list( pagerank_scores.values() )
            min_score = min( pagerank_scores_list )
            max_score = max( pagerank_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in pagerank_scores_list ] 
            pagerank_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, pagerank_tensor )
            if 'pageRank' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'pageRank' )
################### IN DEGREE ################################################    
        if 'inDegree' in self.ablationFeatures:
            nx_graph = self.graph.to_networkx()
            inDegree_scores = nx.in_degree_centrality(nx_graph)
            inDegree_scores_list = list( inDegree_scores.values() )
            min_score = min( inDegree_scores_list )
            max_score = max( inDegree_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in inDegree_scores_list ] 
            inDegree_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, inDegree_tensor )
            if 'inDegree' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'inDegree' )
    ################### OUT DEGREE ################################################    
        if 'outDegree' in self.ablationFeatures:
            nx_graph = self.graph.to_networkx()
            outDegree_scores = nx.out_degree_centrality(nx_graph)
            outDegree_scores_list = list( outDegree_scores.values() )
            min_score = min( outDegree_scores_list )
            max_score = max( outDegree_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in outDegree_scores_list ] 
            outDegree_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, outDegree_tensor )
            if 'outDegree' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'outDegree' )
    ################### KATZ ################################################
        if 'katz' in self.ablationFeatures:
            print( "calculating katzvector!", flush = True )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            katz_scores = nx.katz_centrality( nx_graph, max_iter = 500, tol = 1.0e-5 )
            katz_scores_list = list( katz_scores.values() )
            min_score = min( katz_scores_list )
            max_score = max( katz_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in katz_scores_list ] 
            katz_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ feat2d ] = dynamicConcatenate( self.graph.ndata, katz_tensor )
            if 'Katz' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'Katz' )
        ###################################################################
        if SELF_LOOP:
            self.graph = dgl.add_self_loop( self.graph )           
        print( "self.ablationFeatures (_process_single):", type( self.ablationFeatures ), len( self.ablationFeatures ), self.ablationFeatures )
        print( "---> _process_single DONE.\n\tself.graph.ndata", type( self.graph ), "\n\t", self.graph.ndata, flush = True )
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
    print( "************* INDSIDE DRAWHEAT *****************" )
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


 
def evaluate( g, features, labels, model, path, device ):
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        if( features.dim() == 1 ):
            features = features.unsqueeze(1)
        predicted = model( g, features ) 
        #print("\t>>>> predicted before squeeze:", type(predicted), predicted.shape, "\n", predicted )
        predicted = predicted.squeeze(1)
#        print( "\t>>> features in evaluate:", type( features ), features.shape ) #"\n", list ( features ) )
        print( "\t>>>> labels   in evaluate:", type( labels ), labels.shape, labels[:10] ) #"\n", labels )
        print( "\t>>>> predicted after squeeze:", type( predicted ), predicted.shape, predicted[:10] )# "\n", predicted )

        # tp, tn, fp, fn = getPN( labels, predicted, 0.75 )
        # print( "tp, tn, fp, fn:", tp, tn, fp, fn )
        # f1 = getF1( tp, tn, fp, fn)
        # score_r2 = r2_score( labels.data.cpu(), predicted.data.cpu() )
        # f1 = np.float64(0.0)
        # score_r2 = np.float64(0.0)
        # print( "F1:", f1 )
        # print( "score_r2:", type(score_r2), score_r2 )
        
        kendall = KendallRankCorrCoef( variant = 'a' ).to( device )
        print( "calculating kendal...", flush=True )
        score_kendall = kendall( predicted, labels )
        print( "Kendall calculated", flush=True )

        predicted_cpu = predicted.cpu().detach().numpy()
        labels_cpu = labels.cpu().detach().numpy()
        corrPearson, pPearson = pearsonr( predicted_cpu, labels_cpu )
        print("Pearson correlation:", corrPearson,flush=True)
        corrSpearman, pSpearman = spearmanr( predicted_cpu, labels_cpu )
        print("Spearman correlation:", corrSpearman, flush=True)
        
        # corrPearson,  pPearson  = scipy.stats.pearsonr( predicted, labels )
        # corrSpearman, pSpearman = scipy.stats.spearmanr( predicted, labels )

        #print("score_kendall:", type( score_kendall ), str( score_kendall ), "\n", score_kendall,"\n\n")
        if len( path ) > 0:
            print( "\tdrawing output" )
            path = path +"k{:.4f}".format( score_kendall ) + ".png"
            if DRAWOUTPUTS:
                drawHeat( labels.to( torch.float32 ), predicted.to( torch.float32 ), path, g )
        memory_usage = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("memory usage in evaluate:", memory_usage )
        return score_kendall, corrPearson, pPearson, corrSpearman, pSpearman #score_r2, f1

def evaluate_in_batches( dataloader, device, model, image_path = "" ):
    total_kendall = 0.0
    total_corrPearson      = 0.0
    total_corrSpearman      = 0.0
    # names = dataloader.dataset.names
    # print("names in evaluate_in_batches:", names )
    for batch_id, batched_graph in enumerate( dataloader ):
        print( "batch_id (eval_in_batches):", batch_id )
        batched_graph = batched_graph.to( device )
        print( "batched_graph:", type( batched_graph ), batched_graph )
        print( "theDataset[ batch_id ]:", theDataset[ batch_id ] )
        features = batched_graph.ndata[ feat2d ].float().to( device )
        labels   = batched_graph.ndata[ labelName ].to( device )
        
#        print("features in evaluate_in_batches:", type(features), features.shape,"\n", features )
#        print("labels in evaluate_in_batches:", type(labels), labels.shape,"\n", labels )
        score_kendall, corrPearson,_, corrSpearman,_ = evaluate( batched_graph, features, labels, model, image_path, device )
        print( "partial Kendall (eval_in_batches):", score_kendall ) 
        total_kendall += score_kendall
        total_corrPearson      += corrPearson
        total_corrSpearman      += corrSpearman
    total_kendall =  total_kendall / (batch_id + 1)
    total_corrPearson      =  total_corrPearson / (batch_id + 1)
    total_corrSpearman      =  total_corrSpearman / (batch_id + 1)
    return total_kendall, total_corrPearson, total_corrSpearman # return average score

def evaluate_single( graph, device, model, path ):
    graph = graph.to( device )
    features = graph.ndata[ feat2d ].float().to( device )
    labels   = graph.ndata[ labelName ].to( device )
    print( "evaluate single--->", path )                               
    score_kendall, corrPearson, _, corrSpearman, _ = evaluate( graph, features, labels, model, path, device )
    print( "Single graph score - Kendall:", score_kendall, ", corrPearson:", corrSpearman )
    return score_kendall, corrSpearman, corrPearson


def train( train_dataloader, device, model, writerName ):
    print( "device in train:", device )
    writer = SummaryWriter( comment = writerName )

#    torch.cuda.reset_max_memory_allocated()
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
            memory_usage = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            max_memory_usage = max( max_memory_usage, memory_usage )
            accumulated_memory_usage += memory_usage
            
        average_loss = total_loss / len(train_dataloader)
        average_memory_usage = accumulated_memory_usage / len(train_dataloader)
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


if __name__ == '__main__':
    startTimeAll = time.time()
    imageOutput = "image_outputs"
    print(f'Training Yosys Dataset with DGL built-in GATConv module.')
    
    if CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    listDir = []	
    if COLAB:
        dsPath = '/content/drive/MyDrive/tese - datasets/dataSet'
    else:
        dsPath = Path.cwd() / dsFolderName
        
    for designPath in Path( dsPath ).iterdir():
	    if designPath.is_dir() and "runs" not in str( designPath ):
		    print("designPath:", designPath )
		    listDir.append( designPath )

    ##################################################################################
    ############################# Pre Processing #####################################
    ##################################################################################
    writeDFrameData( listDir, 'gatesToHeat.csv', "DSinfoBeforePreProcess.csv" )
    df = aggregateData( listDir, 'gatesToHeat.csv' )
    print( "\n\n#######################\n## BEFORE PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
    for col in df:
        print( "describe:\n", df[ col ].describe() )
    df.to_csv( "aggregatedDFBefore-pandasLevel.csv" )
    df = df.drop( df.index[ df[ labelName ] < 0 ] )
    df.hist( bins = 50, figsize = (15,12) )
    plt.savefig( "BeforePreProcess-pandasLevel" )
    plt.close( 'all' )
    plt.clf()

    process_and_write_csvs( listDir )
    #preProcessData( listDir )
	            
    writeDFrameData( listDir, 'preProcessedGatesToHeat.csv', "DSinfoAfterPreProcess.csv" )
    df = aggregateData( listDir, 'preProcessedGatesToHeat.csv' )
    print( "\n\n#######################\n## AFTER PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
    for col in df:
	    print( "\n####describe:\n", df[ col ].describe() )
    df.to_csv( "aggregatedDFAfter-pandasLevel.csv" )
    df = df.drop( df.index[ df[ labelName ] < 0 ])
    #df = df.drop( df.index[ df[ rawFeatName ] < 0 ] )
    df.hist( bins = 50, figsize = (15,12) )
    plt.savefig( "AfterPreProcess-pandasLevel" )
    plt.close( 'all' )
    plt.clf()
    ##################################################################################
    ##################################################################################
    ##################################################################################
    
    summary = "runSummary.csv"
    ablationResult = "ablationResult.csv"
    with open( summary, 'w' ) as f:
        f.write("")
    with open( ablationResult, 'w' ) as f:
        f.write( "Features,Train-Mean,Train-SD,Test-Mean,Test-SD,TrainPearson-Mean,TrainPearson-SD,TrainSpearman-Mean,TrainSpearman-SD\n" ) 
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
        for combAux in range( 1, len( fullAblationCombs ) + 1 ):
            print( "iteration:", combAux, ", combinations:", len( list( combinations( fullAblationCombs, combAux ) ) ) )
            ablationList += list( combinations( fullAblationCombs, combAux ) )
            print( "ablationList:", len( ablationList ), ablationList )
    else:
        # ablationList = [('area', 'input_pins', 'output_pins', 'type', 'eigen', 'pageRank', 'inDegree', 'outDegree') ]
        ablationList = [ ('between',), ('closeness',) ]# [ ( 'inDegree', 'outDegree', 'input_pins', 'output_pins' ) ] #('outDegree',), ('inDegree',), ('input_pins',), ('output_pins',), ('inDegree','outDegree'), ('input_pins','output_pins') ]
    for item in ablationList:
        for sub_item in item:
            if sub_item not in validFeatures:
                print(f"Error: '{sub_item}' is not in validFeatures.")
                sys.exit()
            
    print( "MANUALABLATION:", MANUALABLATION )
    print( "ablationList:", len( ablationList ), ablationList )
    for mainIteration in range( 0, mainMaxIter ):
        print( "##################################################################################" )
        print( "########################## NEW MAIN RUN  ########################################" )
        print( "##################################################################################" )
        print( "mainIteration:", mainIteration )
        print( "--> combination_list:", len( ablationList ), ablationList )
        ablationKendalls = []
        for ablationIter in ablationList:
            with open( summary, 'a' ) as f:
                f.write( "DSfolderName: " + dsFolderName+ ",MIXEDTEST:" )
                if MIXEDTEST:
                    f.write( dsFolderName2 )
                else:
                    f.write( "False" )
                f.write( ",#Circuits:" + str( len( listDir ) ) )
                f.write( ",minEpochs:" + str( minEpochs ) )
                f.write( ",maxEpochs:" + str( maxEpochs ) )
                f.write( ",step:" + str( round( step, 5 ) ) )
                f.write( ",labelName:" + labelName )
                f.write( ",features: " ) 
                f.write( "; ".join( ablationIter ) )
                f.write( ",FULLTRAIN: " + str( FULLTRAIN ) )
                f.write( ",MANUALABLATION:" + str( MANUALABLATION ) )
                f.write( ",improvement_threshold:" + str( improvement_threshold ) )
                f.write( ",patience:" + str( patience ) )
                f.write( "\ntrainIndices,testIndices,finalEpoch,runtime(min),MaxMemory,AverageMemory, Circuit Test, TrainKendall, TestKendall, TrainPearson, TestPearson, TrainSpearman, TestSpearman\n" )
            with open( ablationResult, 'a' ) as f:
                # copied_list = [s[:1].capitalize() for s in ablationIter]
                copied_list = [s for s in ablationIter]
                if len( copied_list ) > 1:
                    f.write( "; ".join( copied_list ) )
                else:
                    f.write( str( copied_list ) )
            print( "%%%%%%%%%%%%%%%%%%%%%%%%%%\nablationIter:", type( ablationIter ), len( ablationIter ), ablationIter, "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%", flush = True )
            ablationIter = list( ablationIter )
            if os.path.exists( imageOutput ):
                shutil.rmtree( imageOutput )
            if DRAWOUTPUTS:
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
            

            theDataset = DataSetFromYosys( listDir, ablationIter )#, mode='train' )
            if DRAWCORRMATRIX:
                theDataset.drawSingleCorrelationMatrix( dsFolderName+"-"+str( ablationIter ) )
                # theDataset.drawCorrelationPerGraph( dsFolderName+"-"+str( ablationIter ) )
            if DRAWGRAPHDATA and not FULLTRAIN:
                #theDataset.drawHeatCentrality( dsFolderName+"-"+str( ablationIter ) )
                theDataset.drawDataAnalysis( dsFolderName+"-"+str( ablationIter ) )
                theDataset.drawDataAnalysisForEachGraph( dsFolderName+"-"+str( ablationIter ) )
            if not DOLEARN:
                sys.exit()
            #TODO HOW TO IMPLEMENT THIS ??
            # if DOKFOLD:
            kf = KFold( n_splits = num_folds )
            for fold, ( train_indices, test_indices ) in enumerate( kf.split( theDataset ) ):
            # if not DOKFOLD: # MANUAL SPLIT
            #     fold = 0
            #     # LASCAS comparison, same as HUAWEI
            #     train_indices = [i for i in range(len(theDataset)) if i !=2 and i !=4] # remove swerv and bp_be_top
            #     test_indices = [2]

                # HUAWEI's ablation, only Black_parrot
                # train_indices = [7] 
                # test_indices = [i for i in range(len(theDataset)) if i != 7 and i != 4 ]  #remove bp_be_top and black_parrot
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
                # model = GAT( in_size, 128, out_size, heads=[4,4,6]).to( device )
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
                writerName += "- " + str( ablationIter ) + "-" + str( mainIteration )
                writerName += " T-"+ ';'.join( theDataset.getNames()[i] for i in test_indices )
                finalEpoch, maxMem, avergMem = train( train_dataloader, device, model, writerName )
                finalEpoch += 1

                print( '######################\n## Final Evaluation ##\n######################\n', flush = True )
                startTimeEval = time.time()
                if not SKIPFINALEVAL:
                    if not FULLTRAIN:
                        test_kendall, test_corrPearson, test_corrSpearman    = evaluate_in_batches( test_dataloader,  device, model )
                    else:
                        test_kendall = test_corrPearson = test_corrSpearman = torch.tensor([0]) #valid_kendall = valid_corrPearson = valid_corrSpearman = 
                    train_kendall, train_corrPearson, train_corrSpearman = evaluate_in_batches( train_dataloader, device, model )

                    # TODO: improve this, problem when accessing each graph name with batched graphs
                    if DRAWOUTPUTS: 
                        for n in test_indices:
                            g = theDataset[ n ].to( device )
                            path = theDataset.names[ n ]
                            path = imageOutput + "/test-" + path +"-testIndex"+ str(test_indices)+"-e"+str(finalEpoch)+"-feat"+str(ablationIter)
                            evaluate_single( g, device, model, path ) #using only for drawing for now
                        for n in train_indices:
                            g = theDataset[ n ].to( device )
                            path = theDataset.names[ n ]
                            path = imageOutput + "/train-" + path +"-trainIndex"+ str(train_indices)+"-e"+str(finalEpoch)+"-feat"+str(ablationIter)
                            evaluate_single( g, device, model, path ) #using only for drawing for now
                else:
                    test_kendall= test_corrPearson= test_corrSpearman= train_kendall= train_corrPearson= train_corrSpearman= torch.tensor([0]) #valid_kendall= valid_corrPearson= valid_corrSpearman= 

                print( "Total Train Kendall {:.4f}".format( train_kendall ) )
                print( "Total Train CORRPEARSON {:.4f}".format( train_corrPearson ) )
                print( "Total Train corrSpearman {:.4f}".format( train_corrSpearman ) )
                print( "\n###############################\n## FinalEvalRuntime:", round( ( time.time() - startTimeEval ) / 60, 1) , "min ##\n###############################\n" )
                iterationTime = round( ( time.time() - startIterationTime ) / 60, 1 )
                print( "\n###########################\n## IterRuntime:", iterationTime, "min ##\n###########################\n", flush = True )

                kendallTest.append ( test_kendall.item()  )
                kendallTrain.append( train_kendall.item() )
                pearsonTrain.append( train_corrPearson.item() )
                spearmanTrain.append( train_corrSpearman.item() )
                with open( summary, 'a' ) as f:
                    f.write( str( train_indices ).replace(',', '') +","+ str( test_indices ).replace(',', ';') + ","+str( finalEpoch )+","+str( iterationTime )+","+str( maxMem )+","+str( avergMem / finalEpoch )+"," )
                    f.write( ","+ "; ".join( theDataset.getNames()[i] for i in test_indices ) +","+ str( train_kendall.item() ) +","+ str( test_kendall.item() ))
                    f.write( "," + str( train_corrPearson.item() ) +","+ str( test_corrPearson.item() ) )  #+"\n")
                    f.write( "," + str( train_corrSpearman.item() ) +","+ str( test_corrSpearman.item() )  +"\n")

                # del model
                # del train_dataloader
                # del test_dataloader
                torch.cuda.empty_cache()
                if FULLTRAIN and not DOKFOLD:
                    break
                    break
                # K fold loop end here
            if MIXEDTEST:
                listDir2 = []	
                dsPath2 = Path.cwd() / dsFolderName2    
                for designPath in Path( dsPath2 ).iterdir():
                        if designPath.is_dir() and "runs" not in str( designPath ):
                                print("designPath:", designPath )
                                listDir2.append( designPath )

                ##################################################################################
                ############################# Pre Processing #####################################
                ##################################################################################
                writeDFrameData( listDir2, 'gatesToHeat.csv', "DSinfoBeforePreProcess.csv" )
                df = aggregateData( listDir2, 'gatesToHeat.csv' )
                print( "\n\n#######################\n## BEFORE PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
                for col in df:
                    print( "describe:\n", df[ col ].describe() )
                df.to_csv( "aggregatedDFBefore.csv" )
                df = df.drop( df.index[ df[ labelName ] < 0 ] )
                df.hist( bins = 50, figsize = (15,12) )
                # plt.savefig( "BeforePreProcess-" )
                # plt.close( 'all' )
                # plt.clf()

                preProcessData( listDir2 )

                writeDFrameData( listDir2, 'preProcessedGatesToHeat.csv', "DSinfoAfterPreProcess.csv" )
                df = aggregateData( listDir2, 'preProcessedGatesToHeat.csv' )
                print( "\n\n#######################\n## AFTER PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
                for col in df:
                        print( "\n####describe:\n", df[ col ].describe() )
                df.to_csv( "aggregatedDFAfter.csv" )
                df = df.drop( df.index[ df[ labelName ] < 0 ])
                df = df.drop( df.index[ df[ 'type' ] < 0 ] )
                # df = df.drop( df.index[ df[ rawFeatName ] < 0 ] )
                df.hist( bins = 50, figsize = (15,12) )
                # plt.savefig( "AfterPreProcess-train+valid+test" )
                # plt.close( 'all' )
                # plt.clf()
                theDataset2 = DataSetFromYosys( listDir2, ablationIter )#, mode='train' )
                test_indices2 = [i for i in range(len(theDataset)) ]# if i !=2 and i !=4] # remove swerv and bp_be_top
                # test_indices2 = [2]
                test_dataloader2  = GraphDataLoader( torch.utils.data.dataset.Subset( theDataset2, test_indices2  ), batch_size = 1 )
                test_kendall2, test_corrPearson2, test_corrSpearman2    = evaluate_in_batches( test_dataloader2,  device, model )
                print( "super new", test_kendall2 )

            #TODO averages for spearman and pearson
            with open( summary, 'a' ) as f:
                f.write( ",,,,,,,,Average," + str( sum( kendallTrain ) / len( kendallTrain ) ) +","+ str( sum( kendallTest ) / len( kendallTest ) ) + "\n" )
                f.write( ",,,,,,,,Median,"  + str( statistics.median( kendallTrain ) ) +","+ str( statistics.median( kendallTest ) ) +"\n" )
                f.write( ",,,,,,,,Std Dev," + ( str( statistics.stdev( kendallTrain ) ) if len( kendallTrain ) > 1 else "N/A" ) +","+ ( str( statistics.stdev( kendallTest ) ) if len( kendallTest ) > 1 else "N/A" ) +"\n" )
            with open( ablationResult, 'a' ) as f:
                f.write( ","+ str( sum( kendallTrain ) / len( kendallTrain ) ) +","+ ( str( statistics.stdev( kendallTrain ) ) if len( kendallTrain ) > 1 else "N/A" ) )
                f.write( ","+ str( sum( kendallTest ) / len( kendallTest ) )   +","+ ( str( statistics.stdev( kendallTest ) )  if len( kendallTest )  > 1 else "N/A" ) )
                f.write( ","+ str( sum( pearsonTrain ) / len( pearsonTrain ) ) +","+ ( str( statistics.stdev( pearsonTrain ) ) if len( pearsonTrain ) > 1 else "N/A" ) )
                f.write( ","+ str( sum( spearmanTrain ) / len( spearmanTrain ) ) +","+ ( str( statistics.stdev( spearmanTrain ) ) if len( spearmanTrain ) > 1 else "N/A" ) + "\n" )
                
            folder_name = f"{str(ablationIter)}-{mainIteration}"
            if DRAWOUTPUTS:
                shutil.move( imageOutput, folder_name )
            del theDataset
            # ablation loop end here
            
        with open( ablationResult, 'a' ) as f:
            f.write("\n")
    
    endTimeAll = round( ( time.time() - startTimeAll ) / 3600, 2 )
    with open( summary, 'a' ) as f:
        f.write( ",,,featCombinations:"+ str( len( ablationList ) )+"," + str( endTimeAll ) + " hours" ) 
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
    excluded_folders = ["nangate-STDfeatures-missing-bpQuad-memPool", "nangate", "backup", "c17", "gcd", "regression.py", ".git", "toyDataset", "asap7", "nangateV1" ]
    for item in os.listdir():
        print( "item:", item )
        if not re.match( pattern, item ) and item not in excluded_folders:
            shutil.move( item, os.path.join( folder_name, item ) )
    shutil.copy("regression.py", os.path.join(folder_name, "regression.py"))
    with open( 'log.log', 'w' ) as f:
        f.write('')
