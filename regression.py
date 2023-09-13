import os
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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import matplotlib.colorbar as mcb
import matplotlib.colors as mcolors

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

print( "torch.cuda.is_available():", torch.cuda.is_available() )
print( "torch.cuda.device_count():", torch.cuda.device_count() )
print( "torch.cuda.device(0):", torch.cuda.device(0) )
print( "torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0) )
print( "dgl.__version_:", dgl.__version__ )

mainMaxIter      = 2
FULLTRAIN        = False # False = Kfold
num_folds        = 4 
MANUALABLATION   = True
listFeats = [ 'eigen', 'pageRank', 'inDegree', 'outDegree', 'type' ] #, 'closeness', 'between' ] # logicDepth
featName = 'feat' #listFeats[0]
rawFeatName = 'type' #TODO: change to listFeats[0]

labelName =  'routingHeat'
secondLabel = 'placementHeat'

maxEpochs = 2
minEpochs = 1
useEarlyStop = True
step      = 0.005
improvement_threshold = 0.000001 
patience = 35  # Number of epochs without improvement to stop training
#split = [ 0.9, 0.05, 0.05 ]
# TandV = 1
accumulation_steps = 4

DEBUG         = 0 #1 for evaluation inside train
DRAWOUTPUTS   = False
CUDA          = True
DOLEARN       = True
SKIPFINALEVAL = False #TODO True

SELF_LOOP = True
COLAB     = False


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
    if featName in featTensor:
        if featTensor[ featName ].dim() == 1:
            ret = torch.cat( ( featTensor[ featName ].unsqueeze(1), tensor2.unsqueeze(1) ), dim = 1 )
        else:
            ret = torch.cat( ( featTensor[ featName ], tensor2.unsqueeze(1) ), dim = 1 )
    else:
        ret = tensor2
    #print( "ret:", ret )
    return ret

nameToHeat  =     {}
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
        gateToHeat = pd.read_csv( path / 'gatesToHeat.csv', index_col = 'id', dtype = { 'type':'category' } )
        # nametoHeat = dict( zip( gateToHeat[ 'name' ], gateToHeat[ labelName ] ) )
        labelsAux = pd.concat( [ labelsAux, gateToHeat[ labelName ] ], names = [ labelName ] )
        graphs[ path ] = gateToHeat#.append( gateToHeat )
        for cellType in gateToHeat[ rawFeatName ]:
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
    print( "\n\n\ntypeToNameCat:\n", typeToNameCat, "size:", len( typeToNameCat  ) )
    print( "\n\n\ntypeToSize:\n", typeToSize, "size:", len( typeToSize  ) )

    for key, g in graphs.items():
	#    df = g[ labelName ]
    #		df = df.loc[ df >= 0 ]
    #		dfMin = float( df.min() )
    #		dfMax = float( df.max() )
    #		labelToStandard.clear()
    #		for k in df:
    #			#print( "label:",label,"k:",k )
    #			if k not in labelToStandard: # and k >= 0:
    #			    labelToStandard[ k ] = ( k - dfMin ) / ( dfMax - dfMin )# 0 to 1
    #			#                labelToStandard[ k ] = ( k - series.mean() ) / series.std() # z value

	    #g[ 'size' ]    = g[ rawFeatName  ].replace( typeToSize )
	    g[ rawFeatName ]  = g[ rawFeatName  ].replace( typeToNameCat )
	    g[ labelName ] = g[ labelName ].replace( labelToStandard )
	    
	    #print("\n->g:\n",g)
    #        print( "\n\n###########\n###########\ng[ featName ]:\n", g[ featName ] )
    #        print( "\ng[ labelName ]:\n", g[ labelName ] )
	    g.to_csv( key / 'preProcessedGatesToHeat.csv' )
        
        
     
def aggregateData( listDir, csvName ):
    aggregatedDF = pd.DataFrame()
    for path in listDir:
        inputData = pd.read_csv( path / csvName )
        #TODO: new features are not considered here
        inputData = inputData[ [ rawFeatName, labelName, secondLabel ] ]
#        print( "inputData before concat:\n", inputData )
        aggregatedDF = pd.concat( [ aggregatedDF, inputData ] )
#    aggregatedDF.set_index( 'type' )
    return aggregatedDF
    
def writeDFrameData( listDir, csvName, outName ):
	with open( outName, 'w' ) as f:
		f.write( "IC name,#gates,#edges,"+labelName+"Min,"+labelName+"Max,"+secondLabel+"Min,"+secondLabel+"Max\n" )
		for path in listDir:
			inputData = pd.read_csv( path / csvName )
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
        self.graphPaths = listDir
        self.names      = self.allNames

        super().__init__( name='mydata_from_yosys' )

    def drawCorrelationMatrices( self ):
        num_graphs = len( self.graphs )
        num_rows = int( np.sqrt( num_graphs ) )
        num_cols = int( np.ceil( num_graphs / num_rows ) )

        fig, axes = plt.subplots( num_rows, num_cols, figsize = ( 12, 8 ) )
        for i, graph in enumerate( self.graphs ):
            tensor = graph.ndata[ featName ]
            print( "feat :", graph.ndata[ featName ].shape, graph.ndata[ featName ] )
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
        plt.savefig( "correlation-individual.png" )
        plt.close( 'all' )
        plt.clf()


        # # concatenated_tensor = np.hstack( [ graph.ndata[ featName ].T for graph in self.graphs ] )
        # tensors = [graph.ndata[featName] for graph in self.graphs]
        # num_features = tensors[0].shape[1]
        # print( "num_features:", num_features )
        # # concatenated_tensor = np.hstack(tensors)
        # # correlation_matrix = np.corrcoef( concatenated_tensor, rowvar = False )
        # concatenated_tensor = torch.cat( tensors, dim=1)
        # transposed_tensor = concatenated_tensor.T
        # correlation_coefficient = torch.corrcoef( transposed_tensor[0], transposed_tensor[1] )[0, 1]
        # plt.imshow( correlation_matrix, cmap = 'viridis', interpolation = 'nearest' )
        # plt.colorbar()
        # plt.title( "Combined Correlation Matrix" )
        # plt.xticks(np.arange(len(self.namesOfFeatures) * len(self.graphs)), self.namesOfFeatures * len(self.graphs), rotation=45, ha='right')
        # plt.yticks(np.arange(len(self.namesOfFeatures) * len(self.graphs)), self.namesOfFeatures * len(self.graphs))
        # for i in range(len(self.namesOfFeatures) * len(self.graphs)):
        #     for j in range(len(self.namesOfFeatures) * len(self.graphs)):
        #         plt.text(j, i, format(correlation_matrix[i, j], ".2f"),
        #                  ha="center", va="center", color="white")
        # plt.savefig( "correlation-combined.png" )
        # plt.close( 'all' )
        # plt.clf()
        
    def process( self ):
        for path in self.graphPaths:
	        graph = self._process_single( path )
	        self.graphs.append( graph )

    def _process_single( self, designPath ):
        print( "\n\n########## PROCESS SINGLE #################" )
        print( "      Circuit:", str( designPath ).rsplit( '/' )[-1] )
        print( "###########################################\n" )
        print( "self.ablationFeatures (_process_single):", type( self.ablationFeatures ), len( self.ablationFeatures ), self.ablationFeatures )
        positions = pd.read_csv( str( designPath ) + "/gatesPosition_" + str( designPath ).rsplit( '/' )[-1] + ".csv"  )
        positions = positions.rename( columns = { "Name" : "name" } )
        nodes_data = pd.read_csv( designPath / 'preProcessedGatesToHeat.csv', index_col = 'id' )
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
        #nodes_data.update(pd.DataFrame({column: 0 for column in self.ablationFeatures if column not in nodes_data}, index=[0]))
        df = nodes_data[ [ rawFeatName ] + [ labelName ] ]
        
        # if rawFeatName in self.ablationFeatures:
        #     df_wanted = np.logical_and ( np.where( df[ rawFeatName ] > 0, True, False ), np.where( df[ labelName ] > 0, True, False ) )
        # else:
        #     df_wanted = np.where( df[ labelName ] > 0, True, False )
        # df_wanted = np.invert( df_wanted )
        df_wanted = (df[rawFeatName] > 0) & (df[labelName] >= 0) if rawFeatName in self.ablationFeatures else (df[labelName] >= 0)
        df_wanted = ~df_wanted
        
    #		print( "df_wanted:", df_wanted.shape, "\n", df_wanted )
        removedNodesMask = torch.tensor( df_wanted )
    #		print( "removedNodesMask:", removedNodesMask.shape )#, "\n", removedNodesMask )
        print( "nodes_data:", type( nodes_data ), nodes_data.shape ) #, "\n", nodes_data )
        idsToRemove = torch.tensor( nodes_data.index )[ removedNodesMask ]
        print( "idsToRemove:", idsToRemove.shape ) #,"\n", torch.sort( idsToRemove ) )

    ###################### BUILD GRAPH #####################################        
        self.graph = dgl.graph( ( edges_src, edges_dst ), num_nodes = nodes_data.shape[0] )
        self.graph.name = str( designPath ).rsplit( '/' )[-1]
        # self.graph.ndata[ featName ] =  torch.tensor( nodes_data[ listFeats ]. )
        # self.graph.ndata[ featName ] =  torch.tensor( nodes_data[ self.ablationFeatures ].values )
        if rawFeatName in self.ablationFeatures:
            self.graph.ndata[ featName ] =  torch.tensor( nodes_data[ rawFeatName ].values )
            if rawFeatName not in self.namesOfFeatures:
                self.namesOfFeatures.append( rawFeatName )
        self.graph.ndata[ labelName  ]  = ( torch.from_numpy ( nodes_data[ labelName   ].to_numpy() ) )
        self.graph.ndata[ "position" ] = torch.tensor( nodes_data[ [ "xMin","yMin","xMax","yMax" ] ].values )
    ################### REMOVE NODES #############################################
        print( "---> BEFORE REMOVED NODES:")
        print( "\tself.graph.nodes()", self.graph.nodes().shape )#, "\n", self.graph.nodes() )
        #print( "\tself.graph.ndata\n", self.graph.ndata )		
        self.graph.remove_nodes( idsToRemove )
        # self.drawData = self.drawData.drop( idsToRemove.tolist(), axis=1 )        
        isolated_nodes = ( ( self.graph.in_degrees() == 0 ) & ( self.graph.out_degrees() == 0 ) ).nonzero().squeeze(1)
        print( "isolated_nodes:", isolated_nodes.shape ) #, "\n", isolated_nodes )
        self.graph.remove_nodes( isolated_nodes )
        #self.drawData = self.drawData.drop( isolated_nodes.tolist(), axis=1 )
        print( "\n---> AFTER REMOVED NODES:" )
        print( "\tself.graph.nodes()", self.graph.nodes().shape ) #, "\n", self.graph.nodes() )
        #print( "\tself.graph.ndata\n", self.graph.ndata )
        # print( "DRAW DATA:\n", self.drawData )
    ################### CLOSENESS  ################################################
        #drawGraph( self.graph, self.graph.name )
        if 'closeness' in self.ablationFeatures:
            print( "calculating closeness!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            # print( ".nodes:\n", nx_graph.nodes(data=True))
            # print( ".edges:\n", nx_graph.edges(data=True))
            close_scores = nx.closeness_centrality( nx_graph )
            # close_scores = nx.incremental_closeness_centrality( nx_graph )
            close_scores_list = list( close_scores.values() )
            min_score = min( close_scores_list )
            max_score = max( close_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in close_scores_list ] 
            close_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ featName ] = dynamicConcatenate( self.graph.ndata, close_tensor )
            if 'Closeness' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'Closeness' )
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
            self.graph.ndata[ featName ] = dynamicConcatenate( self.graph.ndata, eigen_tensor )
            if 'Eigen' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'Eigen' )
    ################### GROUP BETWEENNESS  ################################################
        #drawGraph( self.graph, self.graph.name )
        if 'between' in self.ablationFeatures:
            print( "calculating group betweenness!" )
            aux_graph = self.graph.to_networkx()
            nx_graph  = nx.Graph( aux_graph )
            print( "nx_graph:\n", nx_graph, flush = True )
            # print( ".nodes:\n", nx_graph.nodes(data=True))
            # print( ".edges:\n", nx_graph.edges(data=True))

            group_betweenness = {}
            group_distance = 3
            for node in nx_graph.nodes():
                print( "node", node, flush = True )
                subgraph = nx.ego_graph( nx_graph, node, radius = group_distance )
                # group_betweenness[node] = nx.group_betweenness_centrality( nx_graph, subgraph )
                if nx.is_connected( subgraph ) and len( subgraph ) > 2:
                    group_betweenness[node] = nx.group_betweenness_centrality(nx_graph, subgraph)
                else:
                    group_betweenness[node] = {}

            for node, centrality in group_betweenness.items():
                print(f"Node {node}:")
                if isinstance(centrality, float):
                    print(f"  Group: {centrality}")
                else:
                    for group, centrality_score in centrality.items():
                        print(f"  Group {group}: {centrality_score}")
            group_betweenness_list = list( group_betweenness.values() )
            min_score = min( group_betweenness_list )
            max_score = max( group_betweenness_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in group_betweenness_list ] 
            between_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ featName ] = dynamicConcatenate( self.graph.ndata, between_tensor )
            if 'Betweenness' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'Betweenness' )
    ################### PAGE RANK ################################################    
        if 'pageRank' in self.ablationFeatures:
            nx_graph = self.graph.to_networkx()
            pagerank_scores = nx.pagerank(nx_graph)
            pagerank_scores_list = list( pagerank_scores.values() )
            min_score = min( pagerank_scores_list )
            max_score = max( pagerank_scores_list )
            normalized_scores = [ ( score - min_score ) / ( max_score - min_score ) for score in pagerank_scores_list ] 
            pagerank_tensor = torch.tensor( normalized_scores )
            self.graph.ndata[ featName ] = dynamicConcatenate( self.graph.ndata, pagerank_tensor )
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
            self.graph.ndata[ featName ] = dynamicConcatenate( self.graph.ndata, inDegree_tensor )
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
            self.graph.ndata[ featName ] = dynamicConcatenate( self.graph.ndata, outDegree_tensor )
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
            self.graph.ndata[ featName ] = dynamicConcatenate( self.graph.ndata, katz_tensor )
            if 'Katz' not in self.namesOfFeatures:
                self.namesOfFeatures.append( 'Katz' )
        ###################################################################
        if SELF_LOOP:
            self.graph = dgl.add_self_loop( self.graph )           
        print( "self.ablationFeatures (_process_single):", type( self.ablationFeatures ), len( self.ablationFeatures ), self.ablationFeatures )
        print( "---> _process_single DONE.\n\tself.graph.ndata", type( self.graph ), "\n", self.graph.ndata, flush = True )
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
    #         self.graph.ndata[featName] = dynamicConcatenate(self.graph.ndata, torch.tensor(depths))

           
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

################################# VALUES PLOT ########################################################
    # WORKING!
    # types = graph.ndata[ featName ].to( torch.float32 ).to( "cpu" )
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
        f1 = np.float64(0.0)
        score_r2 = np.float64(0.0)
        # print( "F1:", f1 )
        # print( "score_r2:", type(score_r2), score_r2 )
        
        kendall = KendallRankCorrCoef( variant = 'a' ).to( device )
        print( "calculating kendal..." )
        score_kendall = kendall( predicted, labels )
        print( "Kendall calculated" )
        #print("score_kendall:", type( score_kendall ), str( score_kendall ), "\n", score_kendall,"\n\n")
        if len( path ) > 0:
            print( "\tdrawing output" )
            path = path +"k{:.4f}".format( score_kendall ) + ".png"
            if DRAWOUTPUTS:
                drawHeat( labels.to( torch.float32 ), predicted.to( torch.float32 ), path, g )
        memory_usage = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print("memory usage in evaluate:", memory_usage )
        return score_kendall, score_r2, f1

def evaluate_in_batches( dataloader, device, model, image_path = "" ):
    total_kendall = 0.0
    total_r2      = 0.0
    total_f1      = 0.0
    # names = dataloader.dataset.names
    # print("names in evaluate_in_batches:", names )
    for batch_id, batched_graph in enumerate( dataloader ):
        print( "batch_id (eval_in_batches):", batch_id )
        batched_graph = batched_graph.to( device )
        print( "batched_graph:", type( batched_graph ), batched_graph )
        print( "theDataset[ batch_id ]:", theDataset[ batch_id ] )
        features = batched_graph.ndata[ featName ].float().to( device )
        labels   = batched_graph.ndata[ labelName ].to( device )
        
#        print("features in evaluate_in_batches:", type(features), features.shape,"\n", features )
#        print("labels in evaluate_in_batches:", type(labels), labels.shape,"\n", labels )
        score_kendall, score_r2, f1 = evaluate( batched_graph, features, labels, model, image_path, device )
        print( "partial Kendall (eval_in_batches):", score_kendall, ", r2:", score_r2, ", batch_id:", batch_id )
        total_kendall += score_kendall
        total_r2      += score_r2
        total_f1      += f1
    total_kendall =  total_kendall / (batch_id + 1)
    total_r2      =  total_r2 / (batch_id + 1)
    total_f1      =  total_f1 / (batch_id + 1)
    return total_kendall, total_r2, total_f1 # return average score

def evaluate_single( graph, device, model, path ):
    total_kendall = 0.0
    total_r2      = 0.0
    graph = graph.to( device )
    features = graph.ndata[ featName ].float().to( device )
    labels   = graph.ndata[ labelName ].to( device )
    print( "evaluate single--->", path )                               
    score_kendall, score_r2, f1 = evaluate( graph, features, labels, model, path, device )
    print( "Single graph score - Kendall:", score_kendall, ", r2:", score_r2 )
    return score_kendall, score_r2, f1


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
            features = batched_graph.ndata[featName].float()
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
        #     features = batched_graph.ndata[featName].float()
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
        if DEBUG == 1:
            if ( epoch + 1 ) % 5 == 0:
                kendall, r2, f1 = evaluate_in_batches( valid_dataloader, device, model )
                print( "                            Kendall {:.4f} ". format( kendall ) )
                print( "                            R2      {:.4f} ". format( r2 ) )
                print( "                            F1      {:.4f} ". format( f1 ) )
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
        dsPath = Path.cwd() / 'dataSet'
        
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
    df.to_csv( "aggregatedDFBefore.csv" )
    df = df.drop( df.index[ df[ labelName ] < 0 ] )
    df.hist( bins = 50, figsize = (15,12) )
    plt.savefig( "BeforePreProcess-train+valid+test" )
    plt.close( 'all' )
    plt.clf()
    
    preProcessData( listDir )
	            
    writeDFrameData( listDir, 'preProcessedGatesToHeat.csv', "DSinfoAfterPreProcess.csv" )
    df = aggregateData( listDir, 'preProcessedGatesToHeat.csv' )
    print( "\n\n#######################\n## AFTER PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
    for col in df:
	    print( "\n####describe:\n", df[ col ].describe() )
    df.to_csv( "aggregatedDFAfter.csv" )
    df = df.drop( df.index[ df[ labelName ] < 0 ])
    df = df.drop( df.index[ df[ rawFeatName ] < 0 ] )
    df.hist( bins = 50, figsize = (15,12) )
    plt.savefig( "AfterPreProcess-train+valid+test" )
    plt.close( 'all' )
    plt.clf()
    ##################################################################################
    ##################################################################################
    ##################################################################################

    if not DOLEARN:
        sys.exit()
        
    summary = "runSummary.csv"
    ablationResult = "ablationResult.csv"
    with open( summary, 'w' ) as f:
        f.write("")
    with open( ablationResult, 'w' ) as f:
        f.write( "Features,Train-Mean,Train-SD,Valid-Mean,Valid-SD,Test-Mean,Test-SD\n" ) 
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
        for combAux in range( 1, len( listFeats ) + 1 ):
            print( "iteration:", combAux, ", combinations:", len( list( combinations( listFeats, combAux ) ) ) )
            ablationList += list( combinations( listFeats, combAux ) )
            print( "ablationList:", len( ablationList ), ablationList )
    else:
        #ablationList = [ ('type',) ]
         ablationList = [('eigen','pageRank','inDegree','outDegree'), ('pageRank','inDegree','outDegree'), ('eigen','pageRank','outDegree'), ('inDegree','outDegree'), ('pageRank','outDegree')]
        # ablationList =  [ ('inDegree','outDegree','eigen','pageRank', 'type' ) ]
        # ablationList += [ ('eigen',), ('pageRank',), ('inDegree',), ( 'outDegree',), ('type',) ]
        # ablationList += [ ('inDegree','outDegree'), ('inDegree','outDegree', 'type'), ('inDegree','outDegree', 'eigen'), ('inDegree','outDegree', 'pageRank') ]
        # ablationList += [ ('outDegree','eigen','pageRank'), ('inDegree','eigen','pageRank'), ('inDegree','pageRank'), ('outDegree','pageRank') ]
        # ablationList += [ ('inDegree','outDegree','eigen','pageRank'), ('inDegree','outDegree','eigen','pageRank','type'), ('type','eigen','pageRank'), ('eigen','pageRank') ]
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
                f.write( "#Circuits:" + str( len( listDir ) ) )
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
                f.write( "\ntrainIndices,testIndices,validIndices,finalEpoch,runtime(min),MaxMemory,AverageMemory,Circuit Valid, Circuit Test, TrainKendall, ValidKendall, TestKendall, TrainR2, ValidR2, TestR2, TrainF1, ValidF1, TestF1\n" )
            with open( ablationResult, 'a' ) as f:
                copied_list = [s[:1].capitalize() for s in ablationIter]
                f.write( "; ".join( copied_list ) )
            print( "ablationIter:", type( ablationIter ), len( ablationIter ), ablationIter, flush = True )
            ablationIter = list( ablationIter )
            if os.path.exists( imageOutput ):
                shutil.rmtree( imageOutput )
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
            kendallValid = []
            kendallTrain = []

            theDataset = DataSetFromYosys( listDir, ablationIter )#, mode='train' )
            kf = KFold( n_splits = num_folds )
            for fold, ( train_indices, test_indices ) in enumerate( kf.split( theDataset ) ):
############# MANUAL EXPERIMENTS #############
            # if True:
            #     fold = 0
                # LASCAS comparison, same as HUAWEI
                # train_indices = [i for i in range(len(theDataset)) if i !=2 and i !=4] # remove swerv and bp_be_top
                # test_indices = [2]

                # HUAWEI's ablation, only Black_parrot
                # train_indices = [7] 
                # test_indices = [i for i in range(len(theDataset)) if i != 7 and i != 4 ]  #remove bp_be_top and black_parrot
##############################################

                print(f"Fold {fold+1}/{num_folds}")
                #train_indices, valid_indices = train_indices[:-len(test_indices)], train_indices[-len(test_indices):]
                        
                startIterationTime = time.time()
                print( "##################################################################################" )
                print( "#################### New CrossValid iteration  ###################################" )
                print( "##################################################################################" )
                # if DRAWOUTPUTS:
                #     complete_dataset = train_dataset
                #     complete_dataset.appendGraph( val_dataset )
                #     complete_dataset.appendGraph( test_dataset )
                #     complete_dataset.drawCorrelationMatrices()
                #     del complete_dataset

                features = theDataset[0].ndata[ featName ]
                if( features.dim() == 1 ):
                    features = features.unsqueeze(1)
                in_size = features.shape[1]
                out_size = 1 #TODO parametrize this
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
                        test_kendall, test_r2, test_f1    = evaluate_in_batches( test_dataloader,  device, model )
                    else:
                        test_kendall = test_r2 = test_f1 = valid_kendall = valid_r2 = valid_f1 = torch.tensor([0])
                    train_kendall, train_r2, train_f1 = evaluate_in_batches( train_dataloader, device, model )

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
                    test_kendall= test_r2= test_f1= train_kendall= train_r2= train_f1= valid_kendall= valid_r2= valid_f1= torch.tensor([0])

                print( "Total Train Kendall {:.4f}".format( train_kendall ) )
                print( "Total Train R2 {:.4f}".format( train_r2 ) )
                print( "Total Train f1 {:.4f}".format( train_f1 ) )
                print( "\n###############################\n## FinalEvalRuntime:", round( ( time.time() - startTimeEval ) / 60, 1) , "min ##\n###############################\n" )
                iterationTime = round( ( time.time() - startIterationTime ) / 60, 1 )
                print( "\n###########################\n## IterRuntime:", iterationTime, "min ##\n###########################\n", flush = True )

                kendallTest.append ( test_kendall.item()  )
                kendallTrain.append( train_kendall.item() )
                with open( summary, 'a' ) as f:
                    f.write( str( train_indices ).replace(',', '') +","+ str( test_indices ).replace(',', ';') + ",,"+str( finalEpoch )+","+str( iterationTime )+","+str( maxMem )+","+str( avergMem / finalEpoch )+"," )
                    f.write( ","+ "; ".join( theDataset.getNames()[i] for i in test_indices ) +","+ str( train_kendall.item() ) +",,"+ str( test_kendall.item() ))
                    f.write( "," + str( train_r2.item() ) +",,"+ str( test_r2.item() ) )  #+"\n")
                    f.write( "," + str( train_f1.item() ) +",,"+ str( test_f1.item() )  +"\n")

                del model
                del train_dataloader
                del test_dataloader
                torch.cuda.empty_cache()
                if FULLTRAIN:
                    break
                    break
                # K fold loop end here
            with open( summary, 'a' ) as f:
                f.write( ",,,,,,,,Average," + str( sum( kendallTrain ) / len( kendallTrain ) ) +",,"+ str( sum( kendallTest ) / len( kendallTest ) ) + "\n" )
                f.write( ",,,,,,,,Median,"  + str( statistics.median( kendallTrain ) ) +",,"+ str( statistics.median( kendallTest ) ) +"\n" )
                f.write( ",,,,,,,,Std Dev," + ( str( statistics.stdev( kendallTrain ) ) if len( kendallTrain ) > 1 else "N/A" ) +",,"+ ( str( statistics.stdev( kendallTest ) ) if len( kendallTest ) > 1 else "N/A" ) +"\n" )
            with open( ablationResult, 'a' ) as f:
                f.write( ","+ str( sum( kendallTrain ) / len( kendallTrain ) ) +","+ ( str( statistics.stdev( kendallTrain ) ) if len( kendallTrain ) > 1 else "N/A" ) )
                f.write( ",,N/A") 
                f.write( ","+ str( sum( kendallTest ) / len( kendallTest ) )   +","+ ( str( statistics.stdev( kendallTest ) ) if len( kendallTest ) > 1 else "N/A" ) + "\n" )
                
            folder_name = f"{str(ablationIter)}-{mainIteration}"
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
    excluded_folders = ["dataSet", "backup", "c17", "gcd", "regression.py", ".git", "toyDataset", "asap7" ]
    
    for item in os.listdir():
        print( "item:", item )
        if not re.match( pattern, item ) and item not in excluded_folders:
            shutil.move( item, os.path.join( folder_name, item ) )
    shutil.copy( "regression.py", os.path.join( folder_name, "regression.py" ) )
    with open( 'log.log', 'w' ) as f:
        f.write('')
