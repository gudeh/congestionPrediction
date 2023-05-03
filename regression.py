import numpy as np
import os
import shutil
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from pathlib import Path #Reading CSV files
import math #ceil()
from random import shuffle #shuffle train/valid/test
import re # handle tpye to category names AND_X1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter #Graphical visualization
#from maskedtensor import masked_tensor

import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.ppi import PPIDataset #TODO remove
import dgl.nn as dglnn
from sklearn.metrics import r2_score, f1_score #Score metric
from torchmetrics.regression import KendallRankCorrCoef #Same score as congestionNet
import networkx as nx #drawing graphs


#regression metrics:  https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
#    mean_squared_error
#    mean_absolute_error
#    r2_score
#    explained_variance_score
#    mean_pinball_loss
#    d2_pinball_score
#    d2_absolute_error_score

#print( "torch.cuda.is_available():", torch.cuda.is_available() )
#print( "torch.cuda.device_count():", torch.cuda.device_count() )
#print( "torch.cuda.device(0):", torch.cuda.device(0) )
#print( "torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0) )
#print( "dgl.__version_:", dgl.__version__ )

listFeats = [ 'type' ] # , 'size' ]
featName = 'feat' #listFeats[0]
rawFeatName = 'type' #TODO: change to listFeats[0]

labelName =  'routingHeat'
secondLabel = 'placementHeat'

numEpochs = 500
step      = 0.005


DEBUG     = 2
CUDA      = True
DOLEARN   = True


SELF_LOOP = True
COLAB     = False



def preProcessData( listDir ):
    nameToCategory = {}
    typeToNameCat = {}
    typeToSize = {}
    labelToStandard = {}
    labelsAux = pd.Series( name = labelName )
    graphs  = {}
    #	regexp = re.compile( r'(^.*)\_X(\d?\d)$' ) #_X1 works ok for nangate
    regexp = re.compile( r'(^.*)\_?[x|xp](\d?\d)', re.IGNORECASE ) 
    for path in listDir:
	    #print( "Circuit:",path )
	    gateToHeat = pd.read_csv( path / 'gatesToHeat.csv', index_col = 'id', dtype = { 'type':'category' } )
	    labelsAux = pd.concat( [ labelsAux, gateToHeat[ labelName ] ], names = [ labelName ] )
	    graphs[ path ] = gateToHeat#.append( gateToHeat )
	    #Other category encoder possibilities: https://contrib.scikit-learn.org/category_encoders/
	    for cellType in gateToHeat[ rawFeatName ]:
    #            if cellType not in nameToCategory:
    #                nameToCategory[ cellType ] = len( nameToCategory )
		    
		    if "FILLCELL" not in cellType and "TAPCELL" not in cellType:
		        match = re.match( regexp, cellType )
		        #print( "len( match.groups( ) )", len( match.groups( ) ))
		        if match: 
			        if len( match.groups( ) ) == 2:
				        if match.group( 1 ) not in nameToCategory:
					        nameToCategory[ match.group( 1 ) ] =  len( nameToCategory ) + 1
				        #typeTo2Categories[ cellType ] = [ nameToCategory[ match.group( 0 ) ], match.group( 1 ) ]
				        if cellType not in typeToNameCat:
				            if len( listFeats ) == 2:
					            typeToNameCat[ cellType ] = nameToCategory[ match.group( 1 ) ]
					            typeToSize[ cellType ]    = int( match.group( 2 ) )
				            else:
				                typeToNameCat[ cellType ] = len( typeToNameCat )
		        else:
			        print( "WARNING: Unexpected cell type:", cellType )    
			        typeToNameCat[ cellType ] = -1
			        typeToSize[ cellType ]    = -1
		    else:
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
	    
    print( "\n\n\nnameToCategory:\n", nameToCategory, "size:", len( nameToCategory  ) )
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
    def __init__( self, listDir, split, mode='train' ):    
        if len( split ) != 3 or sum( split ) != 1.0:
	        print("!!!!ERROR: fatal, unexpected split sizes." )
	        return
	        
        self.graphPaths = []
        self.graphs = []
        self.names = []
        allNames = []
        self.mode = mode

        for idx in range( len( listDir ) ):
	        allNames.append( str( listDir[idx] ).rsplit( '/' )[-1] )
	        #print( allNames[idx],",", end="" )
    #        train, validate, test = np.split(files, [int(len(files)*0.8), int(len(files)*0.9)])
        firstSlice  = math.ceil( len( listDir )*split[0] ) - 1
        secondSlice = math.ceil( len( listDir )*split[1] + firstSlice )
        if( len( listDir ) == 1 ):
	        firstSlice  = 0
	        secondSlice = 0
        if( len( listDir ) == 2 ):
	        firstSlice  = 1
	        secondSlice = 1
        if( len( listDir ) > 2 and len( listDir ) < 10 ):
	        firstSlice  = -2
	        secondSlice = -1
         
        print( "\nlen(listDir)",len(listDir))
        print( "firstSlice:", firstSlice, "\nsecondSlice", secondSlice )
        if mode == 'train':
	        self.graphPaths = listDir [ : firstSlice ]
	        self.names      = allNames[ : firstSlice ]
        elif mode == 'valid':
	        self.graphPaths = listDir [ firstSlice: secondSlice ]
	        self.names      = allNames[ firstSlice: secondSlice ]
        elif mode == 'test':
	        self.graphPaths = listDir [ secondSlice : ]
	        self.names      = allNames[ secondSlice : ]

        super().__init__( name='mydata_from_yosys_'+mode )

    def process( self ):
        for path in self.graphPaths:
	        graph = self._process_single( path )
	        self.graphs.append( graph )

    def _process_single( self, designPath ):
        print( "\n\n########################################" )
        print( "Circuit:", str( designPath ).rsplit( '/' )[-1] )
        print( "#########################################\n" )
        nodes_data = pd.read_csv( designPath / 'preProcessedGatesToHeat.csv', index_col = 'id' )
        nodes_data = nodes_data.sort_index()
        edges_data = pd.read_csv( designPath / 'DGLedges.csv')
        edges_src  = torch.from_numpy( edges_data['Src'].to_numpy() )
        edges_dst  = torch.from_numpy( edges_data['Dst'].to_numpy() )

        df = nodes_data[ listFeats + [ labelName ] ]
        #print("np.where( df[ rawFeatName ] > 0, True, False ):\n", np.where( df[ rawFeatName ] > 0, True, False ).shape, "\n", np.where( df[ rawFeatName ] > 0, True, False ) )
        

#        df_wanted = np.logical_and ( np.where( df[ rawFeatName ] > 0, True, False ), np.where( df[ labelName ] >= 0, True, False ) )
        df_wanted = np.logical_and ( np.where( df[ rawFeatName ] > 0, True, False ), np.where( df[ labelName ] > 0, True, False ) ) # trying to remove also 0 heat
        df_wanted = np.invert( df_wanted )
    #		print( "df_wanted:", df_wanted.shape, "\n", df_wanted )
        removedNodesMask = torch.tensor( df_wanted )
    #		print( "removedNodesMask:", removedNodesMask.shape )#, "\n", removedNodesMask )
        print( "nodes_data:", nodes_data.shape ) #, "\n", nodes_data )
        idsToRemove = torch.tensor( nodes_data.index )[ removedNodesMask ]
        print( "idsToRemove:", idsToRemove.shape ) #"\n", torch.sort( idsToRemove ) )
        
        self.graph = dgl.graph( ( edges_src, edges_dst ), num_nodes = nodes_data.shape[0] )
        self.graph.ndata[ featName ] =  torch.tensor( nodes_data[ listFeats ].values )
        self.graph.ndata[ labelName  ]  = ( torch.from_numpy ( nodes_data[ labelName   ].to_numpy() ) )
        #self.graph.ndata[ secondLabel ] = ( torch.from_numpy ( nodes_data[ secondLabel ].to_numpy() ) )

        print( "---> BEFORE REMOVED NODES:")
        print( "\tself.graph.nodes()", self.graph.nodes().shape )#, "\n", self.graph.nodes() )
    ##		print( "\tself.graph.ndata\n", self.graph.ndata )		

        self.graph.remove_nodes( idsToRemove )
        
    ################################################################################################
        isolated_nodes = ( ( self.graph.in_degrees() == 0 ) & ( self.graph.out_degrees() == 0 ) ).nonzero().squeeze(1)
        print( "isolated_nodes:", isolated_nodes.shape ) #, "\n", isolated_nodes )
        self.graph.remove_nodes( isolated_nodes )
        if SELF_LOOP:
            self.graph = dgl.add_self_loop( self.graph ) 
    ################################################################################################
        
        print( "\n---> AFTER REMOVED NODES:" )
        print( "\tself.graph.nodes()", self.graph.nodes().shape ) #, "\n", self.graph.nodes() )
    #		print( "\tself.graph.ndata\n", self.graph.ndata )			

        return self.graph
           
    def __getitem__( self, i ):
        return self.graphs[i]

    def __len__( self ):
        #return 1
        return len( self.graphs )
        
    def printDataset( self ):
        totalNodes = 0
        totalEdges = 0
        print( "\n\n###", self.mode, "size:", len( self.graphs ) )
        for idx in range( len( self.graphs ) ):
            print( "\t>>>", idx," - ", self.names[idx], "\t\tV:", self.graphs[idx].num_nodes(), "E:", self.graphs[idx].num_edges() ) #, "\n", self.graphs[idx] )
            totalNodes += self.graphs[idx].num_nodes()
            totalEdges += self.graphs[idx].num_edges()    
        print( "Total Vertices:", totalNodes )
        print( "Total Edges:   ", totalEdges )
            #drawGraph( self.graphs[idx], self.names[idx] )

    def getNames( self ):
	    return self.names

    

class SAGE( nn.Module ):
    def __init__( self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv( in_feats = in_feats, out_feats = hid_feats, aggregator_type = 'lstm' )
        self.conv2 = dglnn.SAGEConv( in_feats = hid_feats, out_feats = out_feats, aggregator_type = 'lstm' )

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1( graph, inputs )
        h = F.relu(h)
        h = self.conv2( graph, h )
        return h


def drawGraph( graph, graphName ):
#    print("graph:",type(graph))
#    print('We have %d nodes.' % graph.number_of_nodes())
#    print('We have %d edges.' % graph.number_of_edges())
    nx_G = graph.to_networkx()
    pos = nx.kamada_kawai_layout( nx_G )
    plt.figure( figsize=[15,7] )
    nx.draw( nx_G, pos, with_labels = True, node_color = [ [ .7, .7, .7 ] ] )
    #	plt.show()
    plt.savefig( graphName )

#    print("len graph.ndata:",len(graph.ndata))
#    print("type graph.ndata:",type(graph.ndata))


def drawHeat( label, predict, drawHeatName ):

    sorted_pairs = sorted( zip( label, predict ) )
    sorted_label, sorted_predict = zip( *sorted_pairs )
    fig, axes = plt.subplots( nrows = 2, ncols = 1, figsize = ( 10, 10 ) )
    axes[0].plot( sorted_label )
    #axes[0].bar( range( len( sorted_label ) ), sorted_label )
    axes[0].set_title( 'Label' )
    axes[0].set_xlabel( 'Index' )
    axes[0].set_ylabel( 'Value' )

    axes[1].plot( sorted_predict )
    #axes[1].bar( range( len( sorted_predict ) ), sorted_predict )
    axes[1].set_title( 'Predicted' )
    axes[1].set_xlabel( 'Index' )
    axes[1].set_ylabel( 'Value' ) 

    plt.subplots_adjust( hspace=0.3 )
    auxName = drawHeatName.replace( "/", "/columns/", 1 )
    # first_slash = auxName.find('/')
    # if first_slash != -1:
    #     auxName[ :first_slash ] + "/columns/" + auxName[ first_slash+1: ]
    print( "auxName:", auxName )
    plt.savefig( auxName )
    plt.close( fig ) 


    vminLabel = np.min( label )
    vmaxLabel = np.max( label )
    vminPred = np.min( predict )
    vmaxPred = np.max( predict )
    
    size = int( np.ceil( np.sqrt( len( label ) ) ) )
    # sorted_label = np.pad( np.array( sorted_label ), ( 0, size**2 - len( label ) ), mode='constant' ).reshape( size, size )
    # sorted_predict = np.pad( np.array( sorted_predict ), ( 0, size**2 - len( predict ) ), mode='constant' ).reshape( size, size )
    label = np.pad( np.array( label ), ( 0, size**2 - len( label ) ), mode='constant', constant_values = vminLabel ).reshape( size, size )
    predict = np.pad( np.array( predict ), ( 0, size**2 - len( predict ) ), mode='constant', constant_values = vminPred ).reshape( size, size )
    # vmin = min( np.min( sorted_label ), np.min( sorted_predict ) )
    # vmax = max( np.max( sorted_label ), np.max( sorted_predict ) )
    fig, ax = plt.subplots( nrows = 1, ncols = 2, figsize = ( 10, 5 ) )

    im1 = ax[0].imshow( label, cmap='YlGnBu', vmin = vminLabel, vmax = vmaxLabel )
    ax[0].set_title( 'Labels' )
    ax[0].set_xlabel( 'Column Index' )
    ax[0].set_ylabel( 'Row Index' )
    fig.colorbar( im1, ax = ax[0] )

    im2 = ax[1].imshow( predict, cmap='YlGnBu', vmin = vminPred, vmax = vmaxPred )
    ax[1].set_title( 'Predicted' )
    ax[1].set_xlabel( 'Column Index' )
    ax[1].set_ylabel( 'Row Index' )
    fig.colorbar( im2, ax = ax[1] )

    plt.subplots_adjust( wspace=0.3 )
    auxName = drawHeatName.replace( "/", "/heatmaps/", 1 )
    # auxName = drawHeatName
    # first_slash = auxName.find("/")
    # if first_slash != -1:
    #     auxName[ : first_slash ] + "/heatmaps/" + auxName[ first_slash + 1 : ]
    print( "auxName:", auxName )
    plt.savefig( auxName )
    plt.close( fig )
   
    
	

class GAT( nn.Module ):
    def __init__( self, in_size, hid_size, out_size, heads ):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        #		self.gat_layers.append(dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu))
        #		self.gat_layers.append(dglnn.GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu))
        #		self.gat_layers.append(dglnn.GATConv(hid_size*heads[1], out_size, heads[2], residual=True, activation=None))
        print("\n\nINIT GAT!!")
        print("in_size:", in_size)
        print("hid_size:", hid_size)
        for k,head in enumerate( heads ):
            print("head:", k,head)
        print("out_size", out_size)
        # if SELF_LOOP:
        #     self.gat_layers.append( dglnn.GATConv( in_size, hid_size, heads[0], activation=F.elu )) #, allow_zero_in_degree=True ) )
        #     self.gat_layers.append( dglnn.GATConv( hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu )) #, allow_zero_in_degree=True ) )
        #     self.gat_layers.append( dglnn.GATConv( hid_size*heads[1], out_size, heads[2], residual=True, activation=None )) #, allow_zero_in_degree=True ) )
        # else:
        #     self.gat_layers.append( dglnn.GATConv( in_size, hid_size, heads[0], activation=F.elu, allow_zero_in_degree=True ) )
        #     self.gat_layers.append( dglnn.GATConv( hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu, allow_zero_in_degree=True ) )
        #     self.gat_layers.append( dglnn.GATConv( hid_size*heads[1], out_size, heads[2], residual=True, activation=None, allow_zero_in_degree=True ) )
        self.gat_layers.append( dglnn.GATConv( in_size, hid_size, heads[0], activation = F.relu, allow_zero_in_degree = not SELF_LOOP ))# , feat_drop = 0.05, attn_drop = 0.05 ) )
        self.gat_layers.append( dglnn.GATConv( hid_size*heads[0], hid_size, heads[1], residual=True, activation = F.relu, allow_zero_in_degree = not SELF_LOOP )) #, feat_drop = 0.5, attn_drop = 0.5 ) )
        # self.gat_layers.append( dglnn.GATConv( hid_size*heads[1], hid_size, heads[1], residual=True, activation=F.relu, allow_zero_in_degree = not SELF_LOOP ) )
        self.gat_layers.append( dglnn.GATConv( hid_size*heads[1], out_size, heads[2], residual=True, activation=None, allow_zero_in_degree = not SELF_LOOP ) )
        
    def forward( self, g, inputs ):
        h = inputs
        for i, layer in enumerate( self.gat_layers ):
            h = layer( g, h )
            if i == 2:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
        return h

 
def evaluate( g, features, labels, model, imageName ):
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
            
        score_r2 = r2_score( labels.data.cpu(), predicted.data.cpu() )
        kendall = KendallRankCorrCoef( variant = 'a' )
        score_kendall = kendall( predicted, labels )
#        print("score_kendall:", type( score_kendall ), str( score_kendall ), "\n", score_kendall,"\n\n")
        if len( imageName ) > 0:
            imageName = imageName +"k{:.4f}".format( score_kendall ) + ".png"
            drawHeat( list( labels.data.cpu() ), list( predicted.data.cpu() ), imageName )
        return score_kendall, score_r2

def evaluate_in_batches( dataloader, device, model ):
    total_kendall = 0.0
    total_r2      = 0.0
    # names = dataloader.dataset.names
    # print("names in evaluate_in_batches:", names )
    for batch_id, batched_graph in enumerate( dataloader ):
        batched_graph = batched_graph.to( device )
        features = batched_graph.ndata[ featName ].float() #TODO remove this float and make work.
        labels   = batched_graph.ndata[ labelName ]
        
#        print("features in evaluate_in_batches:", type(features), features.shape,"\n", features )
#        print("labels in evaluate_in_batches:", type(labels), labels.shape,"\n", labels )
        score_kendall, score_r2 = evaluate( batched_graph, features, labels, model, "" )
        print( "partial Kendall:", score_kendall, ", r2:", score_r2, ", batch_id:", batch_id )
        total_kendall += score_kendall
        total_r2      += score_r2
    total_kendall =  total_kendall / (batch_id + 1)
    total_r2      =  total_r2 / (batch_id + 1)
    return total_kendall, total_r2 # return average score


def evaluate_single( graph, device, model, name ):
    total_kendall = 0.0
    total_r2      = 0.0
    features = g.ndata[ featName ].float() #TODO remove this float and make work.
    labels   = g.ndata[ labelName ]
    print( "evaluate--->", name )
    score_kendall, score_r2 = evaluate( graph, features, labels, model, name )
    print( "Single graph score - Kendall:", score_kendall, ", r2:", score_r2 )
    return score_kendall, score_r2

def train( train_dataloader, val_dataloader, device, model, writerName ):
    print( "device in train:", device )
    writer = SummaryWriter( comment = writerName )
    
    #loss_fcn = nn.BCEWithLogitsLoss()
#    loss_fcn = nn.CrossEntropyLoss()
#    loss_fcn = nn.L1Loss() 
    
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters(), lr = step, weight_decay = 0 )
    
    # training loop
    for epoch in range( numEpochs ):
        model.train()
        logits = []
        total_loss = 0
        # mini-batch loop
        for batch_id, batched_graph in enumerate( train_dataloader ):
            batched_graph = batched_graph.to(device)
#            print( "->batched_graph", type( batched_graph ), batched_graph )
#            print("\t%%%% Batch ID ", batch_id )
            features = batched_graph.ndata[ featName ].float()
#            removedNodesMask = batched_graph.ndata[ 'removedNodesMask' ]
            if( features.dim() == 1 ):
#                print("\n\n\nUNSQUEZING\n\n\n")
                features = features.float().unsqueeze(1)
            #features = features[ mask ]
            #print("->features in train:",type(features),features.shape,"\n", features.dtype)
            #print("\n",features)

            logits = model( batched_graph, features )
            labels = batched_graph.ndata[ labelName ].float()
            if( labels.dim() == 1 ): # required if, don't know why shape dont match
                labels = labels.unsqueeze(-1)

#            print("->labels in train:",type(labels),labels.shape)
#            print("\n",labels)
            loss = loss_fcn( logits, labels )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                        
        if DEBUG > 0 :
            print("Epoch {:05d} | Loss {:.4f} |". format( epoch, total_loss / (batch_id + 1) ))
            if ( epoch + 1 ) % 5 == 0:
                print( "                            Kendall {:.4f} ". format( kendall ) )
                print( "                            R2      {:.4f} ". format( r2 ) )
        if DEBUG == 2:
            writer.add_scalar( "Loss Train", total_loss / (batch_id + 1), epoch )
            kendall, r2 = evaluate_in_batches( val_dataloader, device, model )
            writer.add_scalar( "Score TEST Kendall", kendall, epoch )
            kendall, r2 = evaluate_in_batches( train_dataloader, device, model )
            writer.add_scalar( "Score TRAIN Kendall", kendall, epoch )
            
        
    writer.flush()
    writer.close()


if __name__ == '__main__':
    print(f'Training Yosys Dataset with DGL built-in GATConv module.')
    
    if CUDA:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')


    if os.path.exists( "runs" ):
        shutil.rmtree( "runs" )
    imageOutput = "image_outputs"
    if os.path.exists( imageOutput ):
        shutil.rmtree( imageOutput )
    os.makedirs( imageOutput )
    aux = imageOutput + "/columns"
    os.makedirs( aux )
    aux = imageOutput + "/heatmaps"
    os.makedirs( aux )

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

    #df.plot( kind = "scatter",  x = "placementHeat", y = "type" )
    #    df.plot.area( figsize = (15,12), subplots = True )
    #    .savefig( "scatterPlacement" )
    ################################################################################
    ################################################################################    


    if DOLEARN:
        summary = "runSummary.csv"
        with open( summary, 'w' ) as f:
            f.write( "#Circuits:" + str( len( listDir ) ) )
            f.write( ",epochs:" + str( numEpochs ) )
            f.write( ",labelName:" + labelName )
            f.write( ",features:" ) 
            f.write( ";".join( listFeats ) )
            f.write( "\ni,j,Circuit Valid, Circuit Test, TrainKendall, ValidKendall, TestKendall, TrainR2, ValidR2, TestR2\n" )
	    
        split = [ 0.9, 0.05, 0.05 ]
        #shuffle( listDir )
        # for i in range( 0, len( listDir ) - 1 ):
        #     for j in range( 2, len( listDir ) - 1 ):
        #         if i == j:
        #             continue
        #         if j > 2:
        #             continue
        for i in range( 0, len( listDir ) ):
            for j in range( 2, 3 ):
                print( "##################################################################################" )
                print( "############################# New Run ############################################" )
                print( "##################################################################################" )
                currentDir = listDir.copy()

                auxString = currentDir[ -1 ]
                currentDir[ -1 ] = currentDir[ i ]
                currentDir[ i ] = auxString

                auxString = currentDir[ -2 ]
                currentDir[ -2 ] = currentDir[ j ]
                currentDir[ j ] = auxString

                train_dataset = DataSetFromYosys( currentDir, split, mode='train' )
                val_dataset   = DataSetFromYosys( currentDir, split, mode='valid' )
                test_dataset  = DataSetFromYosys( currentDir, split, mode='test'  )

                features = train_dataset[0].ndata[ featName ]      
                if( features.dim() == 1 ):
                    features = features.unsqueeze(1)

                #features = torch.cat( [train_dataset[0].ndata[ featName ][:,None]], dim=1 ) #TODO sometimes shape is unidimension
                in_size = features.shape[1]
                print( "features.shape", features.shape )

                #	node_labels = train_dataset[0].ndata[ labelName ].float()
                #	node_labels[ node_labels == -1 ] = 0
                #	out_size = int(node_labels.max().item() + 1)
                #    out_size = train_dataset.num_labels
                out_size = 1 #TODO parametrize this

                print( "in_size", in_size,",  out_size", out_size )
                # model = GAT( in_size, 256, out_size, heads=[4,4,6]).to( device )
                model = SAGE( in_feats = in_size, hid_feats = 100, out_feats = out_size ).to( device )

                print( "\n###################" )
                print( "## MODEL DEFINED ##"   )
                print( "###################\n" )

#                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
                # train_dataloader = GraphDataLoader( train_dataset, batch_size=1 )
                # val_dataloader   = GraphDataLoader( val_dataset,   batch_size=1 )
                # test_dataloader  = GraphDataLoader( test_dataset,  batch_size=1 )

                train_dataloader = GraphDataLoader( train_dataset, batch_size=1 )
                val_dataloader   = GraphDataLoader( val_dataset,   batch_size=1 )
                test_dataloader  = GraphDataLoader( test_dataset,  batch_size=1 )

                print( "\n###################"   )
                print( "### SPLIT INFO ####"   )
                print( "###################"   )
                train_dataset.printDataset()    
                val_dataset.printDataset()    
                test_dataset.printDataset()  
                print( "->original:   ", end="" )
                for item in listDir:
                    print( str( item ).rsplit( '/' )[-1], end=", " )
                print( "\n" )
                print( "->currentDir: ", end="" )
                for item in currentDir:
                    print( str( item ).rsplit( '/' )[-1], end=", " )
                print( "\n" )
                print( "i:", i, "j:", j )
                print( "split lengths:", len( train_dataset ), len( val_dataset ), len( test_dataset ) )

                writerName = "-" + labelName +"-"+ str( len(train_dataset) ) +"-"+ str( len(val_dataset) ) +"-"+ str( len(test_dataset) ) + "-V-"+ val_dataset.getNames()[0] +"-T-"+ test_dataset.getNames()[0]
                train( train_dataloader, test_dataloader, device, model, writerName )

                print( '######################\n## Final Evaluation ##\n######################\n' )
                for k in range( len( train_dataset ) ):
                    g = train_dataset[k].to( device )
                    name = train_dataset.names[k]
                    name = imageOutput + "/train-" + name +"-i"+ str(i)+"j"+ str(j)
                    print( "))))))) executing single evaluation on ", name, "-", k )
                    train_kendall, train_r2 = evaluate_single( g, device, model, name )
                    print( "Single Train Kendall {:.4f}".format( train_kendall ) )
                    print( "Single Train R2 {:.4f}".format( train_r2 ) )
                g = val_dataset[0].to( device )
                name = val_dataset.names[0]
                name = imageOutput + "/valid-" + name +"-i"+ str(i)+"j"+ str(j)
                valid_kendall, valid_r2 = evaluate_single ( g, device, model, name )

                print( "Single valid Kendall {:.4f}".format( valid_kendall ) )
                print( "Single valid R2 {:.4f}".format( valid_r2 ) )
                g = test_dataset[0].to( device )
                name = test_dataset.names[0]
                name = imageOutput + "/test-" + name +"-i"+ str(i)+"j"+ str(j)
                test_kendall, test_r2   = evaluate_single ( g, device, model, name )

                #TODO change this train to new loop with single0
                train_kendall, train_r2 = evaluate_in_batches( train_dataloader, device, model )
                print( "Total Train Kendall {:.4f}".format( train_kendall ) )
                print( "Total Train R2 {:.4f}".format( train_r2 ) )
                
                # test_kendall, test_r2   = evaluate_in_batches( test_dataloader, device, model )
                # valid_kendall, valid_r2 = evaluate_in_batches( val_dataloader, device, model )
                # print( "Valid Kendall {:.4f}".format( valid_kendall ) ) #
                # print( "Test Kendall {:.4f}".format( test_kendall ) )
                # print( "Valid R2 {:.4f}".format( valid_r2 ) )
                # print( "Test R2 {:.4f}".format( test_r2 ) )

                with open( summary, 'a' ) as f:
                    f.write( str(i) + ","+ str(j) +","+ val_dataset.getNames()[0] +","+ test_dataset.getNames()[0] +","+ str( train_kendall.item() ) +","+ str( valid_kendall.item() ) +","+ str( test_kendall.item() )) #  +"\n")
                    f.write( "," + str( train_r2.item() ) +","+ str( valid_r2.item() ) +","+ str( test_r2.item() )  +"\n")
                     			
                del model
                del train_dataset
                del val_dataset
                del test_dataset
                del train_dataloader
                del val_dataloader
                del test_dataloader

