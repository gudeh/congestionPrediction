import numpy as np
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from pathlib import Path #Reading CSV files
import math #ceil()
from random import shuffle #shuffle train/valid/test

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter #Graphical visualization

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

featName = 'type' #'feat'
#labelName = 'placementHeat' #'label'
labelName = 'routingHeat' #'label'
secondLabel = 'placementHeat'
#featName = 'feat'
#labelName = 'label'


def preProcessData( listDir ):
    nameToCategory = {}
    labelToStandard = {}
    labelsAux = pd.Series( name = labelName )
    graphs  = {}
    for path in listDir:
        print( "\n\n@@@@@@@@@@@@@@\n Circuit:",path,"\n@@@@@@@@@@@@@@@@\n\n" )
        gateToHeat = pd.read_csv( path / 'gatesToHeat.csv', index_col = 'id', dtype = { 'type':'category' } )
        #Other category encoder possibilities: https://contrib.scikit-learn.org/category_encoders/
        for cellType in gateToHeat[ featName ]:
            if cellType not in nameToCategory:
                nameToCategory[ cellType ] = len( nameToCategory )
        labelsAux = pd.concat( [ labelsAux, gateToHeat[ labelName ] ], names = [ labelName ] )
        graphs[ path ] = gateToHeat#.append( gateToHeat )
        
    df = labelsAux     
    print( "df before remove -1:\n", type(df), "\n", df,"\n")
#    df = df.drop( df[ df < 0 ] )
    df = df.loc[ df >= 0 ]
    print( "df after remove -1:\n", df,"\n")
    dfMin = float( df.min() )
    dfMax = float( df.max() )
    for key in df:
        #key = round( label, 4 ) # float as key of dict, possible danger
        #print( "label:",label,"key:",key )
        if key not in labelToStandard: # and key >= 0:
            labelToStandard[ key ] = ( key - dfMin ) / ( dfMax - dfMin )# 0 to 1
#                labelToStandard[ key ] = ( key - series.mean() ) / series.std() # z value
            print( "key:", key, "val:", labelToStandard[ key ] )
#            else:
#                print( "key:", key, "already set!" )
    print( "dfMin:", dfMin )
    print( "dfMax:", dfMax )
        
    print( "\n\n\nnameToCategory:\n", nameToCategory, "size:", len( nameToCategory  ) )
    print( "\n\nlabelToStandard:\n", sorted( labelToStandard.items(), key=lambda x:x[1] ), "size:", len( labelToStandard ) )
    for key, g in graphs.items():
        g[ featName ]  = g[ featName  ].cat.rename_categories( nameToCategory )
        g[ labelName ] = g[ labelName ].replace( labelToStandard )
        #print("\n->g:\n",g)
#        print( "\n\n###########\n###########\ng[ featName ]:\n", g[ featName ] )
#        print( "\ng[ labelName ]:\n", g[ labelName ] )
        g.to_csv( key / 'preProcessedGatesToHeat.csv' )
        
        
     
def aggregateData( listDir, csvName ):
    aggregatedDF = pd.DataFrame()
    for path in listDir:
        inputData = pd.read_csv( path / csvName )#, index_col = 'type' )        
        inputData = inputData[ [ featName, labelName, secondLabel ] ]
#        print( "inputData before concat:\n", inputData )
        aggregatedDF = pd.concat( [ aggregatedDF, inputData ] )
#    aggregatedDF.set_index( 'type' )
    return aggregatedDF


class DataSetFromYosys( DGLDataset ):
    
    def __init__( self, listDir, split, mode='train' ):    
        if len( split ) != 3 or sum( split ) != 1.0:
            print("!!!!ERROR!!!!")
            
        self.graphPaths = []
        self.graphs = []
        self.names = []
        allNames = []
        self.mode = mode

        for idx in range( len( listDir ) ):
            allNames.append( str( listDir[idx] ).rsplit( '/' )[-1] )
            print( allNames[idx],",", end="" )
#        train, validate, test = np.split(files, [int(len(files)*0.8), int(len(files)*0.9)])
        firstSlice  = math.ceil( len( listDir )*split[0] ) - 1
        secondSlice = math.ceil( len( listDir )*split[1] + firstSlice ) 
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
#        print("designPath in _process_single:", designPath )
        nodes_data = pd.read_csv( designPath / 'preProcessedGatesToHeat.csv' )
        edges_data  = pd.read_csv( designPath / 'DGLedges.csv')
#        edges_data = pd.read_csv( designPath / 'preProcessedDGLedges.csv')
        edges_src  = torch.from_numpy( edges_data['Src'].to_numpy() )
        edges_dst  = torch.from_numpy( edges_data['Dst'].to_numpy() )

        self.graph = dgl.graph( ( edges_src, edges_dst ), num_nodes = nodes_data.shape[0] ) #TODO int problem fix here
        self.graph.ndata[ featName ] = ( torch.from_numpy( nodes_data[ featName ].to_numpy() ) ).int()
        
        #self.graph.ndata['conCount'] = torch.from_numpy(nodes_data['conCount'].to_numpy())        #TODO possible feature, needs fix in yosys

        self.graph.ndata[ labelName  ]  = ( torch.from_numpy ( nodes_data[ labelName   ].to_numpy() ) )#.float()
        self.graph.ndata[ secondLabel ] = ( torch.from_numpy ( nodes_data[ secondLabel ].to_numpy() ) )#.float()        
#        self.graph.ndata['powerHeat'] = torch.from_numpy (nodes_data['powerHeat'].to_numpy()).float()
#        self.graph.ndata['irDropHeat'] = torch.from_numpy (nodes_data['irDropHeat'].to_numpy()).float()


        removedNodesMask = torch.where( self.graph.ndata[ labelName ] != -1, True, False) #torch.zeros(n_nodes, dtype=torch.bool)
#        val_mask = #torch.zeros(n_nodes, dtype=torch.bool)
#        test_mask = #torch.zeros(n_nodes, dtype=torch.bool)

#        val_mask[n_train : n_train + n_val] = True
#        test_mask[n_train + n_val :] = True
        self.graph.ndata['removedNodesMask'] = removedNodesMask
#        self.graph.ndata["val_mask"] = val_mask
#        self.graph.ndata["test_mask"] = test_mask
        return self.graph
           
    def __getitem__( self, i ):
	    return self.graphs[i]

    def __len__( self ):
	    #return 1
	    return len( self.graphs )
	    
    def printDataset( self ):
        print( "\n\n", self.mode, "size:", len( self.graphs ) )
        for idx in range( len( self.graphs ) ):
            print( "\t>>>", idx," - ", self.names[idx] ) #, "\n", self.graphs[idx] )
            #drawGraph( self.graphs[idx], self.names[idx] )

    

class SAGE( nn.Module ):
	def __init__(self, in_feats, hid_feats, out_feats):
		super().__init__()
		self.conv1 = dglnn.SAGEConv( in_feats=in_feats, out_feats=hid_feats, aggregator_type='lstm' )
		self.conv2 = dglnn.SAGEConv( in_feats=hid_feats, out_feats=out_feats, aggregator_type='lstm' )

	def forward(self, graph, inputs):
		# inputs are features of nodes
		h = self.conv1( graph, inputs )
		h = F.relu(h)
		h = self.conv2( graph, h )
		return h


def drawGraph(graph, graphName):
#    print("graph:",type(graph))
#    print('We have %d nodes.' % graph.number_of_nodes())
#    print('We have %d edges.' % graph.number_of_edges())
    nx_G = graph.to_networkx()
    pos = nx.kamada_kawai_layout(nx_G)
    plt.figure(figsize=[15,7])
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    #	plt.show()
    plt.savefig(graphName)

#    print("len graph.ndata:",len(graph.ndata))
#    print("type graph.ndata:",type(graph.ndata))

	

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
		self.gat_layers.append( dglnn.GATConv( in_size, hid_size, heads[0], activation=F.elu, allow_zero_in_degree=True ) )
		self.gat_layers.append( dglnn.GATConv( hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu, allow_zero_in_degree=True ) )
		self.gat_layers.append( dglnn.GATConv( hid_size*heads[1], out_size, heads[2], residual=True, activation=None, allow_zero_in_degree=True ) )

	def forward( self, g, inputs ):
		h = inputs
		for i, layer in enumerate( self.gat_layers ):
			h = layer( g, h )
			if i == 2:  # last layer 
				h = h.mean(1)
			else:       # other layer(s)
				h = h.flatten(1)
		return h

 
def evaluate( g, features, labels, model ):
    model.eval()
    with torch.no_grad():
#        print(">>>features in evaluate:",type(features),features.shape)
#        print("\n",features,"\n")
#        print(">>> g in evaluate:",type(g),"\n",g,"\n")
        if( features.dim() == 1 ):
            features = features.unsqueeze(1)
        output = model( g, features )
        #print("++++ output:",type(output),"\n",output.shape,"\n",output.transpose)
#        pred = np.where( output.data.cpu().numpy() >= 0, 1, 0) #this is not like this
##        print("++++ pred:",type(pred),"\n",pred.shape,"\n", np.transpose(pred ))
#        score = f1_score( labels.data.cpu().numpy(), pred, average='micro' )

#        score = r2_score( labels.data.cpu().numpy(), output )
        
        kendall = KendallRankCorrCoef()
        score = kendall( labels.data.cpu(), output.squeeze(1) )
#        print( "kendall:", score )
        return score

def evaluate_in_batches(dataloader, device, model):
    total_score = 0
    for batch_id, batched_graph in enumerate(dataloader):
        removedNodesMask = batched_graph.ndata[ 'removedNodesMask' ]
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata[ featName ].float()
        labels = batched_graph.ndata[ labelName ] #.float()
#        print("features in evaluate_in_batches:", type(features), features.shape,"\n", features )
#        print("labels in evaluate_in_batches:", type(labels), labels.shape,"\n", labels )
        score = evaluate( batched_graph, features, labels, model )
        total_score += score
    return total_score / (batch_id + 1) # return average score


def train( train_dataloader, val_dataloader, device, model, writerName ):
    print( "device in train:", device )
    writer = SummaryWriter( comment = writerName )
    
    #loss_fcn = nn.BCEWithLogitsLoss()
#    loss_fcn = nn.CrossEntropyLoss()
#    loss_fcn = nn.L1Loss() 
    
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters(), lr=0.005, weight_decay=0 )
    
    
    # training loop
    for epoch in range( 150 ):
        model.train()
        logits = []
        total_loss = 0
        # mini-batch loop
        for batch_id, batched_graph in enumerate( train_dataloader ):
            batched_graph = batched_graph.to(device)
#            print("->batched_graph",batched_graph)			
#            print("\t%%%% Batch ID ", batch_id )
            features = batched_graph.ndata[ featName ].float()
            removedNodesMask = batched_graph.ndata[ 'removedNodesMask' ]
            if( features.dim() == 1 ):
#                print("\n\n\nUNSQUEZING\n\n\n")
                features = features.float().unsqueeze(1)
            #print("->features in train:",type(features),features.shape,"\n", features.dtype)
            #print("\n",features)

            logits = model( batched_graph, features )
            labels = batched_graph.ndata[ labelName ].float()
            if( labels.dim() == 1 ): # required if, don't know why shape dont match
                labels = labels.unsqueeze(-1)

#            print("->labels in train:",type(labels),labels.shape)
#            print("\n",labels)
#            loss = loss_fcn( logits, labels )
            loss = loss_fcn( logits[ removedNodesMask ], labels[ removedNodesMask ] )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_score = evaluate_in_batches( val_dataloader, device, model )
        writer.add_scalar( "Loss Train", total_loss / (batch_id + 1), epoch )
        writer.add_scalar( "Score Valid", avg_score, epoch )
        print("Epoch {:05d} | Loss {:.4f} |". format(epoch, total_loss / (batch_id + 1) ))
        
        if (epoch + 1) % 5 == 0:
            #avg_score = evaluate_in_batches( val_dataloader, device, model) # evaluate r2-score instead of loss
            print("                            Acc. (r2-score) {:.4f} ". format(avg_score))
    writer.flush()
    writer.close()


if __name__ == '__main__':
    print(f'Training Yosys Dataset with DGL built-in GATConv module.')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

#    train_dataset = PPIDataset(mode='train')
#    val_dataset = PPIDataset(mode='valid')
#    test_dataset = PPIDataset(mode='test')
    listDir = []	
    for designPath in Path( Path.cwd() ).iterdir():
        if designPath.is_dir() and "runs" not in str( designPath ):
            listDir.append( designPath )
################################################################################
################################################################################
    if( featName != 'feat' ):
        df = aggregateData( listDir, 'gatesToHeat.csv' )
        df = df.drop( df.index[ df[ labelName ] < 0 ] )
    print( "\n\n#######################\n## BEFORE PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
    for col in df:
        print( "describe:\n", df[ col ].describe() )
    df.to_csv( "aggregatedDFBefore.csv" )
    df.hist( bins = 50, figsize = (15,12) )
    plt.savefig( "BeforePreProcess-train+valid+test" )


    if( featName != 'feat' ):
        preProcessData( listDir )
                

    if( featName != 'feat' ):
        df = aggregateData( listDir, 'preProcessedGatesToHeat.csv' )
        df = df.drop( df.index[ df[ labelName ] < 0 ])
    print( "\n\n#######################\n## AFTER PRE PROCESS ##\n####################### \n\nallDFs:\n", df )
    for col in df:
        print( "describe:\n", df[ col ].describe() )
    df.to_csv( "aggregatedDFAfter.csv" )
    df.hist( bins = 50, figsize = (15,12) )
    plt.savefig( "AfterPreProcess-train+valid+test" )
    
    #df.plot( kind = "scatter",  x = "placementHeat", y = "type" )
#    df.plot.area( figsize = (15,12), subplots = True )
#    plt.savefig( "scatterPlacement" )
################################################################################
################################################################################    
    
    shuffle( listDir )
    split = [ 0.8, 0.1, 0.1 ]
    train_dataset = DataSetFromYosys( listDir, split, mode='train' )
    val_dataset   = DataSetFromYosys( listDir, split, mode='valid' )
    test_dataset  = DataSetFromYosys( listDir, split, mode='test'  )

#    train_dataset.printDataset()    
#    val_dataset.printDataset()    
#    test_dataset.printDataset()    

    features = train_dataset[0].ndata[ featName ]      
    if( features.dim() == 1 ):
        features = features.unsqueeze(1)
    
    #features = torch.cat( [train_dataset[0].ndata[ featName ][:,None]], dim=1 ) #TODO sometimes shape is unidimension
    in_size = features.shape[1]
    print("features.shape",features.shape)

    #	node_labels = train_dataset[0].ndata[ labelName ].float()
    #	node_labels[ node_labels == -1 ] = 0
    #	out_size = int(node_labels.max().item() + 1)
#    out_size = train_dataset.num_labels
    out_size = 1 #TODO parametrize this

    print("in_size",in_size,",  out_size",out_size)
    model = GAT(in_size, 256, out_size, heads=[4,4,6]).to(device)
    #model = SAGE( in_feats = in_size, hid_feats=100, out_feats = out_size )

    print( "\n###################"   )
    print( "\n## MODEL DEFINED ##"   )
    print( "\n###################\n" )
    
    train_dataloader = GraphDataLoader( train_dataset, batch_size=2 )
    val_dataloader   = GraphDataLoader( val_dataset,   batch_size=1 )
    test_dataloader  = GraphDataLoader( test_dataset,  batch_size=1 )
    #	train_dataloader = ppi_dataloader
    #	val_dataloader = ppi_dataloader
    #print( "\n\n-->train_dataloader", type(train_dataloader),"\n", train_dataloader)
    for batch_id, batched_graph in enumerate( train_dataloader ):
	    batched_graph = batched_graph.to(device)
	    print("batch_id",batch_id)#"->batched_graph",type(batched_graph),"\n",batched_graph)


    print( "split lengths:", len(train_dataset), len(val_dataset), len(test_dataset) )
    writerName = str( len(train_dataset) ) + "-" + str( len(val_dataset) ) + "-" + str( len(test_dataset) )
    train( train_dataloader, val_dataloader, device, model, writerName )

    # test the model
    print('Testing...')    
    avg_score = evaluate_in_batches( test_dataloader, device, model )
    #	print("Test Accuracy (F1-score) {:.4f}".format(avg_score))
    print("Test Accuracy (r2-score) {:.4f}".format(avg_score))


